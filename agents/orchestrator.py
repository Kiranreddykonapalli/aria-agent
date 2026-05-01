"""
Orchestrator: runs the full multi-agent analytics pipeline.

Receives a CSV path and a natural-language question, then dispatches work
to specialist agents in sequence:

    DataWrangler -> Analyst -> VizBuilder -> ReportWriter

Control flow lives entirely here. Specialist agents are stateless and never
call each other directly.

Each step prints a live status line so the user can see progress. Any agent
failure is caught, wrapped in a clear error dict, and returned rather than
propagating a raw exception.
"""

from __future__ import annotations

import sys
import traceback

from .quality_agent import QualityAgent
from .data_wrangler import DataWrangler
from .analyst import Analyst
from .anomaly_agent import AnomalyAgent
from .decision_agent import DecisionAgent
from .forecasting_agent import ForecastingAgent
from .stats_agent import StatsAgent
from .viz_builder import VizBuilder
from .report_writer import ReportWriter

# Width of the status divider line
_DIV = "-" * 60


def _status(msg: str) -> None:
    """Print a flush-forced status line so CI / redirected output sees it."""
    print(msg, flush=True)


class Orchestrator:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.quality_agent    = QualityAgent(model=model)
        self.data_wrangler    = DataWrangler()
        self.analyst          = Analyst(model=model)
        self.anomaly_agent    = AnomalyAgent(model=model)
        self.decision_agent   = DecisionAgent(model=model)
        self.forecasting_agent = ForecastingAgent(model=model)
        self.stats_agent       = StatsAgent(model=model)
        self.viz_builder       = VizBuilder()
        self.report_writer    = ReportWriter(model=model)

    #: Fallback question used when the user provides none.
    DEFAULT_QUESTION = (
        "Perform a comprehensive exploratory analysis of this dataset. "
        "Identify the most interesting patterns, trends, outliers, and relationships. "
        "What are the most important findings a business leader should know?"
    )

    def run(self, data_path: str, question: str = "") -> dict:
        question = question.strip() or self.DEFAULT_QUESTION
        """
        Execute the full pipeline for a user question against a CSV file.

        Args:
            data_path: Path to the raw CSV file.
            question:  Natural-language question about the dataset.

        Returns:
            dict with keys:
              - "report_path":         str        — path to saved .md report
              - "report_text":         str        — full markdown report text
              - "figure_paths":        list[str]  — paths to saved chart PNGs
              - "data_quality_report": dict       — DataWrangler quality summary
              - "insights":            list[str]  — 5 analyst findings
              - "charts_rendered":     int
              - "error":               str | None — set if a stage failed
        """
        result: dict = {
            "report_path":         None,
            "report_text":         None,
            "figure_paths":        [],
            "data_quality_report": {},
            "insights":            [],
            "charts_rendered":     0,
            "anomalies":           [],
            "anomaly_narrative":   "",
            "severity_counts":     {"high": 0, "medium": 0, "low": 0},
            "decisions":           [],
            "decision_summary":    "",
            "decision_domain":     "",
            "forecasts":           [],
            "forecast_narrative":  "",
            "forecast_figures":    [],
            "stats_tests":         [],
            "stats_significant":   [],
            "stats_narrative":     "",
            "stats_recommendations": [],
            "quality_score":       0.0,
            "quality_grade":       "",
            "quality_dimensions":  {},
            "quality_verdict":     "",
            "quality_recommendations": [],
            "error":               None,
        }

        # ── Stage 0: Quality Agent ───────────────────────────────────
        _status("\n" + _DIV)
        _status("  Quality Agent  —  scoring raw data quality")
        _status(_DIV)
        try:
            import pandas as _pd
            raw_df       = _pd.read_csv(data_path)
            quality_output = self.quality_agent.run(raw_df, wrangler_output=None)
            result["quality_score"]        = quality_output["overall_score"]
            result["quality_grade"]        = quality_output["grade"]
            result["quality_dimensions"]   = quality_output["dimension_scores"]
            result["quality_verdict"]      = quality_output["verdict"]
            result["quality_recommendations"] = quality_output["recommendations"]
            _status(
                f"  OK  Aria Score {quality_output['overall_score']:.0f}/100 "
                f"(Grade {quality_output['grade']})"
            )
        except Exception:
            return self._fail(result, "Quality Agent", traceback.format_exc())

        # ── Stage 1: Data Wrangler ────────────────────────────────────
        _status("\n" + _DIV)
        _status("  Data Wrangler  —  loading and cleaning data")
        _status(_DIV)
        try:
            wrangler_output = self.data_wrangler.run(data_path)
            result["data_quality_report"] = wrangler_output["data_quality_report"]
            qr = wrangler_output["data_quality_report"]
            _status(
                f"  OK  {qr['final_row_count']} rows, "
                f"{qr['final_column_count']} columns, "
                f"{qr['duplicate_rows_dropped']} duplicates dropped"
            )
        except Exception:
            return self._fail(result, "Data Wrangler", traceback.format_exc())

        # ── Stage 2: Analyst ─────────────────────────────────────────
        _status("\n" + _DIV)
        _status("  Analyst  —  profiling data and generating insights")
        _status(_DIV)
        try:
            analyst_output = self.analyst.run(wrangler_output, question=question)
            result["insights"] = analyst_output["insights"]
            _status(f"  OK  {len(analyst_output['insights'])} insights, "
                    f"{len(analyst_output['suggested_charts'])} charts suggested")
        except Exception:
            return self._fail(result, "Analyst", traceback.format_exc())

        # ── Stage 3: Anomaly Agent ───────────────────────────────────
        _status("\n" + _DIV)
        _status("  Anomaly Agent  —  detecting statistical anomalies")
        _status(_DIV)
        try:
            anomaly_output = self.anomaly_agent.run(
                wrangler_output["dataframe"], analyst_output
            )
            result["anomalies"]         = anomaly_output["anomalies"]
            result["anomaly_narrative"] = anomaly_output["narrative"]
            result["severity_counts"]   = anomaly_output["severity_counts"]
            sc = anomaly_output["severity_counts"]
            _status(
                f"  OK  {len(anomaly_output['anomalies'])} anomalies — "
                f"high={sc['high']} medium={sc['medium']} low={sc['low']}"
            )
        except Exception:
            return self._fail(result, "Anomaly Agent", traceback.format_exc())

        # ── Stage 4: Decision Agent ──────────────────────────────────
        _status("\n" + _DIV)
        _status("  Decision Agent  —  generating actionable decisions")
        _status(_DIV)
        try:
            decision_output = self.decision_agent.run(
                question       = question,
                analyst_output = analyst_output,
                anomaly_output = anomaly_output,
                dataframe      = wrangler_output["dataframe"],
            )
            result["decisions"]        = decision_output["decisions"]
            result["decision_summary"] = decision_output["summary"]
            result["decision_domain"]  = decision_output["domain"]
            _status(
                f"  OK  {len(decision_output['decisions'])} decisions — "
                f"domain: {decision_output['domain'][:60]}"
            )
        except Exception:
            return self._fail(result, "Decision Agent", traceback.format_exc())

        # ── Stage 5: Forecasting Agent ───────────────────────────────
        _status("\n" + _DIV)
        _status("  Forecasting Agent  —  fitting trends and projecting forward")
        _status(_DIV)
        try:
            forecast_output = self.forecasting_agent.run(
                wrangler_output["dataframe"], analyst_output
            )
            result["forecasts"]          = forecast_output["forecasts"]
            result["forecast_narrative"] = forecast_output["narrative"]
            result["forecast_figures"]   = forecast_output["figure_paths"]
            _status(
                f"  OK  {len(forecast_output['forecasts'])} forecasts · "
                f"{len(forecast_output['figure_paths'])} charts"
            )
        except Exception:
            return self._fail(result, "Forecasting Agent", traceback.format_exc())

        # ── Stage 6: Stats Agent ─────────────────────────────────────
        _status("\n" + _DIV)
        _status("  Stats Agent  —  running hypothesis tests")
        _status(_DIV)
        try:
            stats_output = self.stats_agent.run(
                wrangler_output["dataframe"], analyst_output, question
            )
            result["stats_tests"]           = stats_output["tests_run"]
            result["stats_significant"]     = stats_output["significant_findings"]
            result["stats_narrative"]       = stats_output["narrative"]
            result["stats_recommendations"] = stats_output["recommendations"]
            _status(
                f"  OK  {len(stats_output['tests_run'])} tests — "
                f"{len(stats_output['significant_findings'])} significant"
            )
        except Exception:
            return self._fail(result, "Stats Agent", traceback.format_exc())

        # ── Stage 7: Viz Builder ─────────────────────────────────────
        _status("\n" + _DIV)
        _status("  Viz Builder  —  rendering charts")
        _status(_DIV)
        try:
            viz_output = self.viz_builder.run(wrangler_output, analyst_output)
            result["figure_paths"]    = viz_output["figure_paths"]
            result["charts_rendered"] = viz_output["charts_rendered"]
            _status(
                f"  OK  {viz_output['charts_rendered']} rendered, "
                f"{viz_output['charts_skipped']} skipped"
            )
            for p in viz_output["figure_paths"]:
                _status(f"       {p}")
        except Exception:
            return self._fail(result, "Viz Builder", traceback.format_exc())

        # ── Stage 8: Report Writer ───────────────────────────────────
        _status("\n" + _DIV)
        _status("  Report Writer  —  composing final report")
        _status(_DIV)
        try:
            report_output = self.report_writer.run(question, analyst_output, viz_output)
            result["report_path"] = report_output["report_path"]
            result["report_text"] = report_output["report_text"]
            _status(f"  OK  report saved to {report_output['report_path']}")
        except Exception:
            return self._fail(result, "Report Writer", traceback.format_exc())

        _status("\n" + _DIV)
        _status("  Pipeline complete")
        _status(_DIV + "\n")
        return result

    @staticmethod
    def _fail(result: dict, stage: str, tb: str) -> dict:
        """Populate the error key and print a readable failure message."""
        msg = f"{stage} failed:\n{tb}"
        result["error"] = msg
        _status(f"\n  ERROR in {stage}:\n")
        _status(tb)
        return result
