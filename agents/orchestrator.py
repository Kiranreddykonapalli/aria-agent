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

from .data_wrangler import DataWrangler
from .analyst import Analyst
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
        self.data_wrangler  = DataWrangler()
        self.analyst        = Analyst(model=model)
        self.viz_builder    = VizBuilder()
        self.report_writer  = ReportWriter(model=model)

    def run(self, data_path: str, question: str) -> dict:
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
            "error":               None,
        }

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

        # ── Stage 3: Viz Builder ─────────────────────────────────────
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

        # ── Stage 4: Report Writer ───────────────────────────────────
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
