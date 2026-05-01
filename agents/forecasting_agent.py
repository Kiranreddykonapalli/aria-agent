"""
Forecasting Agent: fits linear trends on time-series data and projects forward 3 periods.

For each metric column it:
  - Aggregates to one value per time period (mean across all entities)
  - Fits a linear regression with scipy.stats.linregress
  - Projects 3 periods ahead with 95% prediction intervals
  - Produces a matplotlib chart: historical line + dashed forecast + shaded CI band
  - Sends all forecasts to Claude for a plain-English narrative

Requires a time column in the dataset (role='time' in column_descriptions).
Works with any tabular dataset — no column names are hardcoded.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import anthropic

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

OUTPUT_DIR = "output/figures"
SYSTEM_PROMPT = (
    "You are a data analyst. Always respond with valid JSON only. "
    "No markdown, no explanation outside the JSON."
)

# Dark chart theme — mirrors VizBuilder palette
BG_COLOR      = "#0d1117"
SURFACE_COLOR = "#161b22"
TEXT_COLOR    = "#e6edf3"
GRID_COLOR    = "#30363d"
SPINE_COLOR   = "#30363d"
COLOR_HIST    = "#4C8BB5"
COLOR_FORE    = "#E8834E"

MIN_R_SQUARED  = 0.20   # drop forecasts with very weak fit
MAX_CHARTS     = 6      # cap chart count to avoid flooding output/figures


def _apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor":   SURFACE_COLOR,
        "axes.edgecolor":   SPINE_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "axes.titlecolor":  TEXT_COLOR,
        "axes.grid":        True,
        "grid.color":       GRID_COLOR,
        "grid.linestyle":   "--",
        "grid.alpha":       0.4,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "text.color":       TEXT_COLOR,
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.titlesize":   13,
        "figure.dpi":       100,
    })


_apply_style()


class ForecastingAgent:
    def __init__(self, model: str = "claude-sonnet-4-6", output_dir: str = OUTPUT_DIR):
        self.model      = model
        self.output_dir = output_dir
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()
        os.makedirs(output_dir, exist_ok=True)

    def run(self, dataframe: pd.DataFrame, analyst_output: dict) -> dict:
        """
        Forecast every metric column 3 periods ahead of the last time period.

        Args:
            dataframe:      Cleaned DataFrame from DataWrangler.
            analyst_output: Output from Analyst — uses column_descriptions.

        Returns:
            dict with keys:
              - "forecasts":    list[dict] — one entry per metric with slope, r_squared,
                                forecast_<year> and ci_<year> for each future period
              - "narrative":    str  — Claude's plain-English interpretation
              - "figure_paths": list[str] — saved chart PNGs
        """
        col_desc    = analyst_output.get("column_descriptions", {})
        time_col    = self._find_col_by_role(dataframe, col_desc, "time")
        metric_cols = self._metric_columns(dataframe, col_desc)

        if not time_col:
            return {
                "forecasts":    [],
                "narrative":    "No time column detected — forecasting requires a temporal dimension.",
                "figure_paths": [],
            }

        # Aggregate: one value per time period across all entities
        agg = (
            dataframe.groupby(time_col)[metric_cols]
            .mean()
            .reset_index()
            .sort_values(time_col)
        )

        time_vals    = agg[time_col].values.astype(float)
        last_period  = time_vals[-1]
        future_times = [last_period + 1, last_period + 2, last_period + 3]

        forecasts:    list[dict] = []
        figure_paths: list[str]  = []

        for col in metric_cols:
            valid = agg[[time_col, col]].dropna()
            if len(valid) < 4:       # need at least 4 points for a meaningful fit
                continue

            x = valid[time_col].values.astype(float)
            y = valid[col].values.astype(float)

            fc = self._fit_and_forecast(col, x, y, future_times)
            if fc is None or fc["r_squared"] < MIN_R_SQUARED:
                continue

            forecasts.append(fc)

            if len(figure_paths) < MAX_CHARTS:
                path = self._plot(col, x, y, future_times, fc)
                if path:
                    figure_paths.append(path)

        # Sort by r_squared descending so strongest trends appear first
        forecasts.sort(key=lambda f: f["r_squared"], reverse=True)

        narrative = self._interpret(forecasts, col_desc, int(last_period), future_times)

        return {
            "forecasts":    forecasts,
            "narrative":    narrative,
            "figure_paths": figure_paths,
        }

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    def _find_col_by_role(self, df: pd.DataFrame, col_desc: dict, role: str) -> str | None:
        for col, desc in col_desc.items():
            if desc.get("role") == role and col in df.columns:
                return col
        return None

    def _metric_columns(self, df: pd.DataFrame, col_desc: dict) -> list[str]:
        skip = {"id", "rank", "category", "time"}
        numeric = df.select_dtypes(include="number").columns
        return [c for c in numeric if col_desc.get(c, {}).get("role", "metric") not in skip]

    # ------------------------------------------------------------------
    # Regression and forecasting
    # ------------------------------------------------------------------

    def _fit_and_forecast(
        self,
        col: str,
        x: np.ndarray,
        y: np.ndarray,
        future_times: list[float],
    ) -> dict | None:
        try:
            slope, intercept, r_value, p_value, _ = scipy.stats.linregress(x, y)
        except Exception:
            return None

        n       = len(x)
        x_mean  = x.mean()
        Sxx     = float(np.sum((x - x_mean) ** 2))
        y_fit   = slope * x + intercept
        se_res  = float(np.sqrt(np.sum((y - y_fit) ** 2) / (n - 2)))   # residual std error
        t_crit  = scipy.stats.t.ppf(0.975, df=n - 2)                   # 95% two-tailed

        result: dict = {
            "metric":     col,
            "slope":      round(float(slope), 8),
            "intercept":  round(float(intercept), 6),
            "r_squared":  round(float(r_value ** 2), 4),
            "p_value":    round(float(p_value), 4),
            "last_value": round(float(y[-1]), 6),
            "last_period": int(x[-1]),
            "confidence_interval": "95%",
            # internal — used by _plot, not exposed in narrative
            "_se_res":  se_res,
            "_t_crit":  t_crit,
            "_x_mean":  x_mean,
            "_Sxx":     Sxx,
            "_n":       n,
        }

        for ft in future_times:
            y_hat  = slope * ft + intercept
            # Prediction interval (wider than CI — accounts for individual scatter)
            se_pred = se_res * float(np.sqrt(1 + 1 / n + (ft - x_mean) ** 2 / Sxx))
            margin  = t_crit * se_pred
            yr      = int(ft)
            result[f"forecast_{yr}"] = round(float(y_hat), 6)
            result[f"ci_{yr}"]       = (round(float(y_hat - margin), 6),
                                         round(float(y_hat + margin), 6))

        return result

    # ------------------------------------------------------------------
    # Charting
    # ------------------------------------------------------------------

    def _plot(
        self,
        col: str,
        x: np.ndarray,
        y: np.ndarray,
        future_times: list[float],
        fc: dict,
    ) -> str | None:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))

            # Historical data
            ax.plot(x, y, color=COLOR_HIST, linewidth=2.2,
                    marker="o", markersize=5,
                    markerfacecolor=COLOR_HIST, markeredgecolor=BG_COLOR,
                    label="Historical (mean)")

            # Build forecast arrays
            ft_arr  = np.array(future_times)
            y_fore  = np.array([fc[f"forecast_{int(ft)}"] for ft in ft_arr])
            ci_lo   = np.array([fc[f"ci_{int(ft)}"][0]   for ft in ft_arr])
            ci_hi   = np.array([fc[f"ci_{int(ft)}"][1]   for ft in ft_arr])

            # Bridge from last historical point to first forecast
            bridge_x = np.array([x[-1], ft_arr[0]])
            bridge_y = np.array([y[-1], y_fore[0]])
            ax.plot(bridge_x, bridge_y, color=COLOR_FORE, linewidth=2,
                    linestyle="--", alpha=0.7)

            # Forecast line
            ax.plot(ft_arr, y_fore, color=COLOR_FORE, linewidth=2.2,
                    linestyle="--", marker="o", markersize=5,
                    markerfacecolor=COLOR_FORE, markeredgecolor=BG_COLOR,
                    label="Forecast (linear trend)")

            # Confidence band
            all_x = np.concatenate([[x[-1]], ft_arr])
            all_lo = np.concatenate([[y[-1]], ci_lo])
            all_hi = np.concatenate([[y[-1]], ci_hi])
            ax.fill_between(all_x, all_lo, all_hi,
                            color=COLOR_FORE, alpha=0.15,
                            label="95% prediction interval")

            # Historical/forecast divider
            ax.axvline(x[-1], color=SPINE_COLOR, linewidth=1, linestyle=":")

            ax.set_title(
                f"{col.replace('_', ' ').title()} — Trend & 3-Period Forecast "
                f"(R²={fc['r_squared']:.2f})",
                pad=14,
            )
            ax.set_xlabel("Period")
            ax.set_ylabel(col.replace("_", " ").title())
            ax.legend(fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(SPINE_COLOR)
            ax.spines["bottom"].set_color(SPINE_COLOR)
            fig.tight_layout()

            return self._save(fig, col)
        except Exception:
            plt.close("all")
            return None

    def _save(self, fig: plt.Figure, col: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug      = re.sub(r"[^\w]", "_", col)[:50]
        filename  = f"{timestamp}_forecast_{slug}.png"
        path      = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Claude narrative
    # ------------------------------------------------------------------

    def _interpret(
        self,
        forecasts: list[dict],
        col_desc: dict,
        last_period: int,
        future_times: list[float],
    ) -> str:
        if not forecasts:
            return "No statistically meaningful trends found to forecast."

        future_yrs = [int(ft) for ft in future_times]

        # Strip internal keys before sending to Claude
        clean = []
        for fc in forecasts:
            entry = {k: v for k, v in fc.items() if not k.startswith("_")}
            clean.append(entry)

        col_units = {
            col: desc.get("unit", "") for col, desc in col_desc.items()
        }

        prompt = f"""You are reviewing linear trend forecasts from a dataset.

Last observed period: {last_period}
Forecast periods: {future_yrs}

Column units:
{json.dumps(col_units, indent=2)}

Forecasts (sorted by R²):
{json.dumps(clean, indent=2)}

Write a concise narrative (4-6 sentences or short bullet points) interpreting these forecasts.
For each meaningful forecast:
  - State the projected value at the furthest forecast year with the unit
  - State whether the trend is increasing or decreasing and by how much from the last observed value
  - Note the R² (model fit) so the reader understands confidence level
  - Flag any forecast where the prediction interval is so wide it renders the forecast unreliable

Example style: "Obesity rate is projected to reach 38.2% by {future_yrs[-1]}, a 12% increase
from {last_period}, if current trends continue (R²=0.91)."

Return a JSON object with exactly one key:
{{"narrative": "your interpretation here, use \\n to separate distinct findings"}}

Return only valid JSON, no other text."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )

        raw     = response.content[0].text
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned).get("narrative", raw)
        except json.JSONDecodeError:
            return raw
