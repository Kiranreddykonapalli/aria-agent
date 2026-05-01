"""
What-If Simulator Agent: applies a plain-English scenario to a dataset copy and
estimates the downstream impact on correlated columns using linear regression coefficients.

Flow:
  1. Parse the scenario (Claude) → target_column, change_type, change_value, row filter
  2. Apply the change to a DataFrame copy (never mutate the original)
  3. Estimate downstream impacts via β·Δx for each correlated column
  4. Generate a 2-panel before/after chart: KDE overlay + impact bar chart
  5. Claude writes a plain-English narrative of the simulated results

Downstream impact model:
  Δy ≈ β_xy · Δx_mean   where   β_xy = cov(X,Y) / var(X)

This is a first-order linear approximation. Results are projections, not predictions.
No column names are hardcoded — works with any tabular dataset.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime, timezone

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import anthropic

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

OUTPUT_DIR = "output/figures"

SYSTEM_PROMPT = (
    "You are a data scientist. Always respond with valid JSON only. "
    "No markdown, no explanation outside the JSON."
)

# Dark chart theme
BG      = "#0d1117"
SURFACE = "#161b22"
TEXT    = "#e6edf3"
SPINE   = "#30363d"
GRID    = "#30363d"
C1      = "#4C8BB5"   # before
C2      = "#E8834E"   # after / impact

VALID_OPS = {"==", "!=", ">", "<", ">=", "<="}


def _apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BG,    "axes.facecolor":  SURFACE,
        "axes.edgecolor":   SPINE,  "axes.labelcolor": TEXT,
        "axes.titlecolor":  TEXT,   "axes.grid":       True,
        "grid.color":       GRID,   "grid.linestyle":  "--",
        "grid.alpha":       0.4,    "xtick.color":     TEXT,
        "ytick.color":      TEXT,   "text.color":      TEXT,
        "font.size":        10,     "axes.titlesize":  12,
    })


_apply_style()


class WhatIfAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run(
        self,
        dataframe: pd.DataFrame,
        analyst_output: dict,
        forecast_output: dict,
        scenario: str,
    ) -> dict:
        """
        Simulate a plain-English what-if scenario on the dataset.

        Args:
            dataframe:      Cleaned DataFrame from DataWrangler.
            analyst_output: Output from Analyst — uses column_descriptions and stats.
            forecast_output: Output from ForecastingAgent — used for context in narrative.
            scenario:       Plain-English scenario string.

        Returns:
            dict with keys:
              - "scenario_parsed":  dict  — Claude's structured interpretation of the scenario
              - "changes_applied":  dict  — actual changes made to the simulated DataFrame
              - "impact_summary":   dict  — estimated downstream effects on correlated columns
              - "narrative":        str   — Claude's plain-English impact narrative
              - "figure_path":      str | None — path to saved before/after chart
        """
        col_desc = analyst_output.get("column_descriptions", {})
        parsed   = self._parse_scenario(dataframe, col_desc, scenario)

        if "error" in parsed:
            return {
                "scenario_parsed":  parsed,
                "changes_applied":  {},
                "impact_summary":   {},
                "narrative":        f"Could not parse scenario: {parsed['error']}",
                "figure_path":      None,
            }

        df_sim, changes = self._apply_scenario(dataframe, parsed)
        impacts         = self._estimate_impacts(dataframe, df_sim, parsed)
        figure_path     = self._plot(dataframe, df_sim, parsed, impacts)
        narrative       = self._narrate(scenario, parsed, changes, impacts, dataframe, df_sim, col_desc)

        return {
            "scenario_parsed": parsed,
            "changes_applied": changes,
            "impact_summary":  impacts,
            "narrative":       narrative,
            "figure_path":     figure_path,
        }

    # ------------------------------------------------------------------
    # 1. Parse scenario
    # ------------------------------------------------------------------

    def _parse_scenario(self, df: pd.DataFrame, col_desc: dict, scenario: str) -> dict:
        col_info = {col: str(df[col].dtype) for col in df.columns}
        numeric  = df.select_dtypes(include="number").columns.tolist()

        prompt = f"""Parse this what-if scenario for a dataset.

Dataset columns (name → dtype):
{json.dumps(col_info, indent=2)}

Numeric columns available: {numeric}

Scenario: "{scenario}"

Return a JSON object:
{{
  "target_column":    "exact column name from the dataset to modify",
  "change_type":      "percentage" | "absolute",
  "change_value":     <number — negative = decrease, positive = increase>,
  "affected_rows":    "all" | "filter",
  "filter_column":    <column name or null if affected_rows is "all">,
  "filter_operator":  "==" | "!=" | ">" | "<" | ">=" | "<=" | null,
  "filter_value":     <value to compare or null>,
  "interpretation":   "one sentence restating the scenario in plain English"
}}

Rules:
- target_column MUST be a numeric column from the dataset.
- change_type "percentage": applies change_value as % (e.g. -5 means multiply by 0.95).
- change_type "absolute": adds change_value directly to the column values.
- If the scenario applies to all rows, set affected_rows to "all" and leave filter fields null.
- Return only valid JSON."""

        response = self.client.messages.create(
            model=self.model, max_tokens=400,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        raw     = response.content[0].text
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return {"error": f"JSON parse failed: {exc}"}

        # Validate
        col = parsed.get("target_column", "")
        if col not in df.columns:
            return {"error": f"target_column '{col}' not found in dataset"}
        if parsed.get("change_type") not in ("percentage", "absolute"):
            return {"error": "change_type must be 'percentage' or 'absolute'"}
        if parsed.get("filter_operator") and parsed["filter_operator"] not in VALID_OPS:
            return {"error": f"invalid filter_operator '{parsed['filter_operator']}'"}

        return parsed

    # ------------------------------------------------------------------
    # 2. Apply scenario
    # ------------------------------------------------------------------

    def _apply_scenario(
        self, df: pd.DataFrame, parsed: dict
    ) -> tuple[pd.DataFrame, dict]:
        df_sim = df.copy()
        col    = parsed["target_column"]
        val    = float(parsed["change_value"])
        ctype  = parsed["change_type"]

        if parsed.get("affected_rows") == "filter" and parsed.get("filter_column"):
            mask = self._build_mask(df_sim, parsed)
        else:
            mask = pd.Series(True, index=df_sim.index)

        orig_vals  = df_sim.loc[mask, col].copy()
        if ctype == "percentage":
            df_sim.loc[mask, col] = orig_vals * (1 + val / 100)
        else:
            df_sim.loc[mask, col] = orig_vals + val

        rows_changed = int(mask.sum())
        changes = {
            "column":        col,
            "change_type":   ctype,
            "change_value":  val,
            "rows_changed":  rows_changed,
            "original_mean": round(float(orig_vals.mean()), 6),
            "simulated_mean": round(float(df_sim.loc[mask, col].mean()), 6),
            "mean_delta":    round(float((df_sim.loc[mask, col] - orig_vals).mean()), 6),
        }
        return df_sim, changes

    def _build_mask(self, df: pd.DataFrame, parsed: dict) -> pd.Series:
        fc, op, fv = parsed["filter_column"], parsed["filter_operator"], parsed["filter_value"]
        try:
            fv = float(fv)
        except (TypeError, ValueError):
            fv = str(fv)
        s = df[fc]
        ops = {"==": s == fv, "!=": s != fv, ">": s > fv,
               "<": s < fv, ">=": s >= fv, "<=": s <= fv}
        return ops.get(op, pd.Series(True, index=df.index))

    # ------------------------------------------------------------------
    # 3. Estimate downstream impacts
    # ------------------------------------------------------------------

    def _estimate_impacts(
        self, df_orig: pd.DataFrame, df_sim: pd.DataFrame, parsed: dict
    ) -> dict:
        target = parsed["target_column"]
        delta  = float((df_sim[target] - df_orig[target]).mean())
        if abs(delta) < 1e-12:
            return {}

        numeric  = df_orig.select_dtypes(include="number").columns
        impacts  = {}
        var_x    = float(df_orig[target].var())
        if var_x == 0:
            return {}

        for col in numeric:
            if col == target:
                continue
            x  = df_orig[target].dropna()
            y  = df_orig[col].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) < 5:
                continue
            beta   = float(x[idx].cov(y[idx]) / var_x)
            dy     = beta * delta
            orig_m = float(df_orig[col].mean())
            pct    = (dy / abs(orig_m) * 100) if orig_m != 0 else 0.0
            try:
                r, _ = scipy.stats.pearsonr(x[idx], y[idx])
            except Exception:
                r = 0.0
            impacts[col] = {
                "correlation":          round(float(r), 3),
                "estimated_delta":      round(float(dy), 6),
                "estimated_pct_change": round(float(pct), 2),
                "original_mean":        round(float(orig_m), 6),
                "simulated_mean":       round(float(orig_m + dy), 6),
            }

        return dict(sorted(impacts.items(), key=lambda x: -abs(x[1]["correlation"])))

    # ------------------------------------------------------------------
    # 4. Chart
    # ------------------------------------------------------------------

    def _plot(
        self, df_orig: pd.DataFrame, df_sim: pd.DataFrame,
        parsed: dict, impacts: dict
    ) -> str | None:
        target = parsed["target_column"]
        try:
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5))

            # Left panel: KDE before vs after
            orig_s = df_orig[target].dropna()
            sim_s  = df_sim[target].dropna()
            from scipy.stats import gaussian_kde
            for s, color, label in [(orig_s, C1, "Before"), (sim_s, C2, "After (simulated)")]:
                xs = np.linspace(s.min() * 0.9, s.max() * 1.1, 300)
                try:
                    kde = gaussian_kde(s)
                    ax_left.plot(xs, kde(xs), color=color, linewidth=2.5, label=label)
                    ax_left.fill_between(xs, kde(xs), alpha=0.15, color=color)
                except Exception:
                    ax_left.hist(s, bins=20, color=color, alpha=0.5, label=label, density=True)

            ax_left.set_title(f"{target.replace('_',' ').title()} — Before vs After")
            ax_left.set_xlabel(target.replace("_", " ").title())
            ax_left.set_ylabel("Density")
            ax_left.legend(fontsize=9)
            ax_left.spines["top"].set_visible(False)
            ax_left.spines["right"].set_visible(False)

            # Right panel: estimated % change on top correlated columns
            top = list(impacts.items())[:8]
            if top:
                cols   = [c.replace("_", " ").title()[:25] for c, _ in top]
                deltas = [v["estimated_pct_change"] for _, v in top]
                colors = [C2 if d >= 0 else C1 for d in deltas]
                bars   = ax_right.barh(cols[::-1], deltas[::-1], color=colors[::-1], height=0.6)
                ax_right.axvline(0, color=SPINE, linewidth=1)
                ax_right.set_title("Estimated Impact on Correlated Columns (%)")
                ax_right.set_xlabel("Estimated % Change")
                ax_right.spines["top"].set_visible(False)
                ax_right.spines["right"].set_visible(False)
            else:
                ax_right.text(0.5, 0.5, "No correlated columns found",
                              ha="center", va="center", transform=ax_right.transAxes, color=TEXT)

            fig.suptitle(
                f"What-If: {parsed.get('interpretation', parsed.get('target_column',''))}",
                fontsize=13, fontweight="bold",
            )
            fig.tight_layout()

            ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = os.path.join(OUTPUT_DIR, f"{ts}_whatif_{target}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            return path
        except Exception:
            plt.close("all")
            return None

    # ------------------------------------------------------------------
    # 5. Narrative
    # ------------------------------------------------------------------

    def _narrate(
        self, scenario: str, parsed: dict, changes: dict,
        impacts: dict, df_orig: pd.DataFrame, df_sim: pd.DataFrame,
        col_desc: dict,
    ) -> str:
        top_impacts = {k: v for k, v in list(impacts.items())[:8]}
        col_roles   = {c: d.get("role", "?") for c, d in col_desc.items()}

        # Find entity column for labelling (category role)
        entity_col = next((c for c, d in col_desc.items()
                           if d.get("role") == "category" and c in df_orig.columns), None)
        # Top movers if entity column exists
        top_movers: list[str] = []
        if entity_col:
            target = parsed["target_column"]
            diff   = (df_sim[target] - df_orig[target]).abs()
            top_movers = df_orig.loc[diff.nlargest(3).index, entity_col].astype(str).tolist()

        prompt = f"""You are narrating a what-if simulation result for a data analyst.

Scenario: "{scenario}"
Interpretation: {parsed.get('interpretation', '')}

Direct change applied:
{json.dumps(changes, indent=2)}

Estimated downstream impacts on correlated columns:
{json.dumps(top_impacts, indent=2)}

Column roles for context:
{json.dumps(col_roles, indent=2)}

Top entities most affected: {top_movers}

Write a clear, specific 3-4 sentence narrative explaining:
1. What was changed and by how much
2. The most important estimated downstream effects (name columns and % figures)
3. Which entities / rows would see the largest gain or reduction
4. A brief caution that these are model projections based on linear correlations, not causal predictions

Be direct and specific. Reference actual numbers.

Return a JSON object with one key:
{{"narrative": "your narrative here — use \\n between distinct points"}}

Return only valid JSON."""

        response = self.client.messages.create(
            model=self.model, max_tokens=800,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        raw     = response.content[0].text
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned).get("narrative", raw)
        except json.JSONDecodeError:
            return raw
