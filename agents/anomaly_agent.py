"""
Anomaly Agent: detects statistical anomalies in any tabular dataset.

Three detection methods run in sequence:
  - Z-score       : flags values more than 3 standard deviations from the column mean
  - IQR           : flags values outside Q1 - 1.5*IQR / Q3 + 1.5*IQR
  - Year-over-year: flags columns where a value changed more than 2 std of all
                    period-to-period changes for that column (requires time + entity columns)

Anomalies caught by multiple methods are promoted to "high" severity.
Claude interprets the top anomalies and produces a plain-English narrative.

No column names hardcoded — works with any dataset.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import anthropic

SYSTEM_PROMPT = (
    "You are a data analyst specialising in anomaly investigation. "
    "Always respond with valid JSON only. No markdown, no explanation outside the JSON."
)

ZSCORE_HIGH_THRESHOLD = 4.0   # z >= 4 → high on its own
ZSCORE_MED_THRESHOLD  = 3.0   # z >= 3 → medium (entry point for z-score)
MAX_NARRATIVE_ITEMS   = 20    # cap anomalies sent to Claude

_SEVERITY_RANK = {"high": 2, "medium": 1, "low": 0}


class AnomalyAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(self, dataframe: pd.DataFrame, analyst_output: dict) -> dict:
        """
        Detect anomalies and return a structured result.

        Args:
            dataframe:      Cleaned DataFrame from DataWrangler.
            analyst_output: Output from Analyst — uses column_descriptions and insights.

        Returns:
            dict with keys:
              - "anomalies":       list[dict] — each anomaly with column/entity/value/severity/reason
              - "narrative":       str        — Claude's plain-English interpretation
              - "severity_counts": dict       — {"high": N, "medium": N, "low": N}
        """
        col_desc    = analyst_output.get("column_descriptions", {})
        insights    = analyst_output.get("insights", [])
        metric_cols = self._metric_columns(dataframe, col_desc)
        entity_col  = self._find_col_by_role(dataframe, col_desc, "category")
        time_col    = self._find_col_by_role(dataframe, col_desc, "time")

        anomalies: list[dict] = []
        anomalies.extend(self._detect_zscore(dataframe, metric_cols, entity_col, time_col))
        anomalies.extend(self._detect_iqr(dataframe, metric_cols, entity_col, time_col))
        if time_col and entity_col:
            anomalies.extend(self._detect_yoy(dataframe, metric_cols, entity_col, time_col))

        anomalies       = self._merge(anomalies)
        severity_counts = self._count_severity(anomalies)
        narrative       = self._interpret(anomalies[:MAX_NARRATIVE_ITEMS], col_desc, insights)

        return {
            "anomalies":       anomalies,
            "narrative":       narrative,
            "severity_counts": severity_counts,
        }

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    def _metric_columns(self, df: pd.DataFrame, col_desc: dict) -> list[str]:
        """Numeric columns with role 'metric' — skip id/rank/category/time."""
        skip = {"id", "rank", "category", "time"}
        numeric = df.select_dtypes(include="number").columns
        return [c for c in numeric if col_desc.get(c, {}).get("role", "metric") not in skip]

    def _find_col_by_role(
        self, df: pd.DataFrame, col_desc: dict, role: str
    ) -> str | None:
        for col, desc in col_desc.items():
            if desc.get("role") == role and col in df.columns:
                return col
        return None

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _detect_zscore(
        self,
        df: pd.DataFrame,
        cols: list[str],
        entity_col: str | None,
        time_col: str | None,
    ) -> list[dict]:
        results = []
        for col in cols:
            s   = df[col].dropna()
            std = s.std()
            if std == 0:
                continue
            mean = s.mean()
            for idx, val in df[col].items():
                if pd.isna(val):
                    continue
                z = abs((val - mean) / std)
                if z < ZSCORE_MED_THRESHOLD:
                    continue
                entity   = str(df.loc[idx, entity_col]) if entity_col else f"row {idx}"
                time_val = df.loc[idx, time_col] if time_col else None
                results.append({
                    "column":   col,
                    "entity":   entity,
                    "time":     time_val,
                    "value":    round(float(val), 6),
                    "z_score":  round(float(z), 3),
                    "method":   "z_score",
                    "severity": "high" if z >= ZSCORE_HIGH_THRESHOLD else "medium",
                    "reason": (
                        f"{col} = {val:.4g} for {entity}"
                        + (f" ({time_val})" if time_val is not None else "")
                        + f" is {z:.2f} std from the mean ({mean:.4g})"
                    ),
                })
        return results

    def _detect_iqr(
        self,
        df: pd.DataFrame,
        cols: list[str],
        entity_col: str | None,
        time_col: str | None,
    ) -> list[dict]:
        results = []
        for col in cols:
            s   = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

            for idx, val in df[col].items():
                if pd.isna(val) or lower <= val <= upper:
                    continue
                entity   = str(df.loc[idx, entity_col]) if entity_col else f"row {idx}"
                time_val = df.loc[idx, time_col] if time_col else None
                side     = "below lower" if val < lower else "above upper"
                bound    = lower if val < lower else upper
                results.append({
                    "column":   col,
                    "entity":   entity,
                    "time":     time_val,
                    "value":    round(float(val), 6),
                    "z_score":  None,
                    "method":   "iqr",
                    "severity": "medium",
                    "reason": (
                        f"{col} = {val:.4g} for {entity}"
                        + (f" ({time_val})" if time_val is not None else "")
                        + f" is {side} IQR fence ({bound:.4g})"
                    ),
                })
        return results

    def _detect_yoy(
        self,
        df: pd.DataFrame,
        cols: list[str],
        entity_col: str,
        time_col: str,
    ) -> list[dict]:
        """Flag per-entity period-over-period changes > mean + 2*std of all changes."""
        results   = []
        df_sorted = df.sort_values([entity_col, time_col])

        for col in cols:
            # Build reference distribution of absolute changes across all entities
            all_abs_changes: list[float] = []
            for _, grp in df_sorted.groupby(entity_col, sort=False):
                vals = grp[col].values
                for i in range(1, len(vals)):
                    if not (pd.isna(vals[i]) or pd.isna(vals[i - 1])):
                        all_abs_changes.append(abs(float(vals[i]) - float(vals[i - 1])))

            if len(all_abs_changes) < 3:
                continue

            arr       = np.array(all_abs_changes)
            threshold = arr.mean() + 2 * arr.std()

            for entity, grp in df_sorted.groupby(entity_col, sort=False):
                grp  = grp.sort_values(time_col)
                vals = grp[col].values
                times = grp[time_col].values

                for i in range(1, len(vals)):
                    if pd.isna(vals[i]) or pd.isna(vals[i - 1]):
                        continue
                    delta = float(vals[i]) - float(vals[i - 1])
                    if abs(delta) <= threshold:
                        continue
                    pct = (delta / abs(float(vals[i - 1])) * 100) if vals[i - 1] != 0 else 0
                    results.append({
                        "column":   col,
                        "entity":   str(entity),
                        "time":     times[i],
                        "value":    round(float(vals[i]), 6),
                        "z_score":  None,
                        "method":   "yoy",
                        "severity": "medium",
                        "reason": (
                            f"{col} for {entity} changed {pct:+.1f}% "
                            f"from {times[i - 1]} to {times[i]} "
                            f"({vals[i - 1]:.4g} → {vals[i]:.4g})"
                        ),
                    })
        return results

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _merge(self, anomalies: list[dict]) -> list[dict]:
        """
        Merge anomalies for the same (entity, column, time) from different methods.
        Records all methods that fired; promotes to 'high' when 2+ methods agree.
        """
        seen: dict[tuple, dict] = {}
        for a in anomalies:
            key = (a["entity"], a["column"], str(a["time"]))
            if key not in seen:
                entry = dict(a)
                entry["methods"] = [a["method"]]
                seen[key] = entry
            else:
                existing = seen[key]
                if a["method"] not in existing["methods"]:
                    existing["methods"].append(a["method"])
                # Higher severity wins
                if _SEVERITY_RANK[a["severity"]] > _SEVERITY_RANK[existing["severity"]]:
                    existing["severity"] = a["severity"]
                # Prefer z_score value when available
                if a.get("z_score") and not existing.get("z_score"):
                    existing["z_score"] = a["z_score"]
                # 2+ methods → escalate to high
                if len(existing["methods"]) >= 2:
                    existing["severity"] = "high"

        merged = list(seen.values())
        merged.sort(key=lambda x: (
            -_SEVERITY_RANK[x["severity"]],
            -(x.get("z_score") or 0),
        ))
        return merged

    def _count_severity(self, anomalies: list[dict]) -> dict:
        counts = {"high": 0, "medium": 0, "low": 0}
        for a in anomalies:
            counts[a.get("severity", "low")] += 1
        return counts

    # ------------------------------------------------------------------
    # Claude narrative
    # ------------------------------------------------------------------

    def _interpret(
        self,
        anomalies: list[dict],
        col_desc: dict,
        insights: list[str],
    ) -> str:
        if not anomalies:
            return "No significant anomalies detected in the dataset."

        summary = [
            {
                "column":   a["column"],
                "entity":   a["entity"],
                "time":     str(a["time"]) if a["time"] is not None else None,
                "value":    a["value"],
                "severity": a["severity"],
                "methods":  a.get("methods", []),
                "reason":   a["reason"],
            }
            for a in anomalies
        ]

        col_roles = {c: d.get("role", "?") for c, d in col_desc.items()}

        prompt = f"""You are reviewing {len(anomalies)} statistical anomalies detected in a dataset.

Column roles:
{json.dumps(col_roles, indent=2)}

Existing insights from statistical analysis:
{json.dumps(insights[:3], indent=2)}

Detected anomalies:
{json.dumps(summary, indent=2)}

Write a concise plain-English narrative that:
1. Highlights the most significant anomalies (especially 'high' severity ones)
2. Groups related findings (same entity or same time period)
3. Suggests plausible real-world explanations where the context supports it
4. Identifies which anomalies most warrant human investigation

Return a JSON object with exactly one key:
{{"narrative": "your interpretation here, use \\n to separate distinct findings"}}

Return only valid JSON. No other text."""

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
