"""
Comparison Agent: side-by-side analysis of two datasets on shared numeric columns.

For every column that exists in both DataFrames with a numeric dtype, the agent:
  - Computes mean, std, and median for each dataset
  - Calculates absolute and % difference (relative to dataset 1)
  - Determines which dataset is higher (not which is "better" — that requires domain context)
  - Runs an independent t-test to assess whether the difference is statistically significant
  - Computes Cohen's d effect size

Claude receives all column-level comparisons and the user's question, then writes:
  - A comparison narrative (plain English with specific numbers)
  - An overall summary of which dataset "wins" on the most columns
  - 3 key takeaways

Returns a structured dict with per-column stats, narrative, winner tally, and takeaways.
No column names are hardcoded — works with any pair of tabular datasets.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import scipy.stats
import anthropic

SYSTEM_PROMPT = (
    "You are a data analyst comparing two datasets. "
    "Always respond with valid JSON only. No markdown, no explanation outside the JSON."
)

P_THRESHOLD = 0.05


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2  = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled = np.sqrt(
        ((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2)
    )
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _effect_label(d: float) -> str:
    d = abs(d)
    if d >= 0.8: return "large"
    if d >= 0.5: return "medium"
    if d >= 0.2: return "small"
    return "negligible"


class ComparisonAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(
        self,
        dataframe1: pd.DataFrame,
        dataframe2: pd.DataFrame,
        name1: str,
        name2: str,
        question: str = "",
    ) -> dict:
        """
        Compare two datasets on their common numeric columns.

        Args:
            dataframe1: First dataset.
            dataframe2: Second dataset.
            name1:      Human-readable label for the first dataset (e.g. "2023 Sales").
            name2:      Human-readable label for the second dataset (e.g. "2024 Sales").
            question:   Optional plain-English question guiding the comparison.

        Returns:
            dict with keys:
              - "comparisons":     list[dict] — per-column stats sorted by effect size
              - "common_columns":  list[str]  — columns analysed
              - "skipped_columns": list[str]  — columns present in one dataset only
              - "winner_tally":    dict       — {name1: N, name2: N, "tied": N}
              - "narrative":       str        — Claude's plain-English comparison
              - "takeaways":       list[str]  — 3 key conclusions
              - "name1":           str
              - "name2":           str
        """
        common, skipped = self._find_common_numeric(dataframe1, dataframe2)
        comparisons = [
            self._compare_column(dataframe1, dataframe2, col, name1, name2)
            for col in common
        ]
        comparisons.sort(key=lambda x: -abs(x.get("cohen_d", 0)))

        tally = self._tally_winners(comparisons, name1, name2)
        result = self._interpret(comparisons, name1, name2, question, tally)

        return {
            "comparisons":     comparisons,
            "common_columns":  common,
            "skipped_columns": skipped,
            "winner_tally":    tally,
            "narrative":       result.get("narrative", ""),
            "takeaways":       result.get("takeaways", []),
            "name1":           name1,
            "name2":           name2,
        }

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    def _find_common_numeric(
        self, df1: pd.DataFrame, df2: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """Return (common_numeric_cols, skipped_cols)."""
        num1    = set(df1.select_dtypes(include="number").columns)
        num2    = set(df2.select_dtypes(include="number").columns)
        common  = sorted(num1 & num2)
        skipped = sorted((num1 | num2) - (num1 & num2))
        return common, skipped

    # ------------------------------------------------------------------
    # Per-column comparison
    # ------------------------------------------------------------------

    def _compare_column(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        col: str,
        name1: str,
        name2: str,
    ) -> dict:
        a = df1[col].dropna().values.astype(float)
        b = df2[col].dropna().values.astype(float)

        mean1, mean2   = float(a.mean()) if len(a) else 0.0, float(b.mean()) if len(b) else 0.0
        std1,  std2    = float(a.std())  if len(a) else 0.0, float(b.std())  if len(b) else 0.0
        med1,  med2    = float(np.median(a)) if len(a) else 0.0, float(np.median(b)) if len(b) else 0.0

        mean_diff = mean2 - mean1
        pct_change = (mean_diff / abs(mean1) * 100) if mean1 != 0 else 0.0

        # t-test
        if len(a) >= 2 and len(b) >= 2:
            stat, p_value = scipy.stats.ttest_ind(a, b)
        else:
            stat, p_value = 0.0, 1.0

        d          = _cohen_d(a, b)
        significant = float(p_value) < P_THRESHOLD
        higher      = name2 if mean2 > mean1 else (name1 if mean1 > mean2 else "tied")

        return {
            "column":       col,
            "mean1":        round(mean1, 6),
            "mean2":        round(mean2, 6),
            "std1":         round(std1, 6),
            "std2":         round(std2, 6),
            "median1":      round(med1, 6),
            "median2":      round(med2, 6),
            "mean_diff":    round(mean_diff, 6),
            "pct_change":   round(pct_change, 2),
            "higher":       higher,
            "t_statistic":  round(float(stat), 4),
            "p_value":      round(float(p_value), 6),
            "significant":  significant,
            "cohen_d":      round(d, 4),
            "effect_label": _effect_label(d),
            "n1":           len(a),
            "n2":           len(b),
        }

    # ------------------------------------------------------------------
    # Winner tally
    # ------------------------------------------------------------------

    def _tally_winners(self, comparisons: list[dict], name1: str, name2: str) -> dict:
        tally = {name1: 0, name2: 0, "tied": 0}
        for c in comparisons:
            h = c.get("higher", "tied")
            if h == name1:
                tally[name1] += 1
            elif h == name2:
                tally[name2] += 1
            else:
                tally["tied"] += 1
        return tally

    # ------------------------------------------------------------------
    # Claude narrative
    # ------------------------------------------------------------------

    def _interpret(
        self,
        comparisons: list[dict],
        name1: str,
        name2: str,
        question: str,
        tally: dict,
    ) -> dict:
        # Send top 12 comparisons (by effect size) to Claude
        top = [
            {k: v for k, v in c.items() if k not in ("std1", "std2", "median1", "median2",
                                                       "t_statistic", "n1", "n2")}
            for c in comparisons[:12]
        ]
        prompt = f"""You are comparing two datasets:
  Dataset A: "{name1}"
  Dataset B: "{name2}"

{f'Comparison question: "{question}"' if question else ''}

Winner tally (which dataset is higher on each column):
{json.dumps(tally, indent=2)}

Per-column comparisons (sorted by effect size, significant findings marked):
{json.dumps(top, indent=2)}

Note: "higher" means which dataset has a larger mean — it does NOT mean "better".
The interpretation of higher/lower depends on the domain context (e.g. higher revenue is good;
higher error rate is bad).

Write a comparison analysis. Return a JSON object:
{{
  "narrative": "3-5 sentence plain-English comparison. Reference specific column names and % figures.
                Use the format '{{name1}} is higher/lower in {{column}} ({{% change}}%)'.
                Highlight statistically significant differences with large or medium effect sizes first.
                Use \\n to separate major points.",
  "takeaways": [
    "Takeaway 1 — one sentence, specific and actionable",
    "Takeaway 2",
    "Takeaway 3"
  ]
}}

Return only valid JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
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
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"narrative": raw, "takeaways": []}
