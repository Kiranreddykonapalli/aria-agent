"""
Data Quality Score Agent: scores any dataset out of 100 across 6 dimensions.

Dimensions and weights:
  Completeness  20 pts — % of non-null values
  Uniqueness    20 pts — % of non-duplicate rows
  Consistency   20 pts — % of numeric values within 3 std of their column mean
  Validity      20 pts — % of columns whose values match the inferred dtype
  Timeliness    10 pts — recency of the latest time/date value
  Uniformity    10 pts — formatting consistency in categorical columns

Grade scale: A ≥ 90 · B ≥ 75 · C ≥ 60 · D ≥ 45 · F < 45

Runs on the RAW dataframe (before DataWrangler) so it reflects true input quality.
wrangler_output is optional — pass None when running before DataWrangler.
No column names are hardcoded.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import anthropic

SYSTEM_PROMPT = (
    "You are a data quality analyst. Always respond with valid JSON only. "
    "No markdown, no explanation outside the JSON."
)

WEIGHTS = {
    "completeness": 20,
    "uniqueness":   20,
    "consistency":  20,
    "validity":     20,
    "timeliness":   10,
    "uniformity":   10,
}

NULL_LIKE = frozenset({"n/a", "na", "null", "none", "nan", "missing", "unknown", "-", ""})


def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 45: return "D"
    return "F"


class QualityAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(self, dataframe: pd.DataFrame, wrangler_output: dict | None = None) -> dict:
        """
        Score the dataset across 6 quality dimensions and produce a verdict.

        Args:
            dataframe:       Raw (or cleaned) DataFrame to assess.
            wrangler_output: Optional DataWrangler output — used to supplement null counts
                             if provided. Pass None when running before DataWrangler.

        Returns:
            dict with keys:
              - "overall_score":      float — weighted score out of 100
              - "dimension_scores":   dict  — {dimension: score} for all 6 dims
              - "dimension_details":  dict  — supporting metrics per dimension
              - "grade":              str   — A / B / C / D / F
              - "verdict":            str   — Claude's one-paragraph assessment
              - "recommendations":    list[str] — 3 actionable fixes
        """
        dim_scores:  dict[str, float] = {}
        dim_details: dict[str, dict]  = {}

        dim_scores["completeness"], dim_details["completeness"] = self._completeness(dataframe)
        dim_scores["uniqueness"],   dim_details["uniqueness"]   = self._uniqueness(dataframe)
        dim_scores["consistency"],  dim_details["consistency"]  = self._consistency(dataframe)
        dim_scores["validity"],     dim_details["validity"]     = self._validity(dataframe)
        dim_scores["timeliness"],   dim_details["timeliness"]   = self._timeliness(dataframe)
        dim_scores["uniformity"],   dim_details["uniformity"]   = self._uniformity(dataframe)

        overall = sum(dim_scores[d] for d in WEIGHTS)
        grade   = _grade(overall)

        result  = self._interpret(dataframe, overall, dim_scores, dim_details, grade)

        return {
            "overall_score":     round(overall, 1),
            "dimension_scores":  {k: round(v, 1) for k, v in dim_scores.items()},
            "dimension_details": dim_details,
            "grade":             grade,
            "verdict":           result.get("verdict", ""),
            "recommendations":   result.get("recommendations", []),
        }

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _completeness(self, df: pd.DataFrame) -> tuple[float, dict]:
        total  = df.size
        nulls  = int(df.isnull().sum().sum())
        pct    = (total - nulls) / total * 100 if total > 0 else 100
        worst  = df.isnull().mean().sort_values(ascending=False).head(3)
        return (
            pct / 100 * WEIGHTS["completeness"],
            {
                "null_cells":     nulls,
                "total_cells":    int(total),
                "pct_complete":   round(pct, 1),
                "worst_columns":  {c: f"{v:.0%}" for c, v in worst.items() if v > 0},
            },
        )

    def _uniqueness(self, df: pd.DataFrame) -> tuple[float, dict]:
        n_dups = int(df.duplicated().sum())
        pct    = (len(df) - n_dups) / len(df) * 100 if len(df) > 0 else 100
        return (
            pct / 100 * WEIGHTS["uniqueness"],
            {"duplicate_rows": n_dups, "total_rows": len(df), "pct_unique": round(pct, 1)},
        )

    def _consistency(self, df: pd.DataFrame) -> tuple[float, dict]:
        numeric = df.select_dtypes(include="number")
        scores: list[float] = []
        outlier_cols: list[str] = []
        for col in numeric.columns:
            s = numeric[col].dropna()
            if len(s) < 3:
                continue
            std = s.std()
            if std == 0:
                scores.append(100.0)
                continue
            in_range = ((s - s.mean()).abs() <= 3 * std).mean() * 100
            scores.append(float(in_range))
            if in_range < 97:
                outlier_cols.append(col)
        pct = float(np.mean(scores)) if scores else 95.0
        return (
            pct / 100 * WEIGHTS["consistency"],
            {"pct_consistent": round(pct, 1), "outlier_prone_columns": outlier_cols[:5]},
        )

    def _validity(self, df: pd.DataFrame) -> tuple[float, dict]:
        col_scores: list[float] = []
        mixed_type_cols: list[str] = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                col_scores.append(100.0)
                continue
            # Object column: check for mixed type signals
            s = df[col].dropna().astype(str)
            num_parsed = pd.to_numeric(s, errors="coerce").notna().mean()
            # If >80% parses as numeric but column is object → mixed type
            if num_parsed > 0.8:
                col_scores.append(60.0)
                mixed_type_cols.append(col)
            else:
                col_scores.append(100.0)
        pct = float(np.mean(col_scores)) if col_scores else 100.0
        return (
            pct / 100 * WEIGHTS["validity"],
            {"pct_valid": round(pct, 1), "mixed_type_columns": mixed_type_cols[:5]},
        )

    def _timeliness(self, df: pd.DataFrame) -> tuple[float, dict]:
        current_year = datetime.now(timezone.utc).year
        for col in df.columns:
            col_lower = col.lower()
            if not any(k in col_lower for k in ("year", "date", "time", "period", "month")):
                continue
            s = df[col].dropna()
            if len(s) == 0:
                continue
            # Numeric year column
            if pd.api.types.is_numeric_dtype(s):
                max_val = float(s.max())
                if 1900 <= max_val <= current_year + 1:
                    years_old = current_year - max_val
                    score_val = (10 if years_old <= 1 else 8 if years_old <= 2
                                 else 6 if years_old <= 5 else 4 if years_old <= 10 else 2)
                    return (
                        float(score_val),
                        {"latest_period": int(max_val), "years_old": int(years_old)},
                    )
            # Datetime-parseable column
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(s, errors="coerce")
                if parsed.notna().any():
                    max_dt    = parsed.max()
                    years_old = (datetime.now(timezone.utc) - max_dt.tz_localize("UTC")
                                 if max_dt.tzinfo is None else
                                 datetime.now(timezone.utc) - max_dt).days / 365
                    score_val = (10 if years_old <= 1 else 8 if years_old <= 2
                                 else 6 if years_old <= 5 else 4 if years_old <= 10 else 2)
                    return float(score_val), {"latest_period": str(max_dt.date()), "years_old": round(years_old, 1)}
            except Exception:
                continue
        return 5.0, {"note": "No time column detected — neutral score applied"}

    def _uniformity(self, df: pd.DataFrame) -> tuple[float, dict]:
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if not cat_cols:
            return 10.0, {"note": "No categorical columns"}
        uniform, issues = 0, []
        for col in cat_cols:
            s = df[col].dropna().astype(str)
            if len(s) == 0:
                uniform += 1
                continue
            has_whitespace = (s != s.str.strip()).any()
            has_null_str   = s.str.lower().isin(NULL_LIKE).any()
            if not has_whitespace and not has_null_str:
                uniform += 1
            else:
                issues.append(col)
        pct   = uniform / len(cat_cols) * 100
        score = pct / 100 * WEIGHTS["uniformity"]
        return round(score, 1), {"pct_uniform": round(pct, 1), "non_uniform_columns": issues[:5]}

    # ------------------------------------------------------------------
    # Claude verdict
    # ------------------------------------------------------------------

    def _interpret(
        self, df: pd.DataFrame, overall: float, dim_scores: dict, dim_details: dict, grade: str
    ) -> dict:
        prompt = f"""You are assessing the quality of a dataset with {len(df)} rows and {len(df.columns)} columns.

Overall Aria Score: {overall:.1f} / 100 (Grade: {grade})

Dimension scores (out of their max):
{json.dumps({k: {"score": round(v, 1), "max": WEIGHTS[k], "details": dim_details[k]}
             for k, v in dim_scores.items()}, indent=2, default=str)}

Write a concise one-paragraph verdict for a non-technical user. The verdict must:
- Start with the score and grade ("Your data scores X/100 — Grade Y.")
- Describe the data quality in plain English (Good/Fair/Poor)
- Name the 1-2 most important issues using actual column names or metrics from the details
- Be encouraging and constructive

Then provide exactly 3 short, specific, actionable recommendations to improve the score.

Return a JSON object:
{{
  "verdict": "single paragraph verdict",
  "recommendations": ["fix 1", "fix 2", "fix 3"]
}}

Return only valid JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
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
            return {"verdict": raw, "recommendations": []}
