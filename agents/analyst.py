"""
Analyst: runs statistical analysis and surfaces insights from clean data.

Responsibilities:
  - Profile all numeric and categorical columns (mean, median, std, skew, etc.)
  - Use Claude to classify what each column represents (metric/category/time/id/rank)
  - Use Claude to identify 5 key findings that answer the user's question
  - Use Claude to suggest 3-4 charts that best visualise those findings
  - Return structured output consumed by VizBuilder and ReportWriter

All Claude calls use a shared JSON-only system prompt with prompt caching.
No column names are hardcoded — works with any tabular dataset.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import anthropic

SYSTEM_PROMPT = (
    "You are a data analyst. Always respond with valid JSON only. "
    "Do not include markdown formatting, code fences, or any explanation outside the JSON."
)


class Analyst:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(self, wrangler_output: dict, question: str, plan: list[str] | None = None) -> dict:
        """
        Profile the dataset, detect column roles, surface insights, and suggest charts.

        Args:
            wrangler_output: Output from DataWrangler.run() — must contain "dataframe".
            question: Natural-language question the pipeline is answering.
            plan: Optional list of sub-tasks from the Orchestrator (unused internally
                  but passed through for future orchestration logic).

        Returns:
            dict with keys:
              - "stats":               dict  — descriptive stats for every column
              - "column_descriptions": dict  — Claude's classification of each column
              - "insights":            list[str] — 5 key findings as plain-English sentences
              - "suggested_charts":    list[dict] — 3-4 chart specs (type/x/y/title)
              - "question":            str  — original question, passed through for ReportWriter
        """
        df: pd.DataFrame = wrangler_output["dataframe"]

        stats = self._profile_data(df)
        column_descriptions = self._detect_columns(df, stats)
        insights = self._find_insights(stats, column_descriptions, question)
        suggested_charts = self._suggest_charts(df, stats, column_descriptions, insights)

        return {
            "stats": stats,
            "column_descriptions": column_descriptions,
            "insights": insights,
            "suggested_charts": suggested_charts,
            "question": question,
        }

    # ------------------------------------------------------------------
    # Step 1 — pure pandas profiling (no Claude)
    # ------------------------------------------------------------------

    def _profile_data(self, df: pd.DataFrame) -> dict:
        """
        Compute descriptive stats for every column.

        Numeric columns get: mean, median, std, min, max, q25, q75, skewness,
        kurtosis, null_count, unique_count.
        Categorical columns get: unique_count, top_values (up to 10), null_count.
        A "_meta" key holds overall dataset shape and column-type lists.
        """
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(exclude="number").columns.tolist()

        stats: dict = {
            "_meta": {
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "numeric_columns": numeric_cols,
                "categorical_columns": cat_cols,
            }
        }

        for col in numeric_cols:
            s = df[col].dropna()
            stats[col] = {
                "mean":         round(float(s.mean()), 6),
                "median":       round(float(s.median()), 6),
                "std":          round(float(s.std()), 6),
                "min":          round(float(s.min()), 6),
                "max":          round(float(s.max()), 6),
                "q25":          round(float(s.quantile(0.25)), 6),
                "q75":          round(float(s.quantile(0.75)), 6),
                "skewness":     round(float(s.skew()), 4),
                "kurtosis":     round(float(s.kurtosis()), 4),
                "null_count":   int(df[col].isnull().sum()),
                "unique_count": int(s.nunique()),
            }

        for col in cat_cols:
            s = df[col]
            top = {str(k): int(v) for k, v in s.value_counts().head(10).items()}
            stats[col] = {
                "type":         "categorical",
                "unique_count": int(s.nunique()),
                "top_values":   top,
                "null_count":   int(s.isnull().sum()),
            }

        return stats

    # ------------------------------------------------------------------
    # Step 2 — column classification (Claude)
    # ------------------------------------------------------------------

    def _detect_columns(self, df: pd.DataFrame, stats: dict) -> dict:
        """
        Ask Claude to classify the role of every column given its name, dtype,
        and sample values. Returns a dict keyed by column name.
        """
        col_info: dict = {}
        for col in df.columns:
            raw_samples = df[col].dropna().head(5).tolist()
            col_info[col] = {
                "dtype": str(df[col].dtype),
                "sample_values": self._to_json_safe(raw_samples),
                "unique_count": int(df[col].nunique()),
            }

        prompt = f"""Analyze these dataset columns and classify each one.

Column information (name → dtype, sample values, unique count):
{json.dumps(col_info, indent=2)}

Return a JSON object where each key is a column name and the value is:
{{
  "role": "metric" | "category" | "time" | "id" | "rank",
  "description": "one sentence describing what this column represents",
  "unit": "unit of measurement, or null if not applicable"
}}

Role definitions:
- metric:   a quantitative measurement (rate, count, dollar amount, days, score)
- category: a grouping label with no intrinsic numeric meaning (name, type, region)
- time:     a temporal value (year, date, month)
- id:       a unique row identifier with no analytical value
- rank:     an ordinal position within a group (1st, 2nd … nth)

Return one entry per column. Return only valid JSON, no other text."""

        raw = self._call_claude(prompt, max_tokens=1500)
        return self._parse_json(raw, "column_descriptions")

    # ------------------------------------------------------------------
    # Step 3 — insight extraction (Claude)
    # ------------------------------------------------------------------

    def _find_insights(
        self, stats: dict, column_descriptions: dict, question: str
    ) -> list[str]:
        """
        Ask Claude to surface 5 key findings from the profiled statistics that
        directly address the user's question.
        """
        stats_summary = {
            col: vals for col, vals in stats.items() if col != "_meta"
        }

        prompt = f"""You are analyzing a dataset to answer this question:
"{question}"

Dataset overview:
- Rows: {stats["_meta"]["row_count"]}
- Columns: {stats["_meta"]["column_count"]}

Column descriptions:
{json.dumps(column_descriptions, indent=2)}

Descriptive statistics:
{json.dumps(stats_summary, indent=2)}

Identify the 5 most important findings that help answer the question.
Prioritise: notable distributions, outliers, correlations implied by the stats,
temporal trends (if a time column exists), and comparisons across categories.

Return a JSON array of exactly 5 strings. Each string must be a complete,
specific finding that references actual column names and numeric values.
Example format: ["Finding one referencing column X (value Y).", "Finding two.", ...]

Return only the JSON array. No other text."""

        raw = self._call_claude(prompt, max_tokens=1500)
        result = self._parse_json(raw, "insights")
        if not isinstance(result, list):
            raise ValueError(f"Claude returned {type(result).__name__} for insights; expected a list.")
        return [str(item) for item in result]

    # ------------------------------------------------------------------
    # Step 4 — chart suggestions (Claude)
    # ------------------------------------------------------------------

    def _suggest_charts(
        self,
        df: pd.DataFrame,
        stats: dict,
        column_descriptions: dict,
        insights: list[str],
    ) -> list[dict]:
        """
        Ask Claude to recommend 3-4 charts that best visualise the key findings.
        Returns a list of dicts consumed directly by VizBuilder.
        """
        column_roles = {col: desc.get("role") for col, desc in column_descriptions.items()}

        prompt = f"""Recommend 3-4 charts to visualise the key findings from this dataset.

Available columns: {json.dumps(list(df.columns))}

Column roles:
{json.dumps(column_roles, indent=2)}

Key findings:
{json.dumps(insights, indent=2)}

Dataset: {stats["_meta"]["row_count"]} rows x {stats["_meta"]["column_count"]} columns

Return a JSON array of 3-4 chart objects. Each object must have exactly these keys:
{{
  "type":  "histogram" | "bar" | "scatter" | "line" | "heatmap",
  "x":     "column name for x-axis, or null if not applicable",
  "y":     "column name for y-axis, or null if not applicable",
  "title": "a specific, descriptive chart title"
}}

Rules — follow these strictly:
- Only use column names that appear in the available columns list above.
- histogram: x = column to distribute, y = null.
- heatmap:   x = null, y = null (plots correlation matrix of all numeric columns).
- line:      x must be a "time" or ordered column; y must be a "metric" column.
- bar:       x must be a "category" column; y must be a "metric" column.
- scatter:   x and y must both be "metric" columns.
- Prefer charts that directly illuminate the key findings.
- Vary chart types — do not return four of the same type.

Return only the JSON array. No other text."""

        raw = self._call_claude(prompt, max_tokens=1000)
        result = self._parse_json(raw, "suggested_charts")
        if not isinstance(result, list):
            raise ValueError(f"Claude returned {type(result).__name__} for suggested_charts; expected a list.")
        return result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _call_claude(self, prompt: str, max_tokens: int = 1024) -> str:
        """
        Call Claude with a cached JSON-only system prompt. Returns response text.
        Prompt caching on the system prompt avoids re-sending it on every Analyst call.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _parse_json(self, text: str, context: str) -> dict | list:
        """
        Parse JSON from a Claude response. Strips markdown code fences if present.
        Raises ValueError with the raw response on failure so the caller can diagnose.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Failed to parse JSON from Claude ({context}).\n"
                f"Parse error: {exc}\n"
                f"Raw response (first 500 chars): {text[:500]}"
            ) from exc

    @staticmethod
    def _to_json_safe(values: list) -> list:
        """Convert numpy scalar types to native Python so json.dumps doesn't fail."""
        safe = []
        for v in values:
            if isinstance(v, np.integer):
                safe.append(int(v))
            elif isinstance(v, np.floating):
                safe.append(float(v))
            else:
                safe.append(v)
        return safe
