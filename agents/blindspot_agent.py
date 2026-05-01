"""
Blind Spot Detector Agent: identifies what the analysis missed.

Sends Claude the full analysis context (what WAS covered) and the user's question,
then asks it to find exactly 3 gaps — unexplored columns, untested relationships,
ignored segments, and unanswered follow-ups.

Each blind spot has:
  - title:              what was missed (short label)
  - why_it_matters:     specific impact with actual numbers from the data
  - suggested_question: the exact question the user should ask next
  - severity:           Critical / Important / Minor

No column names or domain terms are hardcoded — works with any dataset.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import anthropic

SYSTEM_PROMPT = (
    "You are a critical data analyst auditor. "
    "Always respond with valid JSON only. No markdown, no explanation outside the JSON."
)


class BlindSpotAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(
        self, dataframe: pd.DataFrame, analyst_output: dict, question: str
    ) -> dict:
        """
        Identify exactly 3 blind spots in the analysis.

        Args:
            dataframe:      Cleaned DataFrame from DataWrangler.
            analyst_output: Output from Analyst — insights, column_descriptions, stats.
            question:       Original user question.

        Returns:
            dict with keys:
              - "blind_spots": list[dict] — exactly 3 gaps with title/why_it_matters/
                               suggested_question/severity
              - "summary":     str — one-paragraph overview of what was missed
        """
        context = self._build_context(dataframe, analyst_output, question)
        result  = self._detect(context, dataframe, analyst_output, question)
        return result

    # ------------------------------------------------------------------

    def _build_context(
        self, df: pd.DataFrame, ao: dict, question: str
    ) -> str:
        col_desc = ao.get("column_descriptions", {})
        insights = ao.get("insights", [])
        stats    = ao.get("stats", {}).get("_meta", {})

        # Columns mentioned in insights (approximate — just scan text)
        insight_text = " ".join(insights).lower()
        mentioned    = [col for col in df.columns if col.lower() in insight_text]
        unmentioned  = [col for col in df.columns if col.lower() not in insight_text]

        col_summary = "\n".join(
            f"  {col} [{col_desc.get(col, {}).get('role', '?')}]: "
            f"{col_desc.get(col, {}).get('description', '')}"
            for col in df.columns
        )

        # Key numeric stats for context
        stat_lines = []
        for col, s in ao.get("stats", {}).items():
            if col == "_meta" or "mean" not in s:
                continue
            stat_lines.append(
                f"  {col}: mean={s['mean']:.4g}, min={s['min']:.4g}, "
                f"max={s['max']:.4g}, std={s.get('std', 0):.4g}"
            )

        return (
            f"User question: \"{question}\"\n"
            f"Dataset: {stats.get('row_count','?')} rows × {stats.get('column_count','?')} columns\n\n"
            f"ALL COLUMNS:\n{col_summary}\n\n"
            f"COLUMNS MENTIONED IN INSIGHTS: {mentioned}\n"
            f"COLUMNS NOT MENTIONED IN INSIGHTS: {unmentioned}\n\n"
            f"KEY STATISTICS:\n" + "\n".join(stat_lines) + "\n\n"
            f"INSIGHTS THAT WERE PROVIDED:\n"
            + "\n".join(f"  {i+1}. {ins}" for i, ins in enumerate(insights))
        )

    def _detect(
        self, context: str, df: pd.DataFrame, ao: dict, question: str
    ) -> dict:
        prompt = f"""You are auditing a data analysis for gaps and blind spots.

{context}

Identify exactly 3 blind spots — things the analysis FAILED to cover given the user's question.
Consider:
  1. Columns that exist in the data but were never mentioned in the insights
  2. Important relationships between columns that were not explored
  3. Time periods, geographic segments, or sub-groups that were ignored
  4. Follow-up questions the analysis should have answered but didn't

Return a JSON object:
{{
  "blind_spots": [
    {{
      "title": "Short label for what was missed (5-10 words)",
      "why_it_matters": "1-2 specific sentences explaining the impact — reference actual column names and numbers from the statistics above",
      "suggested_question": "The exact question the user should ask next — must be answerable from this dataset",
      "severity": "Critical" | "Important" | "Minor"
    }},
    {{ ... }},
    {{ ... }}
  ],
  "summary": "One paragraph explaining collectively what the analysis missed and why these gaps matter"
}}

Rules:
- Exactly 3 blind spots — no more, no fewer.
- At least one must be "Critical" severity.
- suggested_question must reference specific column names that exist in the dataset.
- why_it_matters must cite at least one actual number from the statistics.
- Return only valid JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
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
            result = json.loads(cleaned)
            # Ensure exactly 3 blind spots
            spots = result.get("blind_spots", [])[:3]
            return {
                "blind_spots": spots,
                "summary":     result.get("summary", ""),
            }
        except json.JSONDecodeError:
            return {
                "blind_spots": [],
                "summary":     raw,
            }
