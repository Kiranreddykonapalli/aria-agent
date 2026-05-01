"""
Decision Intelligence Agent: converts data findings into concrete, actionable decisions.

Given the user question, analyst insights, anomaly findings, and raw dataframe,
Claude produces exactly 5 prioritised decisions — each with a specific action,
data-grounded rationale, expected impact, and timeline.

The prompt enforces specificity: no vague recommendations.
Every action must name a concrete intervention, target entity, and deadline.
"""

from __future__ import annotations

import json
import re

import pandas as pd
import anthropic

SYSTEM_PROMPT = """\
You are a senior decision intelligence consultant. Your job is to convert data analysis
findings into concrete, board-ready recommendations that a decision-maker can act on today.

Rules you must follow:
- Every action must be SPECIFIC: name the exact intervention, the target (entity/region/team),
  and a concrete deadline or milestone. NEVER write vague actions like "improve outcomes"
  or "increase awareness". Instead write: "Deploy 3 mobile screening units to Osceola County
  by Q3 2026 targeting the 26% diabetes prevalence identified in the data."
- Rationale must cite actual numbers from the data provided.
- Expected impact must be quantified wherever possible ("reduce X by ~Y%", "reach N people").
- Priorities: use exactly "Critical", "High", or "Medium" — nothing else.
- Timelines: use exactly "Immediate", "30 days", "90 days", or "6 months" — nothing else.
- Always respond with valid JSON only. No markdown, no explanation outside the JSON.
"""


class DecisionAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(
        self,
        question: str,
        analyst_output: dict,
        anomaly_output: dict,
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Generate 5 concrete, prioritised decisions from the full analysis context.

        Args:
            question:        Original user question.
            analyst_output:  Output from Analyst — insights, column_descriptions, stats.
            anomaly_output:  Output from AnomalyAgent — anomalies, narrative, severity_counts.
            dataframe:       Cleaned DataFrame from DataWrangler (used for data summary).

        Returns:
            dict with keys:
              - "decisions": list[dict] — 5 decisions, each with:
                    priority, action, rationale, expected_impact, timeline
              - "summary":   str  — one-paragraph executive summary of the decision set
              - "domain":    str  — Claude's assessment of the data domain / industry
        """
        prompt = self._build_prompt(question, analyst_output, anomaly_output, dataframe)
        raw    = self._call_claude(prompt)
        result = self._parse(raw)
        return result

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        analyst_output: dict,
        anomaly_output: dict,
        dataframe: pd.DataFrame,
    ) -> str:
        insights    = analyst_output.get("insights", [])
        col_desc    = analyst_output.get("column_descriptions", {})
        stats       = analyst_output.get("stats", {})
        meta        = stats.get("_meta", {})
        anomalies   = anomaly_output.get("anomalies", [])
        anom_narr   = anomaly_output.get("narrative", "")
        sev_counts  = anomaly_output.get("severity_counts", {})

        # Compact column summary
        col_summary = "\n".join(
            f"  {col}: {info.get('role','?')} — {info.get('description','')} "
            f"{'(' + info['unit'] + ')' if info.get('unit') else ''}"
            for col, info in col_desc.items()
        )

        # Top numeric stats (mean / min / max) for context
        stat_lines = []
        for col, s in stats.items():
            if col == "_meta" or "mean" not in s:
                continue
            stat_lines.append(
                f"  {col}: mean={s['mean']:.4g}, min={s['min']:.4g}, max={s['max']:.4g}"
            )
        stat_summary = "\n".join(stat_lines)

        # Top anomalies (high severity first, capped at 8)
        top_anomalies = [a for a in anomalies if a.get("severity") == "high"][:8]
        if len(top_anomalies) < 5:
            top_anomalies += [
                a for a in anomalies if a.get("severity") != "high"
            ][: 5 - len(top_anomalies)]
        anomaly_lines = "\n".join(
            f"  [{a['severity'].upper()}] {a['entity']} — {a['column']} = {a['value']} | {a['reason']}"
            for a in top_anomalies
        )

        return f"""USER QUESTION:
{question}

DATASET OVERVIEW:
  Rows: {meta.get('row_count', len(dataframe))}
  Columns: {meta.get('column_count', len(dataframe.columns))}

COLUMN DESCRIPTIONS:
{col_summary}

KEY STATISTICS:
{stat_summary}

ANALYST INSIGHTS (from statistical analysis):
{chr(10).join(f'  {i+1}. {s}' for i, s in enumerate(insights))}

ANOMALY SUMMARY:
  High severity: {sev_counts.get('high', 0)} | Medium: {sev_counts.get('medium', 0)}
  Narrative: {anom_narr[:600]}

TOP ANOMALIES:
{anomaly_lines}

---

Your task: produce exactly 5 concrete, prioritised decisions a decision-maker can act on NOW.

Return a JSON object with exactly this structure:
{{
  "domain": "one-sentence description of the industry/domain this data represents",
  "summary": "one paragraph summarising the 5 decisions and their collective expected impact",
  "decisions": [
    {{
      "priority": "Critical" | "High" | "Medium",
      "action": "Specific action with named target, intervention, and deadline",
      "rationale": "Why — tied to specific numbers from the data above",
      "expected_impact": "Quantified outcome if action is taken",
      "timeline": "Immediate" | "30 days" | "90 days" | "6 months"
    }}
  ]
}}

Requirements:
- Exactly 5 decisions in the "decisions" array, ordered by priority (Critical first)
- At least one "Critical" priority decision
- Each "action" must be a complete sentence naming WHO does WHAT to WHOM/WHERE by WHEN
- Each "rationale" must quote at least one specific number from the data
- "expected_impact" should be quantified (%, count, dollar amount) wherever the data supports it
- Return only valid JSON — no markdown, no preamble"""

    def _call_claude(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
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

    def _parse(self, raw: str) -> dict:
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"DecisionAgent: failed to parse Claude response as JSON.\n"
                f"Error: {exc}\nRaw (first 500 chars): {raw[:500]}"
            ) from exc

        decisions = data.get("decisions", [])
        if not isinstance(decisions, list):
            raise ValueError("DecisionAgent: 'decisions' is not a list in Claude response.")

        # Normalise each decision to guarantee required keys are present
        required = {"priority", "action", "rationale", "expected_impact", "timeline"}
        for i, d in enumerate(decisions):
            missing = required - set(d.keys())
            if missing:
                raise ValueError(
                    f"DecisionAgent: decision {i+1} is missing keys: {missing}"
                )

        return {
            "decisions": decisions[:5],
            "summary":   data.get("summary", ""),
            "domain":    data.get("domain", ""),
        }
