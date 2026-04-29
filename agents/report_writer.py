"""
Report Writer: composes the final human-readable analytics report.

Responsibilities:
  - Accept user question, analyst output (insights/column_descriptions/suggested_charts),
    and viz output (figure_paths)
  - Use Claude to write a professional markdown report with five sections:
      1. Executive Summary
      2. Key Findings
      3. Data Visualizations
      4. Recommendations
      5. Methodology
  - Embed each saved figure using markdown image syntax
  - Apply prompt caching to the system prompt (large, reusable across runs)
  - Save to output/reports/<ISO-timestamp>_report.md
  - Return dict with keys: report_path, report_text

Works with any dataset — no column names or domain terms are hardcoded.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import anthropic

OUTPUT_DIR = "output/reports"

SYSTEM_PROMPT = """\
You are a senior data analyst writing a professional analytics report for a business stakeholder.
Your writing is clear, precise, and avoids unnecessary jargon.

When given a user question, dataset description, and analysis findings, you will produce a
complete markdown report with exactly these five sections in this order:

## Executive Summary
One concise paragraph (4-6 sentences) that directly answers the user's question.
State the most important conclusion up front. Do not list raw numbers here —
synthesise the story the data tells.

## Key Findings
Exactly 5 bullet points, one per analyst insight provided. Each bullet must:
- Start with a bold claim (the finding in plain English)
- Follow with 1-2 sentences of context or implication
- Reference specific numeric values from the insight

## Data Visualizations
A subsection for each chart, using this format:
### <Chart Title>
One sentence describing what the chart shows and its key takeaway.
![<Chart Title>](<figure_path>)

## Recommendations
3-5 numbered, actionable recommendations that follow directly from the findings.
Each recommendation must name a specific stakeholder or action owner where possible.
Be concrete — avoid vague advice like "improve health outcomes."

## Methodology
2-3 sentences covering: what dataset was used, what statistical profiling was performed,
and how Claude was used in the analysis pipeline.

Rules:
- Use markdown formatting throughout (##, ###, **, bullet lists, numbered lists)
- Never fabricate data — only reference values present in the findings provided
- Keep the total report under 1000 words
- Do not include a title at the top — the caller adds that
"""


class ReportWriter:
    def __init__(self, model: str = "claude-sonnet-4-6", output_dir: str = OUTPUT_DIR):
        self.model = model
        self.output_dir = output_dir
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        question: str,
        analyst_output: dict,
        viz_output: dict,
    ) -> dict:
        """
        Write and save the final markdown report.

        Args:
            question:        Original user question.
            analyst_output:  Output from Analyst.run() — expects keys:
                               insights, column_descriptions, suggested_charts, stats.
            viz_output:      Output from VizBuilder.run() — expects key: figure_paths.

        Returns:
            dict with keys:
              - "report_text": str  — full markdown report
              - "report_path": str  — path to saved .md file
        """
        figure_paths: list[str] = viz_output.get("figure_paths", [])
        charts: list[dict]      = analyst_output.get("suggested_charts", [])

        prompt = self._build_prompt(question, analyst_output, figure_paths, charts)
        body   = self._call_claude(prompt)
        report = self._assemble(question, body, figure_paths, charts)
        path   = self._save(report)

        return {"report_text": report, "report_path": path}

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        analyst_output: dict,
        figure_paths: list[str],
        charts: list[dict],
    ) -> str:
        """
        Assemble the user-turn prompt from question, column descriptions,
        insights, chart metadata, and figure paths.
        """
        insights: list[str]          = analyst_output.get("insights", [])
        col_desc: dict               = analyst_output.get("column_descriptions", {})
        stats: dict                  = analyst_output.get("stats", {})
        meta: dict                   = stats.get("_meta", {})

        # Compact column summary (role + description only, no raw stats)
        col_summary = "\n".join(
            f"  - {col}: {info.get('role', '?')} — {info.get('description', '')}"
            for col, info in col_desc.items()
        )

        # Numbered insights
        insights_block = "\n".join(
            f"  {i+1}. {insight}" for i, insight in enumerate(insights)
        )

        # Chart list with matched figure paths
        charts_block_lines = []
        for i, chart in enumerate(charts):
            path = figure_paths[i] if i < len(figure_paths) else "(not rendered)"
            charts_block_lines.append(
                f"  - Title: {chart.get('title', 'Chart')}\n"
                f"    Type:  {chart.get('type', '?')}  |  "
                f"x: {chart.get('x')}  |  y: {chart.get('y')}\n"
                f"    Path:  {path}"
            )
        charts_block = "\n".join(charts_block_lines)

        return f"""USER QUESTION:
{question}

DATASET OVERVIEW:
  Rows: {meta.get('row_count', 'unknown')}
  Columns: {meta.get('column_count', 'unknown')}

COLUMN DESCRIPTIONS:
{col_summary}

KEY FINDINGS FROM STATISTICAL ANALYSIS:
{insights_block}

CHARTS PRODUCED (use these exact paths in the Visualizations section):
{charts_block}

Write the full five-section markdown report as specified. Use the exact figure paths
provided above in the markdown image tags. Do not add a top-level title."""

    def _call_claude(self, prompt: str) -> str:
        """
        Call Claude with the system prompt cached (ephemeral).
        The system prompt is large and identical across every report run,
        so caching it avoids re-tokenising it on each call.
        """
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

    def _assemble(
        self,
        question: str,
        body: str,
        figure_paths: list[str],
        charts: list[dict],
    ) -> str:
        """
        Prepend a report title and timestamp header to Claude's response.
        Also ensures any figure paths Claude may have omitted are appended
        to the Visualizations section as a fallback.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = f"# Analytics Report\n\n**Question:** {question}\n\n**Generated:** {timestamp}\n\n---\n\n"

        # Fallback: if Claude didn't embed a path, append it after the body
        missing: list[str] = []
        for i, path in enumerate(figure_paths):
            if path not in body:
                title = charts[i].get("title", f"Chart {i+1}") if i < len(charts) else f"Chart {i+1}"
                missing.append(f"### {title}\n\n![{title}]({path})\n")

        suffix = ("\n\n---\n\n" + "\n".join(missing)) if missing else ""
        return header + body + suffix

    def _save(self, report_text: str) -> str:
        """Write the report to output/reports/ with an ISO-UTC timestamp prefix."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename  = f"{timestamp}_report.md"
        path      = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return path
