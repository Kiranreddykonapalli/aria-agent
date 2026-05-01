"""
Agent Debate Mode: two AI analysts argue opposite interpretations of the same data,
then an impartial Aria Judge delivers a balanced verdict.

Structure:
  Round 1 — Optimist opens → Critic responds
  Round 2 — Optimist rebuts → Critic closes
  Judge   — Reviews all four arguments, delivers verdict + key insight

Each statement: 3-4 sentences, specific, references actual numbers from the analysis.
Each round is a separate Claude API call (no streaming — responses are short).
The Judge call uses the same model but with a different system prompt.

No column names are hardcoded — works with any analysis context.
"""

from __future__ import annotations

import json
import re

import anthropic

SYSTEM_OPTIMIST = """\
You are Agent A — an Optimist data analyst in a structured debate.
Your role is to argue the most positive, hopeful, and constructive interpretation of the data.
Find silver linings, positive trends, and reasons for confidence.
Be specific: cite actual numbers and column names from the analysis context.
Respond in exactly 3-4 sentences. Be persuasive and concrete. Do not hedge excessively."""

SYSTEM_CRITIC = """\
You are Agent B — a Critic data analyst in a structured debate.
Your role is to argue the most cautious, risk-focused interpretation and challenge the Optimist's claims.
Find the anomalies, risks, data quality concerns, and reasons for caution.
Be specific: challenge the exact numbers the Optimist cited, and bring in counter-evidence from the analysis.
Respond in exactly 3-4 sentences. Be incisive and evidence-based. Do not dismiss concerns."""

SYSTEM_JUDGE = """\
You are Aria, an impartial AI analyst acting as judge in a data debate.
You have seen both the Optimist and Critic arguments. Your job is to deliver the most honest,
balanced verdict based on the data evidence. You are not required to pick a winner — "balanced"
is a valid outcome. Be direct, reference the strongest point from each side, and give a single
key takeaway that a business leader should remember.
Always respond with valid JSON only. No markdown, no explanation outside the JSON."""


def _strip(text: str) -> str:
    """Remove markdown fences and strip whitespace."""
    return re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()


class DebateAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(
        self,
        analyst_output: dict,
        anomaly_output: dict,
        decision_output: dict,
        question: str,
    ) -> dict:
        """
        Run a two-round debate followed by a judge verdict.

        Args:
            analyst_output:  Output from Analyst — insights, stats, column_descriptions.
            anomaly_output:  Output from AnomalyAgent — anomalies, severity_counts, narrative.
            decision_output: Output from DecisionAgent — decisions, domain, summary.
            question:        Original user question that prompted the analysis.

        Returns:
            dict with keys:
              - "round1_optimist": str
              - "round1_critic":   str
              - "round2_optimist": str
              - "round2_critic":   str
              - "judge_verdict":   str
              - "winner":          "optimist" | "critic" | "balanced"
              - "key_insight":     str
        """
        context = self._build_context(analyst_output, anomaly_output, decision_output, question)

        # ── Round 1 ────────────────────────────────────────────────────
        r1_opt = self._call(
            system=SYSTEM_OPTIMIST,
            prompt=f"Make your opening argument based on this analysis:\n\n{context}",
        )
        r1_crit = self._call(
            system=SYSTEM_CRITIC,
            prompt=(
                f"The Optimist has argued:\n\"{r1_opt}\"\n\n"
                f"Challenge this argument using the analysis context below:\n\n{context}"
            ),
        )

        # ── Round 2 ────────────────────────────────────────────────────
        r2_opt = self._call(
            system=SYSTEM_OPTIMIST,
            prompt=(
                f"The Critic responded:\n\"{r1_crit}\"\n\n"
                f"Rebut the Critic's concerns with specific evidence from the analysis:\n\n{context}"
            ),
        )
        r2_crit = self._call(
            system=SYSTEM_CRITIC,
            prompt=(
                f"The Optimist rebutted:\n\"{r2_opt}\"\n\n"
                f"Give your closing argument with the strongest evidence for caution:\n\n{context}"
            ),
        )

        # ── Judge ───────────────────────────────────────────────────────
        judge_raw = self._call(
            system=SYSTEM_JUDGE,
            prompt=(
                f"You are judging a debate about this analysis:\n{context}\n\n"
                f"=== DEBATE TRANSCRIPT ===\n"
                f"Round 1 — Optimist:\n{r1_opt}\n\n"
                f"Round 1 — Critic:\n{r1_crit}\n\n"
                f"Round 2 — Optimist:\n{r2_opt}\n\n"
                f"Round 2 — Critic:\n{r2_crit}\n\n"
                "Deliver your verdict as a JSON object:\n"
                "{\n"
                '  "judge_verdict": "2-3 sentence balanced assessment of the data, '
                'referencing the strongest points from both sides",\n'
                '  "winner": "optimist" | "critic" | "balanced",\n'
                '  "key_insight": "single sentence — the most important thing a '
                'decision-maker should take away from this debate"\n'
                "}\n\nReturn only valid JSON."
            ),
            max_tokens=512,
        )

        try:
            judge = json.loads(_strip(judge_raw))
        except json.JSONDecodeError:
            judge = {
                "judge_verdict": judge_raw,
                "winner":        "balanced",
                "key_insight":   "Review both perspectives carefully before deciding.",
            }

        return {
            "round1_optimist": r1_opt,
            "round1_critic":   r1_crit,
            "round2_optimist": r2_opt,
            "round2_critic":   r2_crit,
            "judge_verdict":   judge.get("judge_verdict", ""),
            "winner":          judge.get("winner", "balanced"),
            "key_insight":     judge.get("key_insight", ""),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_context(
        self,
        ao: dict,
        ano: dict,
        do: dict,
        question: str,
    ) -> str:
        insights   = ao.get("insights", [])
        sc         = ano.get("severity_counts", {})
        top_anoms  = ano.get("anomalies", [])[:5]
        decisions  = do.get("decisions", [])
        domain     = do.get("domain", "")
        stats      = ao.get("stats", {}).get("_meta", {})

        lines = [
            f"Question: {question}",
            f"Domain: {domain}",
            f"Dataset: {stats.get('row_count','?')} rows × {stats.get('column_count','?')} columns",
            "",
            "KEY INSIGHTS:",
            *[f"  {i+1}. {ins}" for i, ins in enumerate(insights)],
            "",
            f"ANOMALIES: {sc.get('high',0)} high-severity, {sc.get('medium',0)} medium",
        ]
        for a in top_anoms:
            lines.append(f"  [{a.get('severity','').upper()}] {a.get('entity','')} — "
                         f"{a.get('column','')} = {a.get('value','')} | {a.get('reason','')[:70]}")

        lines += ["", "DECISIONS:"]
        for d in decisions:
            lines.append(f"  [{d.get('priority','')}] {d.get('action','')[:90]}")
            lines.append(f"    Rationale: {d.get('rationale','')[:70]}")

        return "\n".join(lines)

    def _call(self, system: str, prompt: str, max_tokens: int = 256) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
