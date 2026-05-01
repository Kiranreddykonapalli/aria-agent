"""
Statistical Test Agent: selects and runs appropriate hypothesis tests for any dataset.

Auto-detects which tests are relevant based on column types and the user's question:
  - Pearson / Spearman correlation : relationship between two numeric columns
  - T-test                         : compare means of exactly 2 groups
  - Mann-Whitney U                 : non-parametric alternative when data is non-normal
  - ANOVA                          : compare means of 3+ groups
  - Chi-square                     : relationship between two categorical columns

Effect sizes are computed for every test so magnitude, not just significance, is reported.
Claude interprets results in plain English tied to actual numbers.

No column names are hardcoded — works with any tabular dataset.
"""

from __future__ import annotations

import json
import re
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import anthropic

warnings.filterwarnings("ignore", category=RuntimeWarning)

SYSTEM_PROMPT = (
    "You are a statistician. Always respond with valid JSON only. "
    "No markdown, no explanation outside the JSON."
)

P_THRESHOLD         = 0.05
MAX_CORRELATION_PAIRS = 30   # cap to keep output readable
MAX_GROUP_TESTS     = 10     # cap group-comparison tests per categorical column
MAX_CAT_GROUPS      = 20     # skip categoricals with more unique values than this
NORMALITY_SAMPLE    = 50     # rows sampled for Shapiro-Wilk (avoids large-n warnings)
MIN_GROUP_SIZE      = 5      # minimum observations per group for group tests


def _effect_label(r: float) -> str:
    """Cohen's conventions for |r|."""
    r = abs(r)
    if r >= 0.5:  return "large"
    if r >= 0.3:  return "medium"
    return "small"


def _eta_squared(groups: list[np.ndarray]) -> float:
    """η² for one-way ANOVA."""
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total   = sum(((v - grand_mean) ** 2) for g in groups for v in g)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent groups."""
    n1, n2  = len(a), len(b)
    pooled  = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    """Cramér's V effect size for chi-square."""
    denom = n * (min(r, c) - 1)
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def _mwu_effect_r(u: float, n1: int, n2: int) -> float:
    """Effect size r for Mann-Whitney U."""
    mean_u = n1 * n2 / 2
    std_u  = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z      = (u - mean_u) / std_u if std_u > 0 else 0
    return float(abs(z) / np.sqrt(n1 + n2))


def _is_normal(series: pd.Series) -> bool:
    sample = series.dropna().sample(min(NORMALITY_SAMPLE, len(series)), random_state=42)
    try:
        _, p = scipy.stats.shapiro(sample)
        return p > P_THRESHOLD
    except Exception:
        return True


class StatsAgent:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model  = model
        # No-arg constructor: auto-detects ANTHROPIC_API_KEY or active Claude Code session.
        self.client = anthropic.Anthropic()

    def run(self, dataframe: pd.DataFrame, analyst_output: dict, question: str) -> dict:
        """
        Select and run statistical tests appropriate for this dataset and question.

        Args:
            dataframe:      Cleaned DataFrame from DataWrangler.
            analyst_output: Output from Analyst — uses column_descriptions.
            question:       Original user question (guides Claude's interpretation).

        Returns:
            dict with keys:
              - "tests_run":           list[dict] — every test with statistic/p/effect
              - "significant_findings": list[dict] — subset where p < 0.05
              - "narrative":           str  — Claude's plain-English interpretation
              - "recommendations":     list[str] — actionable follow-up suggestions
        """
        col_desc    = analyst_output.get("column_descriptions", {})
        numeric_cols = self._numeric_cols(dataframe, col_desc)
        group_cols   = self._group_cols(dataframe, col_desc)

        tests: list[dict] = []
        tests.extend(self._run_correlations(dataframe, numeric_cols))
        for gcol in group_cols:
            tests.extend(self._run_group_tests(dataframe, gcol, numeric_cols))
        tests.extend(self._run_chisquare(dataframe, col_desc))

        significant = [t for t in tests if t["significant"]]
        result      = self._interpret(tests, significant, question)

        return {
            "tests_run":            tests,
            "significant_findings": significant,
            "narrative":            result.get("narrative", ""),
            "recommendations":      result.get("recommendations", []),
        }

    # ------------------------------------------------------------------
    # Column helpers
    # ------------------------------------------------------------------

    def _numeric_cols(self, df: pd.DataFrame, col_desc: dict) -> list[str]:
        skip = {"id", "rank", "category", "time"}
        return [
            c for c in df.select_dtypes(include="number").columns
            if col_desc.get(c, {}).get("role", "metric") not in skip
        ]

    def _group_cols(self, df: pd.DataFrame, col_desc: dict) -> list[str]:
        """Categorical or time columns with ≤ MAX_CAT_GROUPS unique values."""
        eligible_roles = {"category", "time"}
        return [
            col for col, desc in col_desc.items()
            if desc.get("role") in eligible_roles
            and col in df.columns
            and df[col].nunique() <= MAX_CAT_GROUPS
        ]

    # ------------------------------------------------------------------
    # Correlation tests
    # ------------------------------------------------------------------

    def _run_correlations(self, df: pd.DataFrame, cols: list[str]) -> list[dict]:
        results = []
        pairs   = [(cols[i], cols[j]) for i in range(len(cols)) for j in range(i + 1, len(cols))]

        for col1, col2 in pairs[:MAX_CORRELATION_PAIRS]:
            subset = df[[col1, col2]].dropna()
            n      = len(subset)
            if n < 6:
                continue

            normal1 = _is_normal(subset[col1])
            normal2 = _is_normal(subset[col2])

            if normal1 and normal2:
                stat, p   = scipy.stats.pearsonr(subset[col1], subset[col2])
                test_name = "Pearson Correlation"
            else:
                stat, p   = scipy.stats.spearmanr(subset[col1], subset[col2])
                test_name = "Spearman Correlation"

            r      = float(stat)
            effect = abs(r)
            results.append({
                "test_name":      test_name,
                "columns_tested": [col1, col2],
                "statistic":      round(r, 4),
                "p_value":        round(float(p), 6),
                "significant":    float(p) < P_THRESHOLD,
                "effect_size":    round(effect, 4),
                "effect_label":   _effect_label(effect),
                "n":              n,
                "direction":      "positive" if r > 0 else "negative",
            })

        # Sort: significant + largest effect first
        results.sort(key=lambda x: (not x["significant"], -x["effect_size"]))
        return results

    # ------------------------------------------------------------------
    # Group comparison tests (t-test / ANOVA / Mann-Whitney)
    # ------------------------------------------------------------------

    def _run_group_tests(
        self, df: pd.DataFrame, group_col: str, metric_cols: list[str]
    ) -> list[dict]:
        results   = []
        groups_sr = df[group_col].dropna().unique()
        n_groups  = len(groups_sr)

        for metric in metric_cols[:MAX_GROUP_TESTS]:
            sub     = df[[group_col, metric]].dropna()
            groups  = [sub.loc[sub[group_col] == g, metric].values for g in groups_sr]
            groups  = [g for g in groups if len(g) >= MIN_GROUP_SIZE]
            if len(groups) < 2:
                continue

            if n_groups == 2:
                g1, g2   = groups[0], groups[1]
                n1, n2   = len(g1), len(g2)
                use_para = _is_normal(pd.Series(g1)) and _is_normal(pd.Series(g2))

                if use_para:
                    stat, p   = scipy.stats.ttest_ind(g1, g2)
                    d         = _cohen_d(g1, g2)
                    test_name = "Independent T-Test"
                    effect    = abs(d)
                    e_label   = _effect_label(min(abs(d) / 2, 1))  # convert d to approx r
                else:
                    stat, p   = scipy.stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    d         = _mwu_effect_r(float(stat), n1, n2)
                    test_name = "Mann-Whitney U"
                    effect    = d
                    e_label   = _effect_label(d)

                results.append({
                    "test_name":      test_name,
                    "columns_tested": [group_col, metric],
                    "groups":         [str(g) for g in groups_sr[:2]],
                    "statistic":      round(float(stat), 4),
                    "p_value":        round(float(p), 6),
                    "significant":    float(p) < P_THRESHOLD,
                    "effect_size":    round(effect, 4),
                    "effect_label":   e_label,
                    "n":              n1 + n2,
                    "group_means":    {str(groups_sr[i]): round(float(groups[i].mean()), 4)
                                       for i in range(2)},
                })
            else:
                # ANOVA
                stat, p = scipy.stats.f_oneway(*groups)
                eta2    = _eta_squared(groups)
                results.append({
                    "test_name":      "One-Way ANOVA",
                    "columns_tested": [group_col, metric],
                    "groups":         [str(g) for g in groups_sr],
                    "statistic":      round(float(stat), 4),
                    "p_value":        round(float(p), 6),
                    "significant":    float(p) < P_THRESHOLD,
                    "effect_size":    round(eta2, 4),
                    "effect_label":   ("large" if eta2 >= 0.14 else "medium" if eta2 >= 0.06 else "small"),
                    "n":              sum(len(g) for g in groups),
                })

        results.sort(key=lambda x: (not x["significant"], -x["effect_size"]))
        return results

    # ------------------------------------------------------------------
    # Chi-square tests
    # ------------------------------------------------------------------

    def _run_chisquare(self, df: pd.DataFrame, col_desc: dict) -> list[dict]:
        cat_cols = [
            col for col, desc in col_desc.items()
            if desc.get("role") == "category"
            and col in df.columns
            and 2 <= df[col].nunique() <= MAX_CAT_GROUPS
        ]
        results = []
        pairs   = [(cat_cols[i], cat_cols[j])
                   for i in range(len(cat_cols))
                   for j in range(i + 1, len(cat_cols))]

        for col1, col2 in pairs[:5]:
            sub = df[[col1, col2]].dropna()
            if len(sub) < 20:
                continue
            ct   = pd.crosstab(sub[col1], sub[col2])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            try:
                chi2, p, _, _ = scipy.stats.chi2_contingency(ct)
                v             = _cramers_v(chi2, len(sub), ct.shape[0], ct.shape[1])
                results.append({
                    "test_name":      "Chi-Square Test",
                    "columns_tested": [col1, col2],
                    "statistic":      round(float(chi2), 4),
                    "p_value":        round(float(p), 6),
                    "significant":    float(p) < P_THRESHOLD,
                    "effect_size":    round(v, 4),
                    "effect_label":   _effect_label(v),
                    "n":              len(sub),
                })
            except Exception:
                continue

        return results

    # ------------------------------------------------------------------
    # Claude interpretation
    # ------------------------------------------------------------------

    def _interpret(
        self, all_tests: list[dict], significant: list[dict], question: str
    ) -> dict:
        if not all_tests:
            return {
                "narrative":       "No statistical tests could be run on this dataset.",
                "recommendations": [],
            }

        # Send compact summary to Claude
        sig_summary = [
            {k: v for k, v in t.items() if k not in ("groups", "group_means")}
            for t in significant[:15]
        ]
        all_summary = {
            "total_tests":       len(all_tests),
            "significant_count": len(significant),
            "test_types_run":    list({t["test_name"] for t in all_tests}),
        }

        prompt = f"""You are interpreting statistical test results to answer:
"{question}"

Overview:
{json.dumps(all_summary, indent=2)}

Significant findings (p < 0.05):
{json.dumps(sig_summary, indent=2)}

Write a clear statistical interpretation. Requirements:
1. For each significant finding, state in plain English WHAT the relationship is and HOW STRONG it is.
2. Reference the actual p-value and effect size/label.
3. Distinguish correlation from causation.
4. Flag any results where effect size is small despite statistical significance (large-n artifact).
5. Provide 2-4 concrete follow-up analysis recommendations.

Example style:
"There IS a statistically significant positive correlation between obesity_rate and diabetes_rate
(r=0.82, p<0.001, large effect). This is not random chance — counties with higher obesity
rates consistently show higher diabetes rates."

Return a JSON object:
{{
  "narrative": "full interpretation — use \\n to separate distinct findings",
  "recommendations": ["recommendation 1", "recommendation 2", ...]
}}

Return only valid JSON, no other text."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
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
            return {"narrative": raw, "recommendations": []}
