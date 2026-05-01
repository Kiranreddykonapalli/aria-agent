"""
Aria — Autonomous Reasoning & Insight Agent — Streamlit frontend.

Runs the full DataWrangler -> Analyst -> VizBuilder -> ReportWriter pipeline
with live per-stage progress. Results surface in three tabs:
  - Insights   : 5 key findings + data quality metrics + column descriptions
  - Charts     : 2-column grid of every generated PNG
  - Report     : full rendered markdown + download button
"""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Aria — AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dotenv import load_dotenv
load_dotenv()

from agents.data_wrangler import DataWrangler
from agents.analyst import Analyst
from agents.viz_builder import VizBuilder
from agents.report_writer import ReportWriter
from agents.email_agent import EmailAgent
from agents.anomaly_agent import AnomalyAgent
from agents.decision_agent import DecisionAgent
from agents.forecasting_agent import ForecastingAgent
from agents.data_prep_agent import DataPrepAgent
from agents.stats_agent import StatsAgent
from agents.sql_agent import SQLAgent
from agents.quality_agent import QualityAgent
from agents.pptx_agent import PPTXAgent

# ── Constants ─────────────────────────────────────────────────────────
DEMO_CSV      = "data/raw/florida_health_2024.csv"
DEMO_QUESTION = (
    "Which Florida counties have the worst health outcomes "
    "and what factors drive them?"
)
MODELS = {
    "Claude Sonnet 4.6  (best quality)": "claude-sonnet-4-6",
    "Claude Haiku 4.5   (fastest)":      "claude-haiku-4-5-20251001",
}
INSIGHT_ICONS   = ["🔑", "📊", "💡", "📌", "🎯"]
INSIGHT_COLORS  = ["#1a56db", "#7c3aed", "#059669", "#d97706", "#dc2626"]

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ───────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background: #ffffff;
}

/* ── Hero header ─────────────────────────────────────────── */
.hero-wrap {
    padding: 2.2rem 0 1.4rem 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.6rem;
}
.gradient-title {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #1a56db 0%, #3b82f6 55%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
}
.hero-tagline {
    font-size: 1.2rem;
    color: #374151;
    font-weight: 400;
    margin: 0 0 0.5rem 0;
}
.powered-by {
    font-size: 0.78rem;
    color: #9ca3af;
    letter-spacing: 0.03em;
    margin: 0;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #f8faff;
    border-right: 1px solid #dde3f0;
}
[data-testid="stSidebar"] h2 {
    color: #1a56db;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.sidebar-footer {
    border-top: 1px solid #dde3f0;
    padding-top: 1rem;
    margin-top: 1rem;
    font-size: 0.78rem;
    color: #6b7280;
    line-height: 1.7;
}
.sidebar-footer strong { color: #111827; }

/* ── Run button ──────────────────────────────────────────── */
button[kind="primary"] {
    background: linear-gradient(135deg, #1a56db 0%, #3b82f6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(26, 86, 219, 0.35) !important;
    transition: box-shadow 0.2s ease !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(26, 86, 219, 0.5) !important;
}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #f0f4ff;
    border: 1px solid #c7d7f8;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}
[data-testid="stMetricLabel"] { color: #6b7280; font-size: 0.8rem; }
[data-testid="stMetricValue"] { color: #111827; font-weight: 700; }

/* ── Insight cards ───────────────────────────────────────── */
.insight-card {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    transition: box-shadow 0.15s ease;
}
.insight-card:hover { box-shadow: 0 4px 14px rgba(0,0,0,0.1); }
.insight-icon {
    font-size: 1.4rem;
    line-height: 1;
    flex-shrink: 0;
    margin-top: 0.1rem;
}
.insight-body { flex: 1; }
.insight-num {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.insight-text {
    font-size: 0.93rem;
    line-height: 1.65;
    color: #374151;
    margin: 0;
}

/* ── Step cards (landing) ────────────────────────────────── */
.step-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-top: 4px solid var(--step-color, #1a56db);
    border-radius: 10px;
    padding: 1.2rem 1rem 1rem 1rem;
    text-align: center;
    height: 100%;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.step-icon { font-size: 1.9rem; margin-bottom: 0.4rem; }
.step-label {
    font-weight: 700;
    font-size: 0.9rem;
    color: #111827;
    margin-bottom: 0.3rem;
}
.step-desc { font-size: 0.8rem; color: #6b7280; line-height: 1.45; }
.arrow-col {
    display: flex;
    align-items: center;
    justify-content: center;
    padding-top: 1.8rem;
    font-size: 1.3rem;
    color: #c7d7f8;
}

/* ── Demo card (landing) ─────────────────────────────────── */
.demo-card {
    background: linear-gradient(135deg, #eff6ff 0%, #f0f4ff 100%);
    border: 1px solid #c7d7f8;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
}
.demo-card h3 { color: #1a56db; margin-top: 0; }

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid #e5e7eb; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    color: #6b7280;
    padding: 0.55rem 1.1rem;
}
.stTabs [aria-selected="true"] { color: #1a56db; }

/* ── Anomaly cards ───────────────────────────────────────── */
.anomaly-card {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.anomaly-badge {
    flex-shrink: 0;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.1rem;
}
.anomaly-body { flex: 1; min-width: 0; }
.anomaly-entity {
    font-weight: 700;
    font-size: 0.9rem;
    color: #111827;
    margin-bottom: 0.15rem;
}
.anomaly-meta {
    font-size: 0.78rem;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.anomaly-reason {
    font-size: 0.83rem;
    color: #374151;
    line-height: 1.55;
}
.severity-pill {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 0.4rem;
}

/* ── Decision cards ──────────────────────────────────────── */
.decision-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.decision-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
}
.priority-badge {
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    flex-shrink: 0;
}
.timeline-badge {
    padding: 0.2rem 0.65rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    background: #f3f4f6;
    color: #374151;
    flex-shrink: 0;
}
.decision-action {
    font-size: 0.95rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.5;
    margin-bottom: 0.55rem;
}
.decision-meta {
    font-size: 0.82rem;
    color: #374151;
    line-height: 1.6;
}
.decision-meta strong { color: #111827; }
.domain-pill {
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 999px;
    padding: 0.3rem 1rem;
    font-size: 0.8rem;
    font-weight: 600;
    color: #1a56db;
    margin-bottom: 1.2rem;
}

/* ── Forecast cards ──────────────────────────────────────── */
.forecast-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.2rem 0.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.forecast-metric {
    font-size: 1rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.5rem;
}
.forecast-row {
    display: flex;
    gap: 1.2rem;
    flex-wrap: wrap;
    margin-bottom: 0.4rem;
    align-items: center;
}
.r2-badge {
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.forecast-vals {
    font-size: 0.82rem;
    color: #374151;
    line-height: 1.7;
}
.forecast-vals strong { color: #111827; }
.ci-note {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 0.2rem;
}

/* ── Data prep ───────────────────────────────────────────── */
.prep-stat {
    display: inline-block;
    background: #f0f4ff;
    border: 1px solid #c7d7f8;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.82rem;
    font-weight: 600;
    color: #1a56db;
    margin-right: 0.5rem;
    margin-bottom: 0.4rem;
}
.prep-warning {
    background: #fef9c3;
    border-left: 3px solid #d97706;
    border-radius: 0 6px 6px 0;
    padding: 0.5rem 0.8rem;
    font-size: 0.82rem;
    color: #78350f;
    margin-bottom: 0.4rem;
}

/* ── Stats tab ───────────────────────────────────────────── */
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}
.stat-sig   { border-left: 4px solid #059669; }
.stat-insig { border-left: 4px solid #d1d5db; }
.stat-badge {
    flex-shrink: 0;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.15rem;
}
.stat-body { flex: 1; min-width: 0; }
.stat-cols {
    font-size: 0.88rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.2rem;
}
.stat-nums {
    font-size: 0.78rem;
    color: #6b7280;
    line-height: 1.6;
}
.stat-nums strong { color: #374151; }

/* ── SQL section ─────────────────────────────────────────── */
.sql-meta {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    align-items: center;
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 0.6rem;
}
.sql-meta strong { color: #111827; }
.sql-chip {
    background: #f0f4ff;
    border: 1px solid #c7d7f8;
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #1a56db;
}

/* ── Quality score card ──────────────────────────────────── */
.score-card {
    background: linear-gradient(135deg, #f0f4ff 0%, #ffffff 100%);
    border: 1px solid #c7d7f8;
    border-radius: 16px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
}
.score-number {
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1;
    color: #111827;
    flex-shrink: 0;
}
.score-grade {
    font-size: 2rem;
    font-weight: 900;
    width: 2.6rem;
    height: 2.6rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: #ffffff;
}
.score-body { flex: 1; min-width: 220px; }
.score-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.3rem;
}
.score-verdict {
    font-size: 0.9rem;
    color: #374151;
    line-height: 1.65;
}

/* ── Caption ─────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
for key, default in [("results", None), ("error", None), ("prep", None), ("sql", None), ("pptx_path", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Hero header ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <p class="gradient-title">📊 Aria</p>
    <p class="hero-tagline">Ask anything about your data. Get answers in seconds.</p>
    <p class="powered-by">Autonomous Reasoning &amp; Insight Agent &nbsp;·&nbsp; Powered by Claude · Anthropic</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Any tabular CSV — column names are auto-detected by Claude.",
    )
    question_input = st.text_area(
        "Your question",
        placeholder="e.g. Which regions have the highest risk factors?",
        height=110,
    )
    model_label = st.selectbox("Model", list(MODELS.keys()))
    model_id    = MODELS[model_label]

    st.divider()
    run_btn  = st.button("▶  Run Analysis",          type="primary", use_container_width=True)
    demo_btn = st.button("⚡  Demo: Florida Health", use_container_width=True)

    if st.session_state.results:
        st.success("Last run complete")

    # ── Share Results ─────────────────────────────────────────────
    if st.session_state.results:
        st.divider()
        st.subheader("📧 Share Results")
        recipient = st.text_input(
            "Recipient email",
            placeholder="colleague@example.com",
            key="email_recipient",
        )
        send_btn = st.button("Send Report via Email", use_container_width=True)

        if send_btn:
            if not recipient.strip():
                st.error("Enter a recipient email address.")
            else:
                r = st.session_state.results
                with st.spinner("Sending…"):
                    outcome = EmailAgent().run(
                        recipient_email=recipient.strip(),
                        report_text=r["report"]["report_text"],
                        figure_paths=r["viz"]["figure_paths"],
                        question=r["question"],
                    )
                if outcome["success"]:
                    st.success(f"Report sent to {outcome['recipient']}")
                else:
                    st.error(f"Failed: {outcome['message']}")

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-footer">
        Built by <strong>Kiran Reddy Konapalli</strong>
    </div>
    """, unsafe_allow_html=True)


# ── Pipeline runner ───────────────────────────────────────────────────
def run_pipeline(data_path: str, question: str, model: str) -> None:
    st.session_state.results = None
    st.session_state.error   = None

    wrangler_output = analyst_output = viz_output = report_output = None

    with st.status("Running Aria pipeline…", expanded=True) as status:
        try:
            st.write("🏅 **Quality Agent** — scoring raw data quality…")
            import pandas as _pd
            _raw_df = _pd.read_csv(data_path)
            quality_output = QualityAgent(model=model).run(_raw_df)
            st.write(
                f"✓ Aria Score {quality_output['overall_score']:.0f}/100 · "
                f"Grade {quality_output['grade']}"
            )

            st.write("🔍 **Data Wrangler** — loading and cleaning CSV…")
            wrangler_output = DataWrangler().run(data_path)
            qr = wrangler_output["data_quality_report"]
            st.write(
                f"✓ {qr['final_row_count']:,} rows · "
                f"{qr['final_column_count']} columns · "
                f"{qr['duplicate_rows_dropped']} duplicates removed"
            )

            st.write("🧠 **Analyst** — profiling data and extracting insights…")
            analyst_output = Analyst(model=model).run(wrangler_output, question=question)
            st.write(
                f"✓ {len(analyst_output['insights'])} insights · "
                f"{len(analyst_output['suggested_charts'])} charts planned"
            )

            st.write("🔎 **Anomaly Agent** — detecting statistical anomalies…")
            anomaly_output = AnomalyAgent(model=model).run(
                wrangler_output["dataframe"], analyst_output
            )
            sc = anomaly_output["severity_counts"]
            st.write(
                f"✓ {len(anomaly_output['anomalies'])} anomalies — "
                f"🔴 {sc['high']} high · 🟠 {sc['medium']} medium · 🟡 {sc['low']} low"
            )

            st.write("🎯 **Decision Agent** — generating actionable decisions…")
            decision_output = DecisionAgent(model=model).run(
                question       = question,
                analyst_output = analyst_output,
                anomaly_output = anomaly_output,
                dataframe      = wrangler_output["dataframe"],
            )
            st.write(
                f"✓ {len(decision_output['decisions'])} decisions · "
                f"domain: {decision_output['domain'][:70]}"
            )

            st.write("📈 **Forecasting Agent** — fitting trends and projecting forward…")
            forecast_output = ForecastingAgent(model=model).run(
                wrangler_output["dataframe"], analyst_output
            )
            st.write(
                f"✓ {len(forecast_output['forecasts'])} forecasts · "
                f"{len(forecast_output['figure_paths'])} charts"
            )

            st.write("🔬 **Stats Agent** — running hypothesis tests…")
            stats_output = StatsAgent(model=model).run(
                wrangler_output["dataframe"], analyst_output, question
            )
            st.write(
                f"✓ {len(stats_output['tests_run'])} tests · "
                f"{len(stats_output['significant_findings'])} significant"
            )

            st.write("📊 **Viz Builder** — rendering analysis charts…")
            viz_output = VizBuilder().run(wrangler_output, analyst_output)
            st.write(
                f"✓ {viz_output['charts_rendered']} charts saved · "
                f"{viz_output['charts_skipped']} skipped"
            )

            st.write("📝 **Report Writer** — composing analysis report…")
            report_output = ReportWriter(model=model).run(question, analyst_output, viz_output)
            st.write(f"✓ Saved to `{report_output['report_path']}`")

            status.update(label="✅ Pipeline complete", state="complete", expanded=False)

        except Exception:
            status.update(label="❌ Pipeline failed", state="error")
            tb = traceback.format_exc()
            st.session_state.error = tb
            st.code(tb, language="")
            return

    st.session_state.results = {
        "quality":   quality_output,
        "wrangler":  wrangler_output,
        "analyst":   analyst_output,
        "anomaly":   anomaly_output,
        "decision":  decision_output,
        "forecast":  forecast_output,
        "stats":     stats_output,
        "viz":       viz_output,
        "report":    report_output,
        "question":  question,
    }


# ── Button handlers ───────────────────────────────────────────────────
if demo_btn:
    if not os.path.exists(DEMO_CSV):
        st.error(
            f"Demo file not found at `{DEMO_CSV}`. "
            "Run `python generate_synthetic_data.py` first."
        )
    else:
        run_pipeline(DEMO_CSV, DEMO_QUESTION, model_id)

if run_btn:
    q = question_input.strip()
    if not uploaded_file and not st.session_state.prep:
        st.sidebar.error("Upload a CSV file — or use the Demo button.")
    elif not q:
        st.sidebar.error("Enter a question before running.")
    else:
        # Prefer the prep-cleaned CSV if one exists; otherwise use the upload
        if st.session_state.prep and st.session_state.prep.get("output_path"):
            run_pipeline(st.session_state.prep["output_path"], q, model_id)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as f:
                f.write(uploaded_file.getbuffer())
                tmp = f.name
            try:
                run_pipeline(tmp, q, model_id)
            finally:
                os.unlink(tmp)

# ── Data Prep expander ────────────────────────────────────────────────
if uploaded_file or st.session_state.prep:
    with st.expander("🧹 Data Prep — Optional AI-assisted cleaning", expanded=False):
        st.caption(
            "Describe what to clean in plain English. "
            "Claude will translate your instruction into pandas operations and apply them. "
            "If you run this, the cleaned data will be used in the main analysis."
        )

        prep_instruction = st.text_area(
            "Cleaning instruction",
            placeholder=(
                "e.g. Drop rows where age is negative. "
                "Fill missing income values with the median. "
                "Remove the passenger_id column."
            ),
            height=90,
            key="prep_instruction_input",
        )
        col_run, col_clear = st.columns([2, 1])
        prep_btn  = col_run.button("🧹 Clean My Data", type="primary", use_container_width=True)
        clear_btn = col_clear.button("✕ Clear", use_container_width=True)

        if clear_btn:
            st.session_state.prep = None
            st.rerun()

        if prep_btn and prep_instruction.strip():
            if not uploaded_file:
                st.error("Upload a CSV file first.")
            else:
                import pandas as _pd
                raw_df = _pd.read_csv(uploaded_file)
                with st.spinner("Claude is planning and applying operations…"):
                    try:
                        prep_result = DataPrepAgent(model=model_id).run(
                            raw_df, prep_instruction.strip()
                        )
                        st.session_state.prep = prep_result
                    except Exception as exc:
                        st.error(f"Data prep failed: {exc}")

        if st.session_state.prep:
            p = st.session_state.prep
            before_r, before_c = p["before_shape"]
            after_r,  after_c  = p["after_shape"]
            rows_removed = before_r - after_r
            cols_removed = before_c - after_c

            st.success("Data prep complete — cleaned CSV will be used in the main analysis.")

            st.markdown(
                f'<span class="prep-stat">Before: {before_r:,} rows × {before_c} cols</span>'
                f'<span class="prep-stat">After: {after_r:,} rows × {after_c} cols</span>'
                f'<span class="prep-stat">Rows removed: {rows_removed:,}</span>'
                f'<span class="prep-stat">Cols removed: {cols_removed}</span>',
                unsafe_allow_html=True,
            )

            if p.get("explanation"):
                st.markdown(f"**Plan:** {p['explanation']}")

            for w in p.get("warnings", []):
                st.markdown(
                    f'<div class="prep-warning">⚠ {w}</div>',
                    unsafe_allow_html=True,
                )

            if p.get("operations_skipped"):
                with st.expander(f"⚠ {len(p['operations_skipped'])} operation(s) skipped"):
                    for op in p["operations_skipped"]:
                        st.caption(f"{op.get('type')}: {op.get('error')}")

            with open(p["output_path"], "rb") as fh:
                st.download_button(
                    "⬇ Download cleaned CSV",
                    data=fh.read(),
                    file_name="aria_cleaned.csv",
                    mime="text/csv",
                )

# ── Ask SQL ───────────────────────────────────────────────────────────
def _get_active_df() -> pd.DataFrame | None:
    """Return the best available DataFrame for SQL queries."""
    import pandas as _pd
    if st.session_state.prep and st.session_state.prep.get("dataframe") is not None:
        return st.session_state.prep["dataframe"]
    if st.session_state.results and st.session_state.results.get("wrangler"):
        return st.session_state.results["wrangler"]["dataframe"]
    if uploaded_file is not None:
        uploaded_file.seek(0)
        return _pd.read_csv(uploaded_file)
    if os.path.exists(DEMO_CSV):
        return _pd.read_csv(DEMO_CSV)
    return None

if uploaded_file or st.session_state.results or st.session_state.prep:
    with st.expander("🔍 Ask SQL — Query your data with plain English", expanded=False):
        st.caption(
            "Ask any question about your data in plain English. "
            "Aria will write and run the SQL for you against an in-memory SQLite database."
        )

        sql_question = st.text_input(
            "Your question",
            placeholder='e.g. "Which 5 counties have the highest diabetes rate in 2024?"',
            key="sql_question_input",
        )
        sql_col1, sql_col2 = st.columns([3, 1])
        run_sql_btn   = sql_col1.button("▶  Run Query", type="primary", use_container_width=True)
        clear_sql_btn = sql_col2.button("✕ Clear", use_container_width=True, key="clear_sql")

        if clear_sql_btn:
            st.session_state.sql = None
            st.rerun()

        if run_sql_btn:
            if not sql_question.strip():
                st.error("Enter a question first.")
            else:
                df_for_sql = _get_active_df()
                if df_for_sql is None:
                    st.error("No data available — upload a CSV or run the demo first.")
                else:
                    import time as _time
                    with st.spinner("Writing and running SQL…"):
                        t0 = _time.perf_counter()
                        try:
                            sql_result = SQLAgent(model=model_id).run(
                                df_for_sql, sql_question.strip()
                            )
                            sql_result["execution_time_ms"] = (_time.perf_counter() - t0) * 1000
                            st.session_state.sql = sql_result
                        except Exception as exc:
                            st.error(f"SQL Agent error: {exc}")

        if st.session_state.sql:
            s = st.session_state.sql

            # Metadata row
            rows    = s.get("row_count", 0)
            elapsed = s.get("execution_time_ms", 0)
            chart   = s.get("chart_suggestion")
            st.markdown(
                f'<div class="sql-meta">'
                f'<span class="sql-chip">{rows} row{"s" if rows != 1 else ""} returned</span>'
                f'<span class="sql-chip">{elapsed:.1f} ms</span>'
                + (f'<span class="sql-chip">💡 try a {chart} chart</span>' if chart else "")
                + f"</div>",
                unsafe_allow_html=True,
            )

            # Explanation
            if s.get("explanation"):
                st.markdown(f"**What the query does:** {s['explanation']}")

            # SQL code block
            st.code(s.get("sql_query", ""), language="sql")

            # Error banner
            if s.get("error"):
                st.error(s["error"])
            else:
                # Results dataframe
                result_df = s.get("results_dataframe")
                if result_df is not None and not result_df.empty:
                    st.dataframe(result_df, use_container_width=True)
                else:
                    st.info("Query returned no rows.")

# ── Error banner ──────────────────────────────────────────────────────
if st.session_state.error and not st.session_state.results:
    st.error("Pipeline failed — see traceback above in the progress panel.")

# ── Landing page ──────────────────────────────────────────────────────
if not st.session_state.results:
    STEPS = [
        ("#1a56db", "📤", "Upload",    "Any CSV. No data prep needed."),
        ("#7c3aed", "🔍", "Wrangle",   "Auto-clean, validate & profile."),
        ("#059669", "🧠", "Analyse",   "5 AI-powered key insights."),
        ("#d97706", "📈", "Visualise", "Charts tailored to findings."),
        ("#dc2626", "📝", "Report",    "Full markdown report, instant."),
    ]

    st.markdown("#### How it works")

    # 5 step cards with arrows between them using 9 columns
    cols = st.columns([4, 1, 4, 1, 4, 1, 4, 1, 4])
    step_cols  = [cols[i] for i in range(0, 9, 2)]
    arrow_cols = [cols[i] for i in range(1, 8, 2)]

    for col, (color, icon, label, desc) in zip(step_cols, STEPS):
        with col:
            st.markdown(f"""
            <div class="step-card" style="--step-color:{color}">
                <div class="step-icon">{icon}</div>
                <div class="step-label">{label}</div>
                <div class="step-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    for col in arrow_cols:
        with col:
            st.markdown('<div class="arrow-col">›</div>', unsafe_allow_html=True)

    st.divider()

    _, demo_col, _ = st.columns([1, 2, 1])
    with demo_col:
        st.markdown("""
        <div class="demo-card">
            <h3>⚡ Try the demo</h3>
            <p>Click <strong>Demo: Florida Health</strong> in the sidebar to run a full analysis on
            a 402-row Florida county health dataset (2019–2024) — covering uninsured rates,
            obesity, diabetes, physician access, income, and more.</p>
            <p style="margin:0;color:#6b7280;font-size:0.85rem;">
                No file upload needed · Results in ~30 seconds
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ── Results tabs ──────────────────────────────────────────────────────
r        = st.session_state.results
quality  = r.get("quality", {})
wrangler = r["wrangler"]
analyst  = r["analyst"]
anomaly  = r["anomaly"]
decision = r["decision"]
forecast = r["forecast"]
stats    = r["stats"]
viz      = r["viz"]
report   = r["report"]
question = r["question"]

st.markdown(f"#### Results — *{question}*")

# ── Quality score card ────────────────────────────────────────────────
if quality:
    GRADE_COLOR = {"A": "#059669", "B": "#1a56db", "C": "#d97706", "D": "#ea580c", "F": "#dc2626"}
    q_score  = quality.get("overall_score", 0)
    q_grade  = quality.get("grade", "?")
    q_dims   = quality.get("dimension_scores", {})
    q_verdict = quality.get("verdict", "")
    q_recs   = quality.get("recommendations", [])
    g_color  = GRADE_COLOR.get(q_grade, "#6b7280")
    score_pct = int(q_score)

    st.markdown(f"""
    <div class="score-card">
      <div class="score-number">{score_pct}<span style="font-size:1.4rem;color:#9ca3af;">/100</span></div>
      <div class="score-grade" style="background:{g_color};">{q_grade}</div>
      <div class="score-body">
        <div class="score-label">Aria Data Quality Score</div>
        <div class="score-verdict">{q_verdict}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dimension bars
    DIM_MAX = {"completeness": 20, "uniqueness": 20, "consistency": 20,
               "validity": 20, "timeliness": 10, "uniformity": 10}
    dim_cols = st.columns(3)
    for i, (dim, max_pts) in enumerate(DIM_MAX.items()):
        score_val = q_dims.get(dim, 0)
        pct       = score_val / max_pts
        label     = f"{dim.title()} — {score_val:.0f}/{max_pts}"
        with dim_cols[i % 3]:
            st.caption(label)
            st.progress(pct)

    if q_recs:
        with st.expander("💡 Recommendations to improve your score"):
            for i, rec in enumerate(q_recs, 1):
                st.markdown(f"**{i}.** {rec}")

    st.divider()

tab_insights, tab_charts, tab_anomalies, tab_decisions, tab_forecasts, tab_stats, tab_report = st.tabs(
    ["💡  Insights", "📊  Charts", "🔎  Anomalies", "🎯  Decisions", "📈  Forecasts", "🔬  Statistics", "📄  Report"]
)

# ── Tab 1: Insights ───────────────────────────────────────────────────
with tab_insights:
    qr    = wrangler["data_quality_report"]
    nulls = qr.get("nulls_dropped", 0)
    dups  = qr.get("duplicate_rows_dropped", 0)
    flags = [v for v in qr.get("suspicious_values", []) if v != "none detected"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows analysed",      f"{qr['final_row_count']:,}")
    col2.metric("Columns",             qr["final_column_count"])
    col3.metric("Rows removed",        nulls + dups)
    col4.metric("Data quality flags",  len(flags) if flags else "None")

    if flags:
        with st.expander("⚠ Data quality flags"):
            for flag in flags:
                st.warning(flag)

    st.divider()
    st.subheader("Key Findings")

    for i, insight in enumerate(analyst["insights"]):
        icon  = INSIGHT_ICONS[i % len(INSIGHT_ICONS)]
        color = INSIGHT_COLORS[i % len(INSIGHT_COLORS)]
        st.markdown(f"""
        <div class="insight-card" style="border-left: 4px solid {color}">
            <div class="insight-icon">{icon}</div>
            <div class="insight-body">
                <div class="insight-num" style="color:{color}">Finding {i+1}</div>
                <p class="insight-text">{insight}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    with st.expander("Column descriptions (auto-detected by Claude)"):
        header = "| Column | Role | Description |\n|---|---|---|\n"
        rows   = []
        for col, info in analyst["column_descriptions"].items():
            unit = f" ({info['unit']})" if info.get("unit") else ""
            rows.append(
                f"| `{col}` | {info.get('role','?')} "
                f"| {info.get('description','')}{unit} |"
            )
        st.markdown(header + "\n".join(rows))

# ── Tab 2: Charts ─────────────────────────────────────────────────────
with tab_charts:
    paths = viz.get("figure_paths", [])
    metas = analyst.get("suggested_charts", [])

    if not paths:
        st.info("No charts were generated.")
    else:
        st.caption(
            f"{viz['charts_rendered']} chart(s) rendered · "
            f"{viz['charts_skipped']} skipped"
        )
        for i in range(0, len(paths), 2):
            cols = st.columns(2, gap="medium")
            for j in range(2):
                idx = i + j
                if idx < len(paths) and os.path.exists(paths[idx]):
                    with cols[j]:
                        title = (
                            metas[idx].get("title", "")
                            if idx < len(metas) else ""
                        )
                        if title:
                            st.caption(title)
                        st.image(paths[idx], use_container_width=True)

# ── Tab 3: Anomalies ─────────────────────────────────────────────────
with tab_anomalies:
    sc        = anomaly["severity_counts"]
    all_anom  = anomaly["anomalies"]
    narrative = anomaly["narrative"]

    # Severity summary badges
    st.markdown(f"""
    <div style="display:flex;gap:0.75rem;margin-bottom:1.2rem;flex-wrap:wrap;">
      <span class="severity-pill" style="background:#fee2e2;color:#991b1b;">
        🔴 {sc['high']} High
      </span>
      <span class="severity-pill" style="background:#ffedd5;color:#9a3412;">
        🟠 {sc['medium']} Medium
      </span>
      <span class="severity-pill" style="background:#fef9c3;color:#854d0e;">
        🟡 {sc['low']} Low
      </span>
      <span class="severity-pill" style="background:#f3f4f6;color:#374151;">
        Total: {len(all_anom)}
      </span>
    </div>
    """, unsafe_allow_html=True)

    if not all_anom:
        st.info("No anomalies detected.")
    else:
        SEV_STYLE = {
            "high":   ("🔴", "#fee2e2", "#991b1b"),
            "medium": ("🟠", "#ffedd5", "#9a3412"),
            "low":    ("🟡", "#fef9c3", "#854d0e"),
        }

        # Severity filter
        filter_sev = st.radio(
            "Show", ["All", "High only", "High + Medium"],
            horizontal=True, label_visibility="collapsed",
        )
        shown = all_anom
        if filter_sev == "High only":
            shown = [a for a in all_anom if a["severity"] == "high"]
        elif filter_sev == "High + Medium":
            shown = [a for a in all_anom if a["severity"] in ("high", "medium")]

        st.caption(f"Showing {len(shown)} of {len(all_anom)} anomalies")

        for a in shown:
            icon, bg, fg = SEV_STYLE.get(a["severity"], ("⚪", "#f3f4f6", "#374151"))
            methods_str  = " · ".join(a.get("methods", [a.get("method", "")]))
            time_str     = f" · {a['time']}" if a.get("time") is not None else ""
            st.markdown(f"""
            <div class="anomaly-card" style="border-left:4px solid {fg}">
              <span class="anomaly-badge" style="background:{bg};color:{fg};">
                {icon} {a['severity']}
              </span>
              <div class="anomaly-body">
                <div class="anomaly-entity">{a['entity']}</div>
                <div class="anomaly-meta">
                  <strong>{a['column']}</strong> &nbsp;·&nbsp;
                  value: <code>{a['value']}</code>
                  {f'&nbsp;·&nbsp; z={a["z_score"]}' if a.get("z_score") else ""}
                  {time_str} &nbsp;·&nbsp; detected by: {methods_str}
                </div>
                <div class="anomaly-reason">{a['reason']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Claude's Interpretation")
    for line in narrative.split("\n"):
        if line.strip():
            st.markdown(line)

# ── Tab 4: Decisions ─────────────────────────────────────────────────
with tab_decisions:
    PRIORITY_STYLE = {
        "Critical": ("#fef2f2", "#991b1b", "🔴"),
        "High":     ("#fff7ed", "#9a3412", "🟠"),
        "Medium":   ("#eff6ff", "#1e40af", "🔵"),
    }

    if decision.get("domain"):
        st.markdown(
            f'<div class="domain-pill">🏷 {decision["domain"]}</div>',
            unsafe_allow_html=True,
        )

    decisions = decision.get("decisions", [])
    if not decisions:
        st.info("No decisions generated.")
    else:
        for i, d in enumerate(decisions, 1):
            priority  = d.get("priority", "Medium")
            bg, fg, icon = PRIORITY_STYLE.get(priority, ("#f3f4f6", "#374151", "⚪"))
            timeline  = d.get("timeline", "")

            st.markdown(f"""
            <div class="decision-card" style="border-left: 4px solid {fg};">
              <div class="decision-header">
                <span class="priority-badge" style="background:{bg};color:{fg};">
                  {icon} {priority}
                </span>
                <span class="timeline-badge">⏱ {timeline}</span>
                <span style="font-size:0.75rem;color:#9ca3af;">Decision {i} of {len(decisions)}</span>
              </div>
              <div class="decision-action">{d.get('action', '')}</div>
              <div class="decision-meta">
                <strong>Rationale:</strong> {d.get('rationale', '')}<br>
                <strong>Expected impact:</strong> {d.get('expected_impact', '')}
              </div>
            </div>
            """, unsafe_allow_html=True)

    if decision.get("summary"):
        st.divider()
        st.subheader("Decision Summary")
        st.markdown(decision["summary"])

# ── Tab 5: Forecasts ─────────────────────────────────────────────────
with tab_forecasts:
    fc_list    = forecast.get("forecasts", [])
    fc_narr    = forecast.get("narrative", "")
    fc_figures = forecast.get("figure_paths", [])

    # Build a path → forecast lookup so we can show chart under its card
    # forecast_figures are named  …_forecast_<col>.png  — match by position
    fig_by_metric: dict[str, str] = {}
    for fc in fc_list:
        metric = fc["metric"]
        for p in fc_figures:
            if metric.lower().replace(" ", "_") in Path(p).stem.lower():
                fig_by_metric[metric] = p
                break

    # Narrative at the top
    if fc_narr:
        st.subheader("Trend Interpretation")
        for line in fc_narr.split("\n"):
            if line.strip():
                st.markdown(line)
        st.divider()

    if not fc_list:
        st.info(
            "No statistically meaningful trends found. "
            "Forecasting requires a time column and at least 4 data points per metric."
        )
    else:
        st.caption(
            f"{len(fc_list)} forecast(s) · sorted by model fit (R²) · "
            "dashed lines show 95% prediction interval"
        )

        for fc in fc_list:
            metric    = fc["metric"]
            r2        = fc.get("r_squared", 0)
            last_val  = fc.get("last_value")
            last_per  = fc.get("last_period")
            slope     = fc.get("slope", 0)

            # R² colour
            if r2 >= 0.70:
                r2_bg, r2_fg, r2_label = "#dcfce7", "#166534", "Strong fit"
            elif r2 >= 0.40:
                r2_bg, r2_fg, r2_label = "#fef9c3", "#854d0e", "Moderate fit"
            else:
                r2_bg, r2_fg, r2_label = "#fee2e2", "#991b1b", "Weak fit"

            # Collect forecast years
            future_keys = sorted(
                [k for k in fc if k.startswith("forecast_")],
                key=lambda k: int(k.split("_")[1]),
            )
            future_years = [int(k.split("_")[1]) for k in future_keys]

            # Projected values and CI for each year
            proj_lines = []
            for yr in future_years:
                val = fc.get(f"forecast_{yr}")
                ci  = fc.get(f"ci_{yr}")
                if val is None:
                    continue
                ci_str = (
                    f"[{ci[0]:.4g} – {ci[1]:.4g}]"
                    if ci else ""
                )
                pct_chg = ((val - last_val) / abs(last_val) * 100) if last_val else 0
                arrow   = "▲" if val > last_val else "▼"
                proj_lines.append(
                    f"<strong>{yr}:</strong> {val:.4g} "
                    f"<span style='color:{'#166534' if val>last_val else '#991b1b'}'>"
                    f"{arrow} {abs(pct_chg):.1f}%</span> &nbsp; "
                    f"<span style='color:#9ca3af;font-size:0.75rem;'>{ci_str}</span>"
                )

            proj_html = "<br>".join(proj_lines)
            trend_dir = "increasing" if slope > 0 else "decreasing"

            st.markdown(f"""
            <div class="forecast-card">
              <div class="forecast-metric">{metric.replace('_', ' ').title()}</div>
              <div class="forecast-row">
                <span class="r2-badge" style="background:{r2_bg};color:{r2_fg};">
                  R² = {r2:.2f} &nbsp;·&nbsp; {r2_label}
                </span>
                <span style="font-size:0.78rem;color:#6b7280;">
                  Trend: {trend_dir} · Last observed: {last_val:.4g} ({last_per})
                </span>
              </div>
              <div class="forecast-vals">{proj_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # Show chart if available
            chart_path = fig_by_metric.get(metric)
            if not chart_path:
                # Fallback: match by index order
                idx = fc_list.index(fc)
                chart_path = fc_figures[idx] if idx < len(fc_figures) else None

            if chart_path and os.path.exists(chart_path):
                st.image(chart_path, use_container_width=True)

        # Any leftover charts not matched to a card
        unmatched = [p for p in fc_figures if p not in fig_by_metric.values()]
        for p in unmatched:
            if os.path.exists(p):
                st.image(p, use_container_width=True)

# ── Tab 6: Statistics ────────────────────────────────────────────────
with tab_stats:
    all_tests  = stats.get("tests_run", [])
    sig_tests  = stats.get("significant_findings", [])
    narr       = stats.get("narrative", "")
    recs       = stats.get("recommendations", [])

    # Summary bar
    n_total = len(all_tests)
    n_sig   = len(sig_tests)
    n_insig = n_total - n_sig
    types_run = list({t["test_name"] for t in all_tests})

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Tests run",      n_total)
    col_b.metric("Significant",    n_sig)
    col_c.metric("Not significant", n_insig)

    if types_run:
        st.caption("Tests performed: " + " · ".join(types_run))

    # Narrative
    if narr:
        st.divider()
        st.subheader("Statistical Interpretation")
        for line in narr.split("\n"):
            if line.strip():
                st.markdown(line)

    # Significant findings
    if sig_tests:
        st.divider()
        st.subheader(f"✅ Significant Findings  ({n_sig})")

        EFFECT_COLOR = {"large": "#166534", "medium": "#854d0e", "small": "#1e40af"}

        for t in sig_tests:
            cols_str  = " × ".join(t.get("columns_tested", []))
            p_val     = t.get("p_value", 0)
            effect    = t.get("effect_size", 0)
            e_label   = t.get("effect_label", "")
            e_color   = EFFECT_COLOR.get(e_label, "#374151")
            direction = t.get("direction", "")
            dir_str   = f" · {direction}" if direction else ""
            stat_val  = t.get("statistic", "")

            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"

            st.markdown(f"""
            <div class="stat-card stat-sig">
              <div>
                <span class="stat-badge" style="background:#dcfce7;color:#166534;">
                  ✓ p={p_str}
                </span>
              </div>
              <div class="stat-body">
                <div class="stat-cols">{cols_str}</div>
                <div class="stat-nums">
                  <strong>{t.get('test_name','')}</strong>
                  &nbsp;·&nbsp; statistic = {stat_val}
                  &nbsp;·&nbsp;
                  <span style="color:{e_color};font-weight:700;">
                    {e_label} effect (r={effect}){dir_str}
                  </span>
                  &nbsp;·&nbsp; n = {t.get('n','')}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Non-significant findings collapsed
    insig_tests = [t for t in all_tests if not t["significant"]]
    if insig_tests:
        with st.expander(f"Non-significant results ({n_insig})", expanded=False):
            for t in insig_tests:
                cols_str = " × ".join(t.get("columns_tested", []))
                p_val    = t.get("p_value", 0)
                p_str    = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
                st.markdown(f"""
                <div class="stat-card stat-insig">
                  <div>
                    <span class="stat-badge" style="background:#f3f4f6;color:#6b7280;">
                      p={p_str}
                    </span>
                  </div>
                  <div class="stat-body">
                    <div class="stat-cols" style="color:#6b7280;">{cols_str}</div>
                    <div class="stat-nums">{t.get('test_name','')} · effect={t.get('effect_size','')} ({t.get('effect_label','')})</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Recommendations
    if recs:
        st.divider()
        st.subheader("Recommendations")
        for i, rec in enumerate(recs, 1):
            st.markdown(f"**{i}.** {rec}")

# ── Tab 7: Report ─────────────────────────────────────────────────────
with tab_report:
    st.markdown(report["report_text"])
    st.divider()

    dl_md, dl_pptx, path_col = st.columns([1, 1, 2])

    with dl_md:
        st.download_button(
            label="⬇  Download (.md)",
            data=report["report_text"],
            file_name=Path(report["report_path"]).name,
            mime="text/markdown",
            use_container_width=True,
        )

    with dl_pptx:
        gen_btn = st.button("📊  Generate PowerPoint", use_container_width=True)
        if gen_btn:
            with st.spinner("Building presentation…"):
                try:
                    pptx_result = PPTXAgent().run(
                        question        = question,
                        analyst_output  = analyst,
                        viz_output      = viz,
                        decision_output = decision,
                        forecast_output = forecast,
                        quality_output  = quality,
                        anomaly_output  = anomaly,
                    )
                    st.session_state.pptx_path   = pptx_result["pptx_path"]
                    st.session_state.pptx_slides = pptx_result["slide_count"]
                except Exception as exc:
                    st.error(f"PowerPoint generation failed: {exc}")

        pptx_path = st.session_state.get("pptx_path")
        if pptx_path and os.path.exists(pptx_path):
            n = st.session_state.get("pptx_slides", "?")
            with open(pptx_path, "rb") as fh:
                st.download_button(
                    label=f"⬇  Download PPTX  ({n} slides)",
                    data=fh.read(),
                    file_name=Path(pptx_path).name,
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                    use_container_width=True,
                )

    with path_col:
        st.caption(f"Saved to `{report['report_path']}`")
