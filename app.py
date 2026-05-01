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

/* ── Caption ─────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
for key, default in [("results", None), ("error", None)]:
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

            st.write("📈 **Viz Builder** — rendering charts…")
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
        "wrangler": wrangler_output,
        "analyst":  analyst_output,
        "anomaly":  anomaly_output,
        "viz":      viz_output,
        "report":   report_output,
        "question": question,
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
    if not uploaded_file:
        st.sidebar.error("Upload a CSV file — or use the Demo button.")
    elif not q:
        st.sidebar.error("Enter a question before running.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as f:
            f.write(uploaded_file.getbuffer())
            tmp = f.name
        try:
            run_pipeline(tmp, q, model_id)
        finally:
            os.unlink(tmp)

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
wrangler = r["wrangler"]
analyst  = r["analyst"]
anomaly  = r["anomaly"]
viz      = r["viz"]
report   = r["report"]
question = r["question"]

st.markdown(f"#### Results — *{question}*")
tab_insights, tab_charts, tab_anomalies, tab_report = st.tabs(
    ["💡  Insights", "📊  Charts", "🔎  Anomalies", "📄  Report"]
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

# ── Tab 4: Report ─────────────────────────────────────────────────────
with tab_report:
    st.markdown(report["report_text"])
    st.divider()
    dl_col, path_col = st.columns([1, 3])
    with dl_col:
        st.download_button(
            label="⬇  Download (.md)",
            data=report["report_text"],
            file_name=Path(report["report_path"]).name,
            mime="text/markdown",
            use_container_width=True,
        )
    with path_col:
        st.caption(f"Saved to `{report['report_path']}`")
