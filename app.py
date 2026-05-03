"""
Aria v2.0 — Autonomous Reasoning & Insight Agent — Streamlit frontend.
Premium redesign: Inter font · #667eea–#764ba2 gradient · Plotly charts · 18 agents.
"""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Aria — AI Data Analyst",
    page_icon="✦",
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
from agents.whatif_agent import WhatIfAgent
from agents.debate_agent import DebateAgent
from agents.blindspot_agent import BlindSpotAgent
import plotly.express as px
import plotly.graph_objects as go

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
st.html("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
/* ═══════════════════════════════════════════════════════════
   ARIA v2.0 — Premium Design System
   Font: Inter | Primary: #667eea → #764ba2
   ═══════════════════════════════════════════════════════════ */

/* ── Global ───────────────────────────────────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #f0f2ff !important;
}
.main .block-container { padding: 1.2rem 2rem 5rem; max-width: 1350px; }

/* ── Hero header ─────────────────────────────────────────── */
.hero-outer { text-align:center; padding:2.5rem 0 1.8rem; margin-bottom:0.5rem; }
.hero-logo {
    font-size:3.8rem; font-weight:900; letter-spacing:-3px; line-height:1;
    background:linear-gradient(135deg,#667eea 0%,#764ba2 50%,#f093fb 100%);
    background-size:200% 200%;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:gradPulse 5s ease infinite;
    margin-bottom:0.6rem;
}
@keyframes gradPulse {
    0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}
}
.hero-sub-wrap { position:relative; height:2.1rem; margin-bottom:0.8rem; }
.cycle-text {
    position:absolute; left:50%; transform:translateX(-50%);
    font-size:1.25rem; font-weight:500; color:#475569;
    opacity:0; white-space:nowrap;
}
.ct1{animation:cycleT 16s infinite 0s}
.ct2{animation:cycleT 16s infinite 4s}
.ct3{animation:cycleT 16s infinite 8s}
.ct4{animation:cycleT 16s infinite 12s}
@keyframes cycleT {
    0%,100%{opacity:0;transform:translateX(-50%) translateY(8px)}
    8%,22%{opacity:1;transform:translateX(-50%) translateY(0)}
    30%{opacity:0;transform:translateX(-50%) translateY(-8px)}
}
.hero-tagline { font-size:1rem; color:#64748b; margin:0 0 0.9rem; }
.hero-badge {
    display:inline-block;
    background:linear-gradient(135deg,rgba(102,126,234,.1),rgba(118,75,162,.1));
    border:1px solid rgba(102,126,234,.25); border-radius:999px;
    padding:0.3rem 1.1rem; font-size:0.75rem; font-weight:600;
    color:#667eea; letter-spacing:.03em;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0f0c29 0%,#302b63 55%,#24243e 100%) !important;
    border-right:1px solid rgba(255,255,255,.04) !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.stSelectbox span) {
    color:rgba(255,255,255,.82) !important;
}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:#fff !important; }
[data-testid="stFileUploaderDropzone"] {
    background-color:rgba(255,255,255,0.05) !important;
    border:1.5px dashed rgba(255,255,255,0.3) !important;
    border-radius:12px !important;
}
/* Upload button */
section[data-testid="stFileUploaderDropzone"] button {
    background:linear-gradient(135deg,#667eea,#764ba2) !important;
    color:white !important;
    border:none !important;
    font-weight:600 !important;
    border-radius:8px !important;
}
section[data-testid="stFileUploaderDropzone"] button p {
    color:white !important;
    font-weight:600 !important;
}
section[data-testid="stFileUploaderDropzone"] button span {
    color:white !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small {
    color:rgba(255,255,255,0.6) !important;
}
[data-testid="stSidebar"] textarea {
    background:rgba(255,255,255,.08) !important;
    border:1px solid rgba(255,255,255,.15) !important;
    color:#fff !important; border-radius:10px !important;
}
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.1) !important; }
[data-testid="stSidebar"] .stSuccess { background:rgba(16,185,129,.2) !important; }
.sidebar-footer {
    border-top:1px solid rgba(255,255,255,.1); padding-top:1rem; margin-top:1rem;
    font-size:0.77rem; color:rgba(255,255,255,.45); line-height:1.7; text-align:center;
}
.sidebar-footer strong { color:rgba(255,255,255,.75); }

/* ── Buttons ─────────────────────────────────────────────── */
button[kind="primary"] {
    background:linear-gradient(135deg,#667eea 0%,#764ba2 100%) !important;
    border:none !important; color:#fff !important; font-weight:700 !important;
    font-size:0.9rem !important; border-radius:12px !important;
    padding:0.65rem 1.4rem !important;
    box-shadow:0 4px 20px rgba(102,126,234,.45) !important;
    transition:all .25s cubic-bezier(.4,0,.2,1) !important;
    font-family:'Inter',sans-serif !important;
}
button[kind="primary"]:hover {
    box-shadow:0 8px 30px rgba(102,126,234,.6) !important;
    transform:translateY(-1px) !important;
}
/* Demo button */
div[data-testid="stBaseButton-secondary"] button {
    background:white !important;
    color:#5b21b6 !important;
    border:2px solid #667eea !important;
    font-weight:700 !important;
}
div[data-testid="stBaseButton-secondary"] button p {
    color:#5b21b6 !important;
    font-weight:700 !important;
}
div[data-testid="stBaseButton-secondary"] button span {
    color:#5b21b6 !important;
}

/* ── Metric cards ────────────────────────────────────────── */
[data-testid="stMetric"] {
    background:#fff; border-radius:16px; padding:1.1rem 1.2rem;
    border:1px solid rgba(102,126,234,.12);
    box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 12px rgba(102,126,234,.08);
    transition:transform .2s,box-shadow .2s;
}
[data-testid="stMetric"]:hover {
    transform:translateY(-2px);
    box-shadow:0 4px 14px rgba(0,0,0,.1),0 8px 24px rgba(102,126,234,.15);
}
[data-testid="stMetricLabel"] { color:#64748b; font-size:0.72rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase; }
[data-testid="stMetricValue"] { color:#0f172a; font-weight:800; font-size:1.9rem; }

/* ── Insight cards ───────────────────────────────────────── */
.insight-card {
    display:flex; align-items:flex-start; gap:.9rem; background:#fff;
    border:1px solid rgba(102,126,234,.12); border-radius:14px;
    padding:1.1rem 1.3rem; margin-bottom:.7rem;
    box-shadow:0 1px 3px rgba(0,0,0,.05),0 4px 12px rgba(102,126,234,.06);
    transition:all .2s; animation:fadeUp .4s ease;
}
.insight-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(102,126,234,.15); }
@keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.insight-icon { font-size:1.5rem; line-height:1; flex-shrink:0; margin-top:.1rem; }
.insight-body { flex:1; }
.insight-num { font-size:.68rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.2rem; }
.insight-text { font-size:.92rem; line-height:1.7; color:#374151; margin:0; }

/* ── Step cards (landing) ────────────────────────────────── */
.step-card {
    background:#fff; border:1px solid rgba(102,126,234,.12);
    border-top:4px solid var(--step-color,#667eea); border-radius:14px;
    padding:1.3rem 1rem 1.1rem; text-align:center;
    box-shadow:0 2px 8px rgba(102,126,234,.08); transition:all .2s;
}
.step-card:hover { transform:translateY(-3px); box-shadow:0 8px 24px rgba(102,126,234,.18); }
.step-icon { font-size:2rem; margin-bottom:.5rem; }
.step-label { font-weight:800; font-size:.9rem; color:#0f172a; margin-bottom:.3rem; }
.step-desc { font-size:.78rem; color:#64748b; line-height:1.5; }
.arrow-col { display:flex; align-items:center; justify-content:center; padding-top:1.8rem; font-size:1.4rem; color:rgba(102,126,234,.3); }

/* ── Demo card (landing) ─────────────────────────────────── */
.demo-card {
    background:linear-gradient(135deg,rgba(102,126,234,.06) 0%,rgba(118,75,162,.06) 100%);
    border:1px solid rgba(102,126,234,.2); border-radius:16px; padding:1.5rem 1.8rem;
}
.demo-card h3 { color:#667eea; margin-top:0; }

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background:#fff; border-radius:14px; padding:4px; gap:2px;
    box-shadow:0 1px 3px rgba(0,0,0,.06); border:1px solid rgba(102,126,234,.1);
    overflow-x:auto;
}
.stTabs [data-baseweb="tab"] {
    border-radius:10px; color:#64748b; font-weight:500;
    font-size:.78rem; padding:.45rem .9rem; transition:all .2s; white-space:nowrap;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#667eea,#764ba2) !important;
    color:#fff !important; font-weight:700 !important;
}

/* ── Anomaly cards ───────────────────────────────────────── */
.anomaly-card {
    display:flex; align-items:flex-start; gap:.9rem; background:#fff;
    border:1px solid rgba(102,126,234,.1); border-radius:12px;
    padding:.9rem 1.1rem; margin-bottom:.55rem;
    box-shadow:0 1px 3px rgba(0,0,0,.04); transition:all .2s;
}
.anomaly-card:hover { box-shadow:0 4px 14px rgba(0,0,0,.09); }
.anomaly-badge {
    flex-shrink:0; padding:.2rem .55rem; border-radius:999px;
    font-size:.68rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; margin-top:.1rem;
}
/* Critical pulsing dot */
.pulse-dot {
    width:10px; height:10px; border-radius:50%; flex-shrink:0; margin-top:4px;
}
.pulse-dot.critical { animation:pulseDot 1.6s infinite; }
@keyframes pulseDot {
    0%{box-shadow:0 0 0 0 rgba(239,68,68,.5)}
    70%{box-shadow:0 0 0 8px rgba(239,68,68,0)}
    100%{box-shadow:0 0 0 0 rgba(239,68,68,0)}
}
.anomaly-body { flex:1; min-width:0; }
.anomaly-entity { font-weight:700; font-size:.9rem; color:#0f172a; margin-bottom:.12rem; }
.anomaly-meta { font-size:.76rem; color:#64748b; margin-bottom:.25rem; }
.anomaly-reason { font-size:.82rem; color:#374151; line-height:1.6; }
.severity-pill { display:inline-block; padding:.22rem .75rem; border-radius:999px; font-size:.72rem; font-weight:700; margin-right:.4rem; }

/* ── Decision cards ──────────────────────────────────────── */
.decision-card {
    background:#fff; border:1px solid rgba(102,126,234,.12); border-radius:16px;
    padding:1.2rem 1.4rem; margin-bottom:.85rem;
    box-shadow:0 2px 8px rgba(102,126,234,.07); transition:all .2s;
}
.decision-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(102,126,234,.14); }
.decision-header { display:flex; align-items:center; gap:.7rem; margin-bottom:.65rem; flex-wrap:wrap; }
.priority-badge { padding:.22rem .75rem; border-radius:999px; font-size:.68rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; flex-shrink:0; }
.timeline-badge { padding:.2rem .65rem; border-radius:8px; font-size:.7rem; font-weight:600; background:#f1f5f9; color:#475569; flex-shrink:0; }
.decision-action { font-size:.95rem; font-weight:700; color:#0f172a; line-height:1.55; margin-bottom:.55rem; }
.decision-meta { font-size:.82rem; color:#374151; line-height:1.65; }
.decision-meta strong { color:#0f172a; }
.domain-pill {
    display:inline-block;
    background:linear-gradient(135deg,rgba(102,126,234,.1),rgba(118,75,162,.1));
    border:1px solid rgba(102,126,234,.2); border-radius:999px;
    padding:.3rem 1.1rem; font-size:.78rem; font-weight:700;
    color:#667eea; margin-bottom:1.2rem; letter-spacing:.02em;
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

/* ── Chat section ────────────────────────────────────────── */
.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
}
.suggest-btn-row { margin-bottom: 0.8rem; }

/* ── Debate cards ────────────────────────────────────────── */
.debate-opt {
    background: #eff6ff; border-left: 4px solid #1a56db;
    border-radius: 10px; padding: 1rem 1.1rem; height: 100%;
}
.debate-crit {
    background: #fff1f2; border-right: 4px solid #dc2626;
    border-radius: 10px; padding: 1rem 1.1rem; height: 100%;
    text-align: right;
}
.debate-label {
    font-size: 0.7rem; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.debate-text { font-size: 0.88rem; color: #111827; line-height: 1.65; }
.debate-judge {
    background: #faf5ff; border: 2px solid #7c3aed;
    border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
    margin-top: 1rem;
}
.debate-judge-label {
    font-size: 0.75rem; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; color: #7c3aed; margin-bottom: 0.6rem;
}
.winner-badge {
    display: inline-block; padding: 0.2rem 0.8rem; border-radius: 999px;
    font-size: 0.75rem; font-weight: 700; margin-top: 0.5rem;
}

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background:#fff; border:1px solid rgba(102,126,234,.12) !important;
    border-radius:14px !important; box-shadow:0 1px 3px rgba(0,0,0,.05); overflow:hidden;
}
details summary { font-weight:600; color:#0f172a; }

/* ── Chat ────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background:#fff; border-radius:14px; border:1px solid rgba(102,126,234,.1);
    box-shadow:0 1px 3px rgba(0,0,0,.04); margin-bottom:.75rem;
    animation:fadeUp .3s ease;
}

/* ── Status (pipeline) ───────────────────────────────────── */
[data-testid="stStatus"] {
    background:#fff !important; border-radius:16px !important;
    border:1px solid rgba(102,126,234,.15) !important;
    box-shadow:0 4px 20px rgba(102,126,234,.1) !important;
}

/* ── Score card ──────────────────────────────────────────── */
.score-card {
    background:linear-gradient(135deg,rgba(102,126,234,.08) 0%,#fff 60%);
    border:1px solid rgba(102,126,234,.18); border-radius:20px;
    padding:1.8rem 2.2rem; margin-bottom:1.4rem;
    display:flex; align-items:center; gap:2rem; flex-wrap:wrap;
    box-shadow:0 4px 20px rgba(102,126,234,.1);
}
.score-number { font-size:4rem; font-weight:900; line-height:1; color:#0f172a; flex-shrink:0; }
.score-grade {
    font-size:1.9rem; font-weight:900; width:3rem; height:3rem; border-radius:50%;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; color:#fff;
    box-shadow:0 4px 14px rgba(0,0,0,.2);
}
.score-body { flex:1; min-width:220px; }
.score-label { font-size:.72rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:#64748b; margin-bottom:.3rem; }
.score-verdict { font-size:.9rem; color:#374151; line-height:1.7; }

/* ── KPI row ─────────────────────────────────────────────── */
.kpi-section { margin-bottom:1.2rem; }

/* ── Misc ────────────────────────────────────────────────── */
hr { border-color:rgba(102,126,234,.12) !important; }
[data-testid="stCaptionContainer"] { color:#94a3b8; font-size:.74rem; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid rgba(102,126,234,.1); }

/* ── Footer ──────────────────────────────────────────────── */
.aria-footer {
    text-align:center; padding:2rem 0 1rem;
    border-top:1px solid rgba(102,126,234,.1); margin-top:3rem;
    font-size:.75rem; color:#94a3b8; letter-spacing:.02em;
}
.aria-footer strong { color:#667eea; }
</style>
""")

# ── Session state ─────────────────────────────────────────────────────
for key, default in [
    ("results", None), ("error", None), ("prep", None),
    ("sql", None), ("pptx_path", None),
    ("chat_history", []), ("pending_chat", None),
    ("whatif", None), ("debate", None), ("blindspots", None),
    ("question_area", ""), ("auto_run_question", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Hero header ───────────────────────────────────────────────────────
st.markdown("""
<div class="hero-outer">
  <div class="hero-logo">✦ Aria</div>
  <div class="hero-sub-wrap">
    <span class="cycle-text ct1">Your AI Data Analyst</span>
    <span class="cycle-text ct2">Your AI Detective</span>
    <span class="cycle-text ct3">Your AI Advisor</span>
    <span class="cycle-text ct4">Your AI Forecaster</span>
  </div>
  <p class="hero-tagline">18 AI agents. One upload. Infinite insights.</p>
  <span class="hero-badge">Powered by Claude · Anthropic</span>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1.2rem;">
      <div style="font-size:2.2rem;font-weight:900;letter-spacing:-2px;
                  background:linear-gradient(135deg,#a78bfa,#818cf8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;">✦ Aria</div>
      <div style="font-size:.6rem;font-weight:700;letter-spacing:.25em;
                  color:rgba(255,255,255,.35);text-transform:uppercase;margin-top:.2rem;">
        AI Analytics Platform
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Any tabular CSV — column names are auto-detected by Claude.",
    )
    question_input = st.text_area(
        "Your question  *(optional)*",
        placeholder="e.g. Which regions have the highest risk factors?\nLeave blank for automatic discovery.",
        height=110,
        key="question_area",
    )
    model_label = st.selectbox("Model", list(MODELS.keys()))
    model_id    = MODELS[model_label]

    st.divider()
    run_btn  = st.button("▶  Run Analysis",          type="primary", use_container_width=True)
    demo_btn = st.button("⚡ Demo: Florida Health", use_container_width=True)

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

    # ── Sidebar footer ────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-footer">
        Built by <strong>Kiran Kumar Reddy Konapalli</strong><br>
        <span style="font-size:.65rem;opacity:.5;">18 Agents · Claude SDK · Streamlit</span>
    </div>
    """, unsafe_allow_html=True)


# ── Auto-run trigger (from "Ask This →" blind spot buttons) ──────────
if st.session_state.auto_run_question:
    st.session_state.auto_run_question = False
    _q = st.session_state.get("question_area", "").strip()
    if _q and (uploaded_file or st.session_state.get("prep")):
        if st.session_state.prep and st.session_state.prep.get("output_path"):
            _dp = st.session_state.prep["output_path"]
        elif uploaded_file:
            import tempfile as _tmp
            _f = _tmp.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb")
            _f.write(uploaded_file.getbuffer()); _f.close()
            _dp = _f.name
        else:
            _dp = None
        if _dp:
            def _run_auto():
                from agents.data_wrangler import DataWrangler as _DW
                from agents.analyst import Analyst as _An
                from agents.anomaly_agent import AnomalyAgent as _AnoA
                from agents.decision_agent import DecisionAgent as _DecA
                from agents.forecasting_agent import ForecastingAgent as _FoA
                from agents.stats_agent import StatsAgent as _StA
                from agents.viz_builder import VizBuilder as _VB
                from agents.report_writer import ReportWriter as _RW
                from agents.quality_agent import QualityAgent as _QA
                import pandas as _pd2
                raw_df = _pd2.read_csv(_dp)
                qo  = _QA(model=model_id).run(raw_df)
                wo  = _DW().run(_dp)
                ao  = _An(model=model_id).run(wo, question=_q)
                ano = _AnoA(model=model_id).run(wo["dataframe"], ao)
                do  = _DecA(model=model_id).run(_q, ao, ano, wo["dataframe"])
                fo  = _FoA(model=model_id).run(wo["dataframe"], ao)
                so  = _StA(model=model_id).run(wo["dataframe"], ao, _q)
                vzo = _VB().run(wo, ao)
                ro  = _RW(model=model_id).run(_q, ao, vzo)
                st.session_state.results = dict(
                    quality=qo, wrangler=wo, analyst=ao, anomaly=ano,
                    decision=do, forecast=fo, stats=so, viz=vzo, report=ro, question=_q,
                )
            with st.spinner(f"Re-running analysis: {_q[:60]}…"):
                try:
                    _run_auto()
                    st.session_state.blindspots = None
                except Exception as _e:
                    st.error(f"Auto-run failed: {_e}")
            st.rerun()

# ── Pipeline runner ───────────────────────────────────────────────────
def run_pipeline(data_path: str, question: str, model: str) -> None:
    st.session_state.results = None
    st.session_state.error   = None

    wrangler_output = analyst_output = viz_output = report_output = None

    with st.status("Running Aria pipeline…", expanded=True) as status:
        try:
            if not question.strip():
                st.info(
                    "No question entered — Aria will automatically discover "
                    "the most interesting insights in your data."
                )

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
    q = question_input.strip()   # empty string → orchestrator uses default
    if not uploaded_file and not st.session_state.prep:
        st.sidebar.error("Upload a CSV file — or use the Demo button.")
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
            <h3>⚡ Try the demo — no upload needed</h3>
            <p>Click <strong>Demo: Florida Health</strong> in the sidebar to analyse a 402-row
            Florida county health dataset (2019–2024). Includes 18 agents, Plotly charts,
            anomaly detection, AI decisions, forecasts, and a full report.</p>
            <p style="margin:0;font-size:.82rem;opacity:.6;">~60 seconds · 18 agents · Claude Sonnet 4.6</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="aria-footer">
      <strong>Aria v2.0</strong> &nbsp;·&nbsp; 18 AI Agents &nbsp;·&nbsp;
      Powered by Claude · Anthropic &nbsp;·&nbsp;
      Built by <strong>Kiran Kumar Reddy Konapalli</strong>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Plotly chart helper ───────────────────────────────────────────────
def _plotly(df: pd.DataFrame, spec: dict, col_desc: dict):
    """Generate an interactive Plotly figure from a chart spec dict."""
    chart_type = spec.get("type", "").lower()
    x_col  = spec.get("x")
    y_col  = spec.get("y")
    title  = spec.get("title", "")
    if x_col and x_col not in df.columns: return None
    if y_col and y_col not in df.columns: return None
    cat_cols   = df.select_dtypes(exclude="number").columns.tolist()
    entity_col = cat_cols[0] if cat_cols else None
    DARK = dict(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f0c29", font=dict(family="Inter,sans-serif", color="#e2e8f0", size=11),
        title=dict(text=title, font=dict(size=13), x=.5),
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155"),
    )
    try:
        if chart_type == "histogram":
            fig = px.histogram(df, x=x_col, color_discrete_sequence=["#667eea"],
                               nbins=25, opacity=.88)
            fig.update_traces(marker_line_width=0)
        elif chart_type == "scatter":
            plot_df = (df.groupby(entity_col)[[x_col, y_col]].mean().reset_index()
                       if entity_col and entity_col in df.columns
                       else df[[x_col, y_col]].dropna())
            fig = px.scatter(plot_df, x=x_col, y=y_col,
                             hover_name=entity_col if entity_col in plot_df.columns else None,
                             trendline="ols", trendline_color_override="#f093fb",
                             color_discrete_sequence=["#667eea"])
        elif chart_type == "bar":
            agg = (df.groupby(x_col)[y_col].mean()
                   .sort_values(ascending=False).head(25).reset_index())
            fig = px.bar(agg, x=y_col, y=x_col, orientation="h",
                         color=y_col,
                         color_continuous_scale=["#667eea", "#764ba2", "#f093fb"])
            fig.update_coloraxes(showscale=False)
        elif chart_type == "line":
            agg = (df.groupby(x_col)[y_col].mean().reset_index().sort_values(x_col))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=agg[x_col], y=agg[y_col], mode="lines+markers",
                line=dict(color="#667eea", width=2.5),
                marker=dict(size=7, color="#764ba2"),
                fill="tozeroy", fillcolor="rgba(102,126,234,.1)",
                name=y_col.replace("_", " ").title(),
            ))
        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            corr = numeric_df.corr().round(2)
            labels = [c.replace("_", "<br>") for c in corr.columns]
            fig = px.imshow(
                corr, x=labels, y=labels,
                color_continuous_scale=px.colors.diverging.RdBu_r[::-1],
                zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
            )
            fig.update_traces(textfont=dict(size=8))
        else:
            return None
        fig.update_layout(**DARK)
        return fig
    except Exception:
        return None

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

st.markdown(f"""
<div style="margin-bottom:1rem;">
  <div style="font-size:.7rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
              color:#667eea;margin-bottom:.3rem;">Analysis Complete</div>
  <div style="font-size:1.4rem;font-weight:800;color:#0f172a;line-height:1.3;">{question or "Exploratory Analysis"}</div>
</div>
""", unsafe_allow_html=True)

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

# KPI row
kpi_cols = st.columns(6, gap="small")
kpi_data = [
    ("🏅", "Quality",   f"{int(quality.get('overall_score',0))}/100",  quality.get("grade","?")),
    ("💡", "Insights",  len(analyst.get("insights",[])),               "AI-generated"),
    ("🔎", "Anomalies", len(anomaly.get("anomalies",[])),              f"{anomaly.get('severity_counts',{}).get('high',0)} high"),
    ("🎯", "Decisions", len(decision.get("decisions",[])),             "prioritised"),
    ("📈", "Forecasts", len(forecast.get("forecasts",[])),            "trends"),
    ("🔬", "Stat Tests",len(stats.get("tests_run",[])),               f"{len(stats.get('significant_findings',[]))} sig."),
]
for col, (icon, label, val, sub) in zip(kpi_cols, kpi_data):
    col.metric(f"{icon} {label}", val, sub)

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
    metas = analyst.get("suggested_charts", [])
    df_chart = wrangler["dataframe"]
    col_desc_c = analyst.get("column_descriptions", {})

    if not metas:
        st.info("No charts were suggested.")
    else:
        st.caption(f"{len(metas)} interactive Plotly charts — hover to explore, scroll to zoom")
        for i in range(0, len(metas), 2):
            cols = st.columns(2, gap="medium")
            for j in range(2):
                idx = i + j
                if idx >= len(metas): break
                with cols[j]:
                    fig = _plotly(df_chart, metas[idx], col_desc_c)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})
                    else:
                        path = viz.get("figure_paths", [])[idx] if idx < len(viz.get("figure_paths",[])) else None
                        if path and os.path.exists(path):
                            st.caption(metas[idx].get("title",""))
                            st.image(path, use_container_width=True)

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

# ── Chat with Aria ────────────────────────────────────────────────────
import anthropic as _anthropic

CHAT_SYSTEM = """\
You are Aria, an AI data analyst. You have just completed a full analysis of the user's dataset.
Answer follow-up questions using only the analysis results provided in the context below.
Be conversational but precise — always reference specific numbers, column names, or entities
from the analysis. If asked about something outside this dataset or analysis, politely say
you can only discuss this specific dataset and its findings.
Do not repeat the full context back; answer the question directly and concisely."""

SUGGESTIONS = [
    "Which finding is most urgent?",
    "Explain the key insights in simple terms",
    "What should I do first?",
]


def _build_chat_context(r: dict) -> str:
    """Compact structured summary of all analysis outputs for the chat system prompt."""
    analyst  = r.get("analyst", {})
    anomaly  = r.get("anomaly", {})
    decision = r.get("decision", {})
    forecast = r.get("forecast", {})
    stats    = r.get("stats", {})
    quality  = r.get("quality", {})
    wrangler = r.get("wrangler", {})
    qr       = wrangler.get("data_quality_report", {})
    col_desc = analyst.get("column_descriptions", {})

    lines = [
        "=== ANALYSIS CONTEXT ===",
        f"Question: {r.get('question', '')}",
        f"Dataset: {qr.get('final_row_count','?')} rows × {qr.get('final_column_count','?')} columns",
        "",
        f"DATA QUALITY: {quality.get('overall_score', 0):.0f}/100  Grade {quality.get('grade','?')}",
        quality.get("verdict", ""),
        "",
        "COLUMNS:",
    ]
    for col, info in list(col_desc.items())[:15]:
        unit = f" ({info['unit']})" if info.get("unit") else ""
        lines.append(f"  {col} [{info.get('role','?')}]{unit}: {info.get('description','')}")

    lines += ["", "KEY INSIGHTS:"]
    for i, ins in enumerate(analyst.get("insights", []), 1):
        lines.append(f"  {i}. {ins}")

    lines += ["", "TOP ANOMALIES:"]
    for a in anomaly.get("anomalies", [])[:5]:
        lines.append(f"  [{a.get('severity','').upper()}] {a.get('entity','')} — "
                     f"{a.get('column','')} = {a.get('value','')} | {a.get('reason','')[:80]}")

    lines += ["", "DECISIONS:"]
    for d in decision.get("decisions", []):
        lines.append(f"  [{d.get('priority','')}] {d.get('action','')[:100]}")
        lines.append(f"    Rationale: {d.get('rationale','')[:80]}")

    lines += ["", "FORECAST NARRATIVE:"]
    lines.append(forecast.get("narrative", "No forecast narrative."))

    sig = stats.get("significant_findings", [])
    lines += ["", f"SIGNIFICANT STATISTICS ({len(sig)} findings):"]
    for t in sig[:8]:
        lines.append(f"  {t.get('test_name','')} | {' × '.join(t.get('columns_tested',[]))} | "
                     f"p={t.get('p_value','?')} | {t.get('effect_label','')} effect")

    return "\n".join(lines)


def _stream_chat(context: str, history: list[dict], model: str):
    """Generator: streams Claude's response token by token."""
    client = _anthropic.Anthropic()
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": CHAT_SYSTEM + "\n\n" + context,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


if st.session_state.results:
    st.divider()

    # ── Header row ───────────────────────────────────────────────────
    head_l, head_r = st.columns([5, 1])
    head_l.subheader("💬 Chat with Aria")
    if head_r.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.pending_chat = None
        st.rerun()

    st.caption("Ask follow-up questions about your analysis. Aria knows every insight, anomaly, decision, and forecast.")

    context = _build_chat_context(st.session_state.results)

    # ── Suggested questions ──────────────────────────────────────────
    sug_cols = st.columns(3)
    for col, suggestion in zip(sug_cols, SUGGESTIONS):
        if col.button(suggestion, use_container_width=True, key=f"sug_{suggestion[:10]}"):
            st.session_state.pending_chat = suggestion
            st.rerun()

    # ── Conversation history ─────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="📊" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])

    # ── Process pending message (from suggestion buttons) ────────────
    if st.session_state.pending_chat:
        user_msg = st.session_state.pending_chat
        st.session_state.pending_chat = None
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant", avatar="📊"):
            try:
                reply = st.write_stream(
                    _stream_chat(context, st.session_state.chat_history, model_id)
                )
            except Exception as exc:
                reply = f"Sorry, I encountered an error: {exc}"
                st.error(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Chat input ───────────────────────────────────────────────────
    if user_input := st.chat_input("Ask Aria about your data…"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant", avatar="📊"):
            try:
                reply = st.write_stream(
                    _stream_chat(context, st.session_state.chat_history, model_id)
                )
            except Exception as exc:
                reply = f"Sorry, I encountered an error: {exc}"
                st.error(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ── What-If Simulator ─────────────────────────────────────────────────
if st.session_state.results:
    with st.expander("🔮 What-If Simulator — explore hypothetical scenarios", expanded=False):
        st.caption(
            "Describe a scenario in plain English. Aria will apply it to a copy of your data "
            "and estimate the downstream impact on correlated columns."
        )
        st.markdown(
            "**Examples:** &nbsp; `What if obesity dropped 5% in every county?` &nbsp;·&nbsp; "
            "`What if median income increased by $10,000?` &nbsp;·&nbsp; "
            "`What if uninsured rate fell to 10% in all counties?`",
            unsafe_allow_html=True,
        )
        wi_scenario = st.text_input(
            "Your scenario",
            placeholder="e.g. What if diabetes rate dropped 10% in all counties?",
            key="whatif_scenario_input",
        )
        wi_col1, wi_col2 = st.columns([3, 1])
        wi_run_btn   = wi_col1.button("🔮 Run Simulation", type="primary", use_container_width=True)
        wi_clear_btn = wi_col2.button("✕ Clear", use_container_width=True, key="wi_clear")

        if wi_clear_btn:
            st.session_state.whatif = None
            st.rerun()

        if wi_run_btn:
            if not wi_scenario.strip():
                st.error("Enter a scenario first.")
            else:
                r = st.session_state.results
                with st.spinner("Simulating scenario…"):
                    try:
                        st.session_state.whatif = WhatIfAgent(model=model_id).run(
                            dataframe       = r["wrangler"]["dataframe"],
                            analyst_output  = r["analyst"],
                            forecast_output = r["forecast"],
                            scenario        = wi_scenario.strip(),
                        )
                    except Exception as exc:
                        st.error(f"Simulation failed: {exc}")

        if st.session_state.whatif:
            wi = st.session_state.whatif

            if wi["scenario_parsed"].get("error"):
                st.error(f"Could not parse scenario: {wi['scenario_parsed']['error']}")
            else:
                parsed  = wi["scenario_parsed"]
                changes = wi["changes_applied"]
                impacts = wi["impact_summary"]

                st.markdown(
                    f"**Scenario:** {parsed.get('interpretation', parsed.get('target_column', ''))}  \n"
                    f"**Rows changed:** {changes.get('rows_changed', 0):,}  &nbsp;·&nbsp;  "
                    f"**{parsed['target_column']}:** "
                    f"{changes.get('original_mean', 0):.4g} → {changes.get('simulated_mean', 0):.4g} "
                    f"({changes.get('mean_delta', 0):+.4g})",
                    unsafe_allow_html=True,
                )

                st.divider()
                st.subheader("Simulated Impact")
                for line in wi["narrative"].split("\n"):
                    if line.strip():
                        st.markdown(line)

                if impacts:
                    st.divider()
                    st.caption(
                        "Estimated downstream effects (based on linear correlations — projections only)"
                    )
                    import pandas as _pd
                    rows = [
                        {
                            "Column":         col,
                            "Correlation r":  f"{imp['correlation']:+.3f}",
                            "Original Mean":  f"{imp['original_mean']:.4g}",
                            "Simulated Mean": f"{imp['simulated_mean']:.4g}",
                            "Est. % Change":  f"{'▲' if imp['estimated_pct_change'] > 0 else '▼'} {abs(imp['estimated_pct_change']):.1f}%",
                        }
                        for col, imp in list(impacts.items())[:8]
                    ]
                    st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)

                fig_path = wi.get("figure_path")
                if fig_path and os.path.exists(fig_path):
                    st.image(fig_path, use_container_width=True)

# ── Agent Debate ──────────────────────────────────────────────────────
if st.session_state.results:
    with st.expander("⚔️ Agent Debate — Optimist vs Critic", expanded=False):
        st.caption(
            "Two AI analysts debate the same data from opposite perspectives. "
            "Aria Judge delivers the final balanced verdict."
        )

        db_col1, db_col2 = st.columns([3, 1])
        start_debate = db_col1.button(
            "⚔️ Start Debate", type="primary", use_container_width=True
        )
        clear_debate = db_col2.button(
            "✕ Clear", use_container_width=True, key="debate_clear"
        )

        if clear_debate:
            st.session_state.debate = None
            st.rerun()

        if start_debate:
            r = st.session_state.results
            with st.spinner("Agents are debating… (5 Claude calls)"):
                try:
                    st.session_state.debate = DebateAgent(model=model_id).run(
                        analyst_output  = r["analyst"],
                        anomaly_output  = r["anomaly"],
                        decision_output = r["decision"],
                        question        = r["question"],
                    )
                except Exception as exc:
                    st.error(f"Debate failed: {exc}")

        if st.session_state.debate:
            db = st.session_state.debate

            def _debate_card(text: str, role: str, rnd: int) -> str:
                if role == "opt":
                    return (
                        f'<div class="debate-opt">'
                        f'<div class="debate-label" style="color:#1a56db;">🔵 Agent A — Optimist &nbsp; Round {rnd}</div>'
                        f'<div class="debate-text">{text}</div></div>'
                    )
                return (
                    f'<div class="debate-crit">'
                    f'<div class="debate-label" style="color:#dc2626;">Round {rnd} &nbsp; Critic — Agent B 🔴</div>'
                    f'<div class="debate-text">{text}</div></div>'
                )

            for rnd, (opt_key, crit_key) in enumerate(
                [("round1_optimist", "round1_critic"), ("round2_optimist", "round2_critic")], 1
            ):
                st.markdown(f"**— Round {rnd} —**")
                col_a, col_b = st.columns(2, gap="medium")
                with col_a:
                    st.markdown(_debate_card(db[opt_key], "opt", rnd), unsafe_allow_html=True)
                with col_b:
                    st.markdown(_debate_card(db[crit_key], "crit", rnd), unsafe_allow_html=True)

            # Judge verdict
            winner     = db.get("winner", "balanced")
            verdict    = db.get("judge_verdict", "")
            key_insight = db.get("key_insight", "")

            WINNER_STYLE = {
                "optimist": ("#dcfce7", "#166534", "🔵 Optimist wins"),
                "critic":   ("#fee2e2", "#991b1b", "🔴 Critic wins"),
                "balanced": ("#faf5ff", "#7c3aed", "⚖️ Balanced — no clear winner"),
            }
            wb, wf, wl = WINNER_STYLE.get(winner, WINNER_STYLE["balanced"])

            st.markdown(f"""
            <div class="debate-judge">
              <div class="debate-judge-label">⚖️ Aria Judge — Final Verdict</div>
              <div style="font-size:0.92rem;color:#111827;line-height:1.7;margin-bottom:0.7rem;">
                {verdict}
              </div>
              <span class="winner-badge" style="background:{wb};color:{wf};">{wl}</span>
              <div style="margin-top:0.8rem;font-size:0.82rem;color:#374151;">
                <strong>Key Insight:</strong> {key_insight}
              </div>
            </div>
            """, unsafe_allow_html=True)

# ── Blind Spots ───────────────────────────────────────────────────────
if st.session_state.results:
    with st.expander("🔍 Blind Spots — what this analysis missed", expanded=False):
        st.caption(
            "Aria audits its own analysis to find gaps — unexplored columns, "
            "ignored segments, and unanswered questions."
        )

        bs_col1, bs_col2 = st.columns([3, 1])
        detect_btn  = bs_col1.button(
            "🔍 Detect Blind Spots", type="primary", use_container_width=True
        )
        clear_bs = bs_col2.button("✕ Clear", use_container_width=True, key="bs_clear")

        if clear_bs:
            st.session_state.blindspots = None
            st.rerun()

        if detect_btn:
            r = st.session_state.results
            with st.spinner("Auditing the analysis for gaps…"):
                try:
                    st.session_state.blindspots = BlindSpotAgent(model=model_id).run(
                        dataframe      = r["wrangler"]["dataframe"],
                        analyst_output = r["analyst"],
                        question       = r["question"],
                    )
                except Exception as exc:
                    st.error(f"Blind spot detection failed: {exc}")

        if st.session_state.blindspots:
            bs_result = st.session_state.blindspots
            spots     = bs_result.get("blind_spots", [])
            summary   = bs_result.get("summary", "")

            if summary:
                st.markdown(f"**Overview:** {summary}")
                st.divider()

            SEV_STYLE = {
                "Critical":  ("#fee2e2", "#991b1b", "🔴"),
                "Important": ("#fff7ed", "#9a3412", "🟠"),
                "Minor":     ("#f0f4ff", "#1e40af", "🔵"),
            }

            for spot in spots:
                sev          = spot.get("severity", "Minor")
                bg, fg, icon = SEV_STYLE.get(sev, SEV_STYLE["Minor"])
                title        = spot.get("title", "")
                why          = spot.get("why_it_matters", "")
                sug_q        = spot.get("suggested_question", "")

                st.markdown(f"""
                <div style="background:{bg};border-left:4px solid {fg};border-radius:10px;
                            padding:1rem 1.2rem;margin-bottom:0.8rem;">
                  <div style="font-size:0.7rem;font-weight:800;letter-spacing:0.07em;
                              text-transform:uppercase;color:{fg};margin-bottom:0.4rem;">
                    {icon} {sev}
                  </div>
                  <div style="font-size:0.95rem;font-weight:700;color:#111827;
                              margin-bottom:0.35rem;">{title}</div>
                  <div style="font-size:0.85rem;color:#374151;line-height:1.65;
                              margin-bottom:0.5rem;">{why}</div>
                  <div style="font-size:0.8rem;color:#6b7280;">
                    <strong>Suggested question:</strong> {sug_q}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if sug_q and st.button(
                    f"Ask This → {sug_q[:55]}{'…' if len(sug_q) > 55 else ''}",
                    key=f"ask_{title[:20]}",
                    use_container_width=True,
                ):
                    st.session_state.question_area      = sug_q
                    st.session_state.auto_run_question  = True
                    st.session_state.blindspots         = None
                    st.rerun()

# ── Global footer ─────────────────────────────────────────────────────
st.markdown("""
<div class="aria-footer">
  <strong>Aria v2.0</strong> &nbsp;·&nbsp; 18 AI Agents &nbsp;·&nbsp;
  Powered by Claude · Anthropic &nbsp;·&nbsp;
  Built by <strong>Kiran Kumar Reddy Konapalli</strong>
</div>
""", unsafe_allow_html=True)
