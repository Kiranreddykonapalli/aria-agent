# 📊 Aria — Autonomous Reasoning & Insight Agent

> Ask anything about your data. Get answers in seconds.

Aria is a multi-agent AI analytics system that turns any CSV file into a full research report — complete with statistical insights, professional charts, and a written narrative — powered by Claude.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-D97706?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat)

---

## What is Aria?

Most data analysis workflows require stitching together multiple tools: a data cleaning step, a stats library, a charting tool, and then a human to write it all up. Aria replaces that workflow with a single question.

Upload a CSV, type a question in plain English, and Aria dispatches five specialist AI agents that collaborate in a pipeline — cleaning your data, running statistical analysis, generating charts, and composing a polished markdown report. The whole process takes under a minute.

Aria works with **any tabular dataset** — it auto-detects column types and meanings using Claude, so no configuration or schema definition is needed.

---

## How It Works

Aria runs five agents in sequence. Each agent receives the output of the previous one and passes its result forward.

```
User Question
      │
      ▼
 Orchestrator       ← coordinates the pipeline, handles errors
      │
      ▼
 Data Wrangler      ← loads the CSV, validates schema, cleans data,
      │               flags quality issues, saves to data/processed/
      ▼
  Analyst           ← profiles every column, detects column roles via Claude,
      │               surfaces 5 key findings, suggests charts to render
      ▼
 Viz Builder        ← renders histogram / bar / scatter / line / heatmap
      │               charts using matplotlib & seaborn, saves to output/figures/
      ▼
 Report Writer      ← sends all findings to Claude, writes a structured
      │               markdown report with Executive Summary, Key Findings,
      │               Visualizations, Recommendations, and Methodology
      ▼
 Report + Charts
```

### Agent Responsibilities

| Agent | Role |
|---|---|
| **Orchestrator** | Receives the user question and CSV path, runs the pipeline in order, catches errors, returns a unified result dict |
| **Data Wrangler** | Loads the CSV, enforces correct dtypes, checks for nulls and duplicates, flags suspicious values, emits a data quality report |
| **Analyst** | Computes descriptive stats (mean, median, std, skew, etc.) for all columns; uses Claude to classify column roles and extract 5 key insights; suggests chart types |
| **Viz Builder** | Renders up to 4 charts tailored to the analyst's findings; applies a consistent dark professional style; saves timestamped PNGs |
| **Report Writer** | Composes a 5-section markdown report using Claude with prompt caching; embeds figure references; saves to `output/reports/` |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **AI / LLM** | [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) · Claude Sonnet 4.6 / Haiku 4.5 |
| **Frontend** | [Streamlit](https://streamlit.io) |
| **Data** | pandas · numpy · scipy |
| **Visualisation** | matplotlib · seaborn |
| **Config** | python-dotenv |

**Model strategy:** Claude Sonnet 4.6 handles reasoning-heavy tasks (analysis, report writing). Claude Haiku 4.5 is available as a faster, lower-cost alternative selectable from the UI. Prompt caching is applied to all large system prompts to reduce latency and cost.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/dataops-agent.git
cd dataops-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Anthropic API key

```bash
cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=sk-ant-...
```

**Alternative:** If you have [Claude Code](https://claude.ai/code) installed, run `claude login` once — the SDK will pick up your session token automatically and no API key is needed.

### 4. Generate the demo dataset (optional)

```bash
python generate_synthetic_data.py
```

This creates `data/raw/florida_health_2024.csv` — a 402-row synthetic Florida county health dataset (67 counties × 6 years, 2019–2024) used by the built-in demo.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), click **Demo: Florida Health** to try it instantly, or upload your own CSV and ask a question.

---

## CLI Usage

Aria can also be run headlessly from the terminal:

```bash
# Analyse any CSV with a natural-language question
python main.py data/raw/florida_health_2024.csv \
  "Which counties have the worst health outcomes and what factors drive them?"

# Use Haiku for a faster run
python main.py mydata.csv "Summarise the key trends" --model claude-haiku-4-5-20251001

# Skip printing the full report body to terminal
python main.py mydata.csv "What are the top risk factors?" --no-report
```

Output files are saved automatically:
- `output/reports/<timestamp>_report.md` — full markdown report
- `output/figures/<timestamp>_<chart>.png` — one PNG per chart

---

## Project Structure

```
dataops-agent/
├── app.py                    # Streamlit frontend
├── main.py                   # CLI entry point
├── generate_synthetic_data.py
├── requirements.txt
├── .env.example
├── .gitignore
├── CLAUDE.md                 # Codebase instructions for Claude Code
├── agents/
│   ├── orchestrator.py       # Pipeline coordinator
│   ├── data_wrangler.py      # Data loading & cleaning
│   ├── analyst.py            # Statistical analysis & Claude insights
│   ├── viz_builder.py        # Chart rendering
│   └── report_writer.py      # Markdown report composition
├── data/
│   ├── raw/                  # Source files (read-only)
│   └── processed/            # Cleaned outputs (git-ignored)
├── output/
│   ├── figures/              # Chart PNGs (git-ignored)
│   └── reports/              # Final reports (git-ignored)
└── tools/                    # Shared utility functions
```

---

## Built by Kiran Kumar Reddy Konapalli

This project was built as a portfolio piece demonstrating multi-agent AI system design, real-world data pipeline architecture, and applied use of the Anthropic SDK.

It showcases:
- **Multi-agent orchestration** with the Anthropic Python SDK
- **Prompt engineering** — structured JSON outputs, prompt caching, role-specific system prompts
- **End-to-end data pipeline** design: ingest → clean → analyse → visualise → report
- **Full-stack Python** — CLI tool, Streamlit web app, modular agent architecture
- **Healthcare analytics** domain knowledge as a practical application context

Connect on [LinkedIn](https://linkedin.com/in/kiran-kumar-reddy-konapalli).
