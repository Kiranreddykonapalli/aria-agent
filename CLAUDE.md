# Aria — Autonomous Reasoning & Insight Agent

A multi-agent data analytics system for healthcare data, built with Python and the Anthropic SDK. Designed as a portfolio project for a Data Science job search.

## Project Overview

Five specialist agents collaborate in a pipeline: a user question enters the Orchestrator, gets delegated to specialist agents, and exits as a polished report with charts.

```
User Question
     │
     ▼
 Orchestrator        ← routes tasks, manages agent coordination
     │
     ├──▶ Data Wrangler   ← cleans and preps raw healthcare data
     │         │
     ▼         ▼
   Analyst              ← runs statistics and surfaces insights
     │
     ▼
 Viz Builder            ← generates matplotlib charts
     │
     ▼
 Report Writer          ← composes the final narrative summary
     │
     ▼
 Final Report + Charts
```

## Stack

- **Language**: Python 3.11+
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **AI**: Anthropic Python SDK (`anthropic`)
- **Models**: claude-sonnet-4-6 (default), claude-haiku-4-5-20251001 (lightweight tasks)
- **Data domain**: Healthcare (patient records, clinical metrics, outcomes)

## Agent Responsibilities

| Agent | Role | Key Tools/Actions |
|---|---|---|
| `Orchestrator` | Receives user questions, decomposes tasks, routes to specialists, assembles results | Task planning, agent dispatch |
| `Data Wrangler` | Loads raw data, handles missing values, normalizes formats, validates schema | pandas, data quality checks |
| `Analyst` | Runs descriptive stats, correlations, trend analysis, hypothesis testing | pandas, numpy, scipy |
| `Viz Builder` | Creates charts from analyst output, saves figures to `output/figures/` | matplotlib, seaborn |
| `Report Writer` | Writes final human-readable summary combining insights and chart references | Claude API, markdown output |

## Project Structure

```
dataops-agent/
├── CLAUDE.md
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── data_wrangler.py
│   ├── analyst.py
│   ├── viz_builder.py
│   └── report_writer.py
├── data/
│   ├── raw/              # original healthcare datasets (never modified)
│   └── processed/        # cleaned outputs from Data Wrangler
├── output/
│   ├── figures/          # charts saved by Viz Builder
│   └── reports/          # final markdown/PDF reports
├── tools/                # shared utility functions used across agents
├── tests/
├── main.py               # entry point — takes a user question, runs the pipeline
├── requirements.txt
└── .env                  # ANTHROPIC_API_KEY (never commit)
```

## Environment Setup

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py "What are the top risk factors for readmission?"
```

### Authentication — pick one

**Option 1: API key**
Set `ANTHROPIC_API_KEY` in `.env`. The SDK picks it up automatically via `anthropic.Anthropic()`.

**Option 2: Claude Code session token (no key needed)**
Run `claude login` once in your terminal. The SDK detects the active Claude Code session automatically — no env var required. Useful during local development if you already have Claude Code installed.

## Development Guidelines

### Agent Design
- Each agent is a class with a single primary method (e.g., `run(input) -> output`).
- Agents communicate by passing structured Python dicts, not raw strings.
- The Orchestrator owns control flow; specialist agents are stateless and do not call each other directly.
- Use `claude-haiku-4-5-20251001` for lightweight formatting or classification tasks; use `claude-sonnet-4-6` for reasoning-heavy tasks (analysis, report writing).
- Enable prompt caching on large system prompts passed to Claude (use `cache_control` with `ephemeral` type).

### Data Handling
- Raw data in `data/raw/` is read-only. All mutations go to `data/processed/`.
- Healthcare data may contain PII — do not log raw patient fields; anonymize before passing to any Claude API call.
- Data Wrangler must validate schema and emit a data quality report before passing to Analyst.

### Outputs
- Viz Builder saves figures as `output/figures/<timestamp>_<chart_name>.png`.
- Report Writer saves final output as `output/reports/<timestamp>_report.md`.
- All output filenames include an ISO timestamp prefix for traceability.

### Code Style
- No comments unless the WHY is non-obvious.
- No docstrings on simple methods — clear names are sufficient.
- Raise explicit exceptions with context rather than silently returning `None`.
- Keep agent files focused: if a file exceeds ~200 lines, split utilities into `tools/`.

## Running the Pipeline

```bash
# Full pipeline
python main.py "What patient demographics are most associated with longer hospital stays?"

# Run a single agent in isolation (useful for development)
python -m agents.data_wrangler --input data/raw/patients.csv
```

## Testing

```bash
pytest tests/
```

- Unit test each agent with small synthetic DataFrames — do not use real patient data in tests.
- Integration tests live in `tests/integration/` and run the full pipeline on a sanitized sample.

## Portfolio Notes

This project demonstrates:
- Multi-agent orchestration with the Anthropic SDK
- Real-world data pipeline design (ingest → clean → analyze → visualize → report)
- Healthcare analytics domain knowledge
- Clean Python architecture suitable for production extension
