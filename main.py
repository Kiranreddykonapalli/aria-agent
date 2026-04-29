"""
Aria — Autonomous Reasoning & Insight Agent — pipeline entry point.

Usage:
    python main.py <csv_path> <question>
    python main.py data/raw/florida_health_2024.csv "Which counties have the worst health outcomes?"

Optional flags:
    --model   Claude model for reasoning agents  (default: claude-sonnet-4-6)
    --no-report   Skip printing the full report body to terminal
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()  # must run before any agent imports so the API key is available

from agents.orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aria — Autonomous Reasoning & Insight Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("csv_path", help="Path to the raw CSV file to analyse")
    parser.add_argument("question", help="Natural-language question about the dataset")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Claude model for Analyst and ReportWriter (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip printing the report body to terminal (report is still saved to disk)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nAria — Autonomous Reasoning & Insight Agent")
    print(f"  CSV      : {args.csv_path}")
    print(f"  Question : {args.question}")
    print(f"  Model    : {args.model}")

    orchestrator = Orchestrator(model=args.model)
    result = orchestrator.run(data_path=args.csv_path, question=args.question)

    if result["error"]:
        print("\nPipeline failed — see error above.")
        sys.exit(1)

    # ── Data quality summary ─────────────────────────────────────────
    qr = result["data_quality_report"]
    print("\n  Data Quality")
    print(f"    Rows          : {qr.get('final_row_count')}")
    print(f"    Nulls dropped : {qr.get('nulls_dropped')}")
    print(f"    Duplicates    : {qr.get('duplicate_rows_dropped')}")
    flags = qr.get("suspicious_values", [])
    print(f"    Flags         : {'; '.join(flags)}")

    # ── Insights summary ─────────────────────────────────────────────
    print("\n  Key Insights")
    for i, insight in enumerate(result["insights"], 1):
        # Wrap long lines at ~90 chars for terminal readability
        words = insight.split()
        line, lines = [], []
        for word in words:
            if sum(len(w) + 1 for w in line) + len(word) > 88:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        print(f"    {i}. {lines[0]}")
        for cont in lines[1:]:
            print(f"       {cont}")

    # ── Saved artefacts ──────────────────────────────────────────────
    print("\n  Saved Artefacts")
    print(f"    Report : {result['report_path']}")
    for fig in result["figure_paths"]:
        print(f"    Figure : {fig}")

    # ── Full report body ─────────────────────────────────────────────
    if not args.no_report:
        divider = "=" * 70
        print(f"\n{divider}")
        print(result["report_text"])
        print(divider)


if __name__ == "__main__":
    main()
