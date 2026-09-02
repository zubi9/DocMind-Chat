#!/usr/bin/env python3
"""
CLI wrapper for the RAG evaluation harness.

Runs against a live DocMind Chat instance over HTTP (Docker or
`scripts/run_local.py`) — deliberately stdlib-only so it works with
nothing but `python3`, no venv or extra install required.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --llm-judge
    python scripts/evaluate.py --base-url http://localhost:8000
    python scripts/evaluate.py --save

Exit code is non-zero if any question errored (e.g. no embeddings built
yet), so this can be dropped straight into a CI step.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def fetch_json(url: str, method: str = "GET", timeout: int = 600) -> dict:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} from {url}:\n{body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Could not reach {url}: {e}", file=sys.stderr)
        print("Is the app running? (docker compose up, or scripts/run_local.py)", file=sys.stderr)
        sys.exit(1)


def fmt(value, pct: bool = False) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value * 100:.0f}%"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def print_report(report: dict) -> None:
    s = report["summary"]
    active = report.get("active_models", {})

    print("\nDocMind Chat — Evaluation Report")
    print(f"Generated: {report['generated_at']}")
    print(f"Models: LLM={active.get('llm')}  Embedding={active.get('embedding')}  Backend={active.get('backend')}\n")

    rows = [
        ("Questions", s["total_questions"]),
        ("Errors", s["errors"]),
        ("Avg latency (s)", fmt(s.get("avg_latency_s"))),
        ("Retrieval hit rate", fmt(s.get("retrieval_hit_rate"), pct=True)),
        ("Avg retrieval MRR", fmt(s.get("avg_retrieval_mrr"))),
        ("Avg keyword coverage", fmt(s.get("avg_keyword_coverage"), pct=True)),
        ("Avg semantic similarity", fmt(s.get("avg_semantic_similarity"))),
    ]
    if s.get("avg_judge_faithfulness") is not None:
        rows.append(("Avg judge faithfulness /5", fmt(s.get("avg_judge_faithfulness"))))
        rows.append(("Avg judge relevance /5", fmt(s.get("avg_judge_relevance"))))

    for label, value in rows:
        print(f"  {label:<28} {value}")

    print("\nPer-question:")
    for r in report["results"]:
        qid = r.get("id") or "?"
        question_preview = r["question"][:60] + ("..." if len(r["question"]) > 60 else "")
        if r.get("error"):
            print(f"  ✗ [{qid}] {question_preview}  ERROR: {r['error']}")
            continue

        bits = []
        if r.get("retrieval_hit") is not None:
            bits.append(f"hit={'Y' if r['retrieval_hit'] else 'N'}")
        if r.get("keyword_coverage") is not None:
            bits.append(f"kw={fmt(r['keyword_coverage'], pct=True)}")
        if r.get("semantic_similarity") is not None:
            bits.append(f"sim={fmt(r['semantic_similarity'])}")
        if r.get("judge_faithfulness") is not None:
            bits.append(f"faith={r['judge_faithfulness']}/5 rel={r['judge_relevance']}/5")
        bits.append(f"{r.get('latency_s')}s")
        print(f"  ✓ [{qid}] {question_preview}  ({', '.join(bits)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--llm-judge", action="store_true", help="Also score answers with the active LLM as a judge (slower).")
    parser.add_argument("--save", action="store_true", help="Save the full JSON report to eval_results/.")
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}/eval/run"
    if args.llm_judge:
        url += "?llm_judge=true"

    print(f"Running evaluation against {args.base_url} ...")
    if args.llm_judge:
        print("(--llm-judge enabled: one extra LLM call per question, this will take a while)")

    report = fetch_json(url, method="POST", timeout=1800 if args.llm_judge else 600)
    print_report(report)

    if args.save:
        out_dir = Path(__file__).resolve().parent.parent / "eval_results"
        out_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_path = out_dir / f"eval-{ts}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved full report to {out_path}")

    sys.exit(1 if report["summary"]["errors"] > 0 else 0)


if __name__ == "__main__":
    main()
