#!/usr/bin/env python3
"""Convert a local GaRAGe JSONL file into DocMind's evaluation dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.garage_import import GaRAGeImportError, convert_dataset, select_items, write_dataset, write_grounding_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to GaRAGe_benchmark.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Output dataset path (defaults to data/eval_dataset.json)")
    parser.add_argument("--corpus-dir", type=Path, help="Also materialize selected grounding passages into this user_docs directory")
    parser.add_argument("--category")
    parser.add_argument("--complexity")
    parser.add_argument("--question-tag")
    parser.add_argument("--topic-tag")
    parser.add_argument("--start-date", help="Inclusive YYYY-MM-DD date")
    parser.add_argument("--end-date", help="Inclusive YYYY-MM-DD date")
    parser.add_argument("--limit", type=int, help="Keep at most this many records")
    parser.add_argument("--seed", type=int, help="Seed used when selecting a limited sample")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--include-false-premise", action="store_true")
    parser.add_argument("--include-non-seeking", action="store_true")
    parser.add_argument("--include-sensitive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Show selection counts without writing output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output or Path(__file__).resolve().parent.parent / "data" / "eval_dataset.json"
    filters = {
        "include_invalid": args.include_invalid,
        "include_false_premise": args.include_false_premise,
        "include_non_seeking": args.include_non_seeking,
        "include_sensitive": args.include_sensitive,
        "category": args.category,
        "complexity": args.complexity,
        "question_tag": args.question_tag,
        "topic_tag": args.topic_tag,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    try:
        items, counts = convert_dataset(args.input, **filters)
        items = select_items(items, limit=args.limit, seed=args.seed)
    except GaRAGeImportError as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 2

    print(f"Read {counts['read']} records; selected {len(items)}; filtered {counts['filtered']}")
    if args.dry_run:
        print("Dry run: no dataset written")
        return 0
    write_dataset(items, output)
    if args.corpus_dir:
        corpus_counts = write_grounding_corpus(items, args.corpus_dir)
        print(f"Wrote {corpus_counts['passages']} grounding passages to {args.corpus_dir}")
    print(f"Wrote {len(items)} items to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
