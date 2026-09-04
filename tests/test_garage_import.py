import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.core.garage_import import (
    GaRAGeImportError,
    convert_dataset,
    select_items,
    write_dataset,
    write_grounding_corpus,
)
from app.core.evaluation import _retrieval_metrics


def record(sample_id, question="What?", **overrides):
    value = {
        "sample_id": sample_id,
        "question": question,
        "question_valid": True,
        "question_false_premise": False,
        "question_seeking": True,
        "question_sensitive": False,
        "question_category": "Science",
        "question_complexity": "Simple",
        "question_tag": "web",
        "topic_tag": "web",
        "question_date": "2025-01-15T00:00:00Z",
        "answer_generate": "An answer.",
        "grounding": [{"cite_1": "Evidence", "provider": "web", "age": "1", "date": "2025"}],
        "evidence_correct": ["ANSWER-THE-QUESTION"],
        "comments": "checked",
    }
    value.update(overrides)
    return value


class GarageImportTests(unittest.TestCase):
    def write_jsonl(self, records):
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False)
        with handle:
            for value in records:
                handle.write(json.dumps(value) + "\n")
        return Path(handle.name)

    def test_default_filters_and_preserves_metadata(self):
        path = self.write_jsonl([
            record("keep"),
            record("invalid", question_valid=False),
            record("premise", question_false_premise=True),
            record("sensitive", question_sensitive=True),
        ])
        items, counts = convert_dataset(path)
        self.assertEqual(counts, {"read": 4, "selected": 1, "filtered": 3})
        self.assertEqual(items[0]["id"], "keep")
        self.assertEqual(items[0]["reference_answer"], "An answer.")
        self.assertEqual(items[0]["metadata"]["record"]["grounding"][0]["cite_1"], "Evidence")
        self.assertEqual(items[0]["expected_passages"], ["garage:keep:passage:0"])

    def test_optional_filters_and_limit_are_deterministic(self):
        path = self.write_jsonl([
            record("science-1"),
            record("finance", question_category="Finance"),
            record("science-2", question_date="2026-02-01T00:00:00Z"),
        ])
        items, _ = convert_dataset(path, category="Science", start_date="2025-01-01", end_date="2025-12-31")
        self.assertEqual([item["id"] for item in items], ["science-1"])
        source = [{"id": str(index)} for index in range(10)]
        self.assertEqual(select_items(source, limit=4, seed=9), select_items(source, limit=4, seed=9))

    def test_malformed_json_reports_line(self):
        path = self.write_jsonl([record("ok")])
        path.write_text(json.dumps(record("ok")) + "\n{broken\n", encoding="utf-8")
        with self.assertRaisesRegex(GaRAGeImportError, r"line 2: invalid JSON"):
            convert_dataset(path)

    def test_write_dataset_is_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "eval_dataset.json"
            write_dataset([{"id": "x", "question": "Q", "reference_answer": "A"}], output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "GaRAGe")
            self.assertEqual(payload["items"][0]["id"], "x")

    def test_grounding_corpus_contains_passage_only_documents(self):
        path = self.write_jsonl([record("corpus")])
        items, _ = convert_dataset(path)
        with tempfile.TemporaryDirectory() as directory:
            counts = write_grounding_corpus(items, Path(directory))
            self.assertEqual(counts, {"records": 1, "passages": 1})
            files = list(Path(directory).glob("*.json"))
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["_source_id"], "garage:corpus:passage:0")
            self.assertEqual(payload["_text"], "Evidence")
            self.assertNotIn("answer_generate", payload)

    def test_passage_retrieval_metrics(self):
        metrics = _retrieval_metrics(
            ["garage:a:passage:2", "garage:a:passage:8", "garage:a:passage:99"],
            ["garage:a:passage:2", "garage:a:passage:8"],
        )
        self.assertTrue(metrics["retrieval_hit"])
        self.assertEqual(metrics["retrieval_mrr"], 1.0)
        self.assertEqual(metrics["retrieval_recall"], 1.0)
        self.assertEqual(metrics["retrieval_precision"], 0.6667)
        self.assertGreater(metrics["retrieval_ndcg"], 0.9)

    def test_cli_dry_run_and_write(self):
        input_path = self.write_jsonl([record("cli")])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "eval_dataset.json"
            command = [
                sys.executable,
                str(Path(__file__).parents[1] / "scripts" / "import_garage.py"),
                str(input_path),
                "--output",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Wrote 1 items", result.stdout)
            self.assertEqual(json.loads(output.read_text())["items"][0]["id"], "cli")


if __name__ == "__main__":
    unittest.main()
