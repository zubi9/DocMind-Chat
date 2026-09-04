"""Import and filter GaRAGe JSONL records into DocMind's eval format."""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable


GARAGE_FIELDS = (
    "sample_id",
    "question_date",
    "grounding",
    "question",
    "question_valid",
    "question_false_premise",
    "question_seeking",
    "question_sensitive",
    "question_type",
    "question_complexity",
    "question_category",
    "question_popularity",
    "evidence_relevant",
    "evidence_correct",
    "answer_generate",
    "answer_related_info",
    "answer_validate",
    "comments",
    "evidence_cited",
    "question_tag",
    "topic_tag",
)


class GaRAGeImportError(ValueError):
    """Raised when a GaRAGe record cannot be imported safely."""


def _flag(value: Any, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no", "0"}:
        return False
    raise GaRAGeImportError(f"{field} must be a boolean value")


def _require_text(record: dict[str, Any], field: str, line_number: int) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GaRAGeImportError(f"line {line_number}: '{field}' must be a non-empty string")
    return value.strip()


def _date_matches(value: Any, start: str | None, end: str | None) -> bool:
    if not start and not end:
        return True
    if not isinstance(value, str) or not value:
        return False
    candidate = value[:10]
    if start and candidate < start:
        return False
    if end and candidate > end:
        return False
    return True


def validate_record(record: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise GaRAGeImportError(f"line {line_number}: record must be a JSON object")
    question = _require_text(record, "question", line_number)
    sample_id = record.get("sample_id", f"garage-line-{line_number}")
    if not isinstance(sample_id, (str, int)):
        raise GaRAGeImportError(f"line {line_number}: 'sample_id' must be a string or integer")
    for field in ("question_valid", "question_false_premise", "question_seeking", "question_sensitive"):
        _flag(record.get(field), field)
    return {**record, "question": question, "sample_id": str(sample_id)}


def should_include(record: dict[str, Any], *, include_invalid: bool = False,
                   include_false_premise: bool = False, include_non_seeking: bool = False,
                   include_sensitive: bool = False, category: str | None = None,
                   complexity: str | None = None, question_tag: str | None = None,
                   topic_tag: str | None = None, start_date: str | None = None,
                   end_date: str | None = None) -> bool:
    if not include_invalid and _flag(record.get("question_valid"), "question_valid") is not True:
        return False
    if not include_false_premise and _flag(record.get("question_false_premise"), "question_false_premise") is True:
        return False
    if not include_non_seeking and _flag(record.get("question_seeking"), "question_seeking") is not True:
        return False
    if not include_sensitive and _flag(record.get("question_sensitive"), "question_sensitive") is True:
        return False
    for field, expected in (("question_category", category), ("question_complexity", complexity),
                            ("question_tag", question_tag), ("topic_tag", topic_tag)):
        if expected is not None and str(record.get(field, "")).casefold() != expected.casefold():
            return False
    return _date_matches(record.get("question_date"), start_date, end_date)


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {field: record[field] for field in GARAGE_FIELDS if field in record}
    expected_passages = []
    for position, label in enumerate(record.get("evidence_correct") or []):
        if str(label).strip().casefold() == "answer-the-question":
            expected_passages.append(f"garage:{record['sample_id']}:passage:{position}")
    item: dict[str, Any] = {
        "id": record["sample_id"],
        "question": record["question"],
        "reference_answer": record.get("answer_generate") or None,
        "expected_passages": expected_passages or None,
        "metadata": {"source": "GaRAGe", "record": metadata},
    }
    return item


def grounding_documents(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return passage-only documents suitable for indexing without answer leakage."""
    documents = []
    for position, passage in enumerate(record.get("grounding") or []):
        if not isinstance(passage, dict):
            continue
        text_keys = [key for key in passage if key.startswith("cite_")]
        if not text_keys or not isinstance(passage[text_keys[0]], str) or not passage[text_keys[0]].strip():
            continue
        documents.append({
            "_source_id": f"garage:{record['sample_id']}:passage:{position}",
            "_text": passage[text_keys[0]].strip(),
            "sample_id": record["sample_id"],
            "passage_index": position,
            "citation": text_keys[0],
            "age": passage.get("age"),
            "date": passage.get("date"),
            "provider": passage.get("provider"),
        })
    return documents


def read_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise GaRAGeImportError(f"Could not open GaRAGe dataset '{path}': {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GaRAGeImportError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            yield line_number, validate_record(raw, line_number)


def convert_dataset(path: Path, **filters: Any) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    counts = {"read": 0, "selected": 0, "filtered": 0}
    for line_number, record in read_records(path):
        counts["read"] += 1
        if should_include(record, **filters):
            selected.append(normalize_record(record))
        else:
            counts["filtered"] += 1
    counts["selected"] = len(selected)
    return selected, counts


def write_grounding_corpus(items: list[dict[str, Any]], output_dir: Path) -> dict[str, int]:
    """Materialize the selected items' grounding passages as JSON files only."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records = 0
    passages = 0
    for item in items:
        record = item.get("metadata", {}).get("record", {})
        if not record:
            continue
        records += 1
        for document in grounding_documents(record):
            filename = document["_source_id"].replace(":", "_") + ".json"
            write_dataset_file(document, output_dir / filename)
            passages += 1
    return {"records": records, "passages": passages}


def write_dataset_file(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def select_items(items: list[dict[str, Any]], *, limit: int | None = None,
                 seed: int | None = None) -> list[dict[str, Any]]:
    if limit is None or limit >= len(items):
        return items
    if limit < 0:
        raise GaRAGeImportError("limit must be zero or greater")
    selected = list(items)
    if seed is not None:
        random.Random(seed).shuffle(selected)
    return selected[:limit]


def write_dataset(items: list[dict[str, Any]], output: Path) -> None:
    payload = {
        "_readme": "Imported from GaRAGe. The benchmark dataset is kept outside this repository; review its license before use or distribution.",
        "source": "GaRAGe",
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)
