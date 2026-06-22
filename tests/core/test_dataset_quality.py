"""Tests for DatasetQuality — dedup, balance, validation."""
from dataclasses import dataclass
from evolution.core.dataset_quality import (
    dedup_examples, balance_by_kind, validate_examples, improve_dataset, QualityReport
)


@dataclass
class Ex:
    task: str
    kind: str = "clear"
    correct_tool: str = "read_file"


def test_dedup_removes_normalized_duplicates():
    items = [Ex("Find  the FILE"), Ex("find the file"), Ex("other task")]
    out = dedup_examples(items, key_fn=lambda e: e.task)
    assert len(out) == 2


def test_balance_caps_majority_class():
    items = [Ex(f"clear {i}", "clear") for i in range(10)] + [Ex("conf", "confusable")]
    out = balance_by_kind(items, kind_fn=lambda e: e.kind, max_ratio=3.0)
    clears = [e for e in out if e.kind == "clear"]
    assert len(clears) <= 3  # 3x the single minority example


def test_validate_drops_invalid():
    items = [Ex("ok task"), Ex("")]
    kept, dropped = validate_examples(items, validator_fn=lambda e: bool(e.task.strip()))
    assert len(kept) == 1 and dropped == 1


def test_improve_dataset_reports():
    items = [Ex("dup"), Ex("dup"), Ex("x", "confusable"), Ex("")]
    cleaned, report = improve_dataset(
        items,
        key_fn=lambda e: e.task,
        kind_fn=lambda e: e.kind,
        validator_fn=lambda e: bool(e.task.strip()),
    )
    assert isinstance(report, QualityReport)
    assert report.original_count == 4
    assert report.dropped_invalid >= 1
