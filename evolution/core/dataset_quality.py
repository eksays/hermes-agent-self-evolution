"""DatasetQuality — Stage 3: dedup, balance, validate. Reusable across datasets."""
import random
import re
from dataclasses import dataclass, field


@dataclass
class QualityReport:
    original_count: int
    after_dedup: int
    after_balance: int
    dropped_invalid: int
    class_distribution: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def _normalize(text: str) -> str:
    """Normalize text for exact dedup: lowercase, strip punctuation, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (text or "").lower())).strip()


def dedup_examples(examples: list, key_fn) -> list:
    """Remove exact (after normalization) duplicates by key."""
    seen: set = set()
    out: list = []
    for ex in examples:
        norm = _normalize(key_fn(ex))
        if norm and norm not in seen:
            seen.add(norm)
            out.append(ex)
    return out


def balance_by_kind(examples: list, kind_fn, max_ratio: float = 3.0) -> list:
    """Cap majority class to max_ratio x minority size."""
    groups: dict = {}
    for ex in examples:
        groups.setdefault(kind_fn(ex), []).append(ex)
    if not groups:
        return examples
    minority = min(len(v) for v in groups.values())
    cap = max(1, int(minority * max_ratio))
    out: list = []
    for _, items in groups.items():
        if len(items) > cap:
            items = random.sample(items, cap)
        out.extend(items)
    random.shuffle(out)
    return out


def validate_examples(examples: list, validator_fn) -> tuple[list, int]:
    """Return (kept, dropped_count)."""
    kept = [e for e in examples if validator_fn(e)]
    return kept, len(examples) - len(kept)


def improve_dataset(examples, key_fn, kind_fn, validator_fn,
                     max_ratio: float = 3.0) -> tuple[list, QualityReport]:
    """Run dedup → validate → balance, return (cleaned, report)."""
    original = len(examples)
    valid, dropped = validate_examples(examples, validator_fn)
    deduped = dedup_examples(valid, key_fn)
    after_dedup = len(deduped)
    balanced = balance_by_kind(deduped, kind_fn, max_ratio=max_ratio)
    dist: dict = {}
    for ex in balanced:
        k = kind_fn(ex)
        dist[k] = dist.get(k, 0) + 1
    report = QualityReport(
        original_count=original,
        after_dedup=after_dedup,
        after_balance=len(balanced),
        dropped_invalid=dropped,
        class_distribution=dist,
    )
    return balanced, report
