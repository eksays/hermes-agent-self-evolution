"""Tests for benchmark gate (no actual TBLite runs)."""

from pathlib import Path
import evolution.core.benchmark_gate as bg
from evolution.core.benchmark_gate import (
    GateResult,
    compare,
    run_fast_subset,
    TBLITE_FAST_SUBSET,
)


def test_fast_subset_has_20_tasks():
    assert len(TBLITE_FAST_SUBSET) == 20


def test_compare_passes_on_equal():
    baseline = {"pass_rate": 0.8, "passed": 16, "failed": 4, "total": 20, "per_task": {}}
    evolved = {"pass_rate": 0.8, "passed": 16, "failed": 4, "total": 20, "per_task": {}}
    result = compare(baseline, evolved)
    assert result.passed


def test_compare_passes_on_improvement():
    baseline = {"pass_rate": 0.7, "passed": 14, "failed": 6, "total": 20, "per_task": {}}
    evolved = {"pass_rate": 0.85, "passed": 17, "failed": 3, "total": 20, "per_task": {}}
    result = compare(baseline, evolved)
    assert result.passed
    assert result.regression < 0


def test_compare_fails_on_large_regression():
    baseline = {"pass_rate": 0.9, "passed": 18, "failed": 2, "total": 20, "per_task": {}}
    evolved = {"pass_rate": 0.8, "passed": 16, "failed": 4, "total": 20, "per_task": {}}
    result = compare(baseline, evolved, regression_threshold=0.05)
    # 0.9 - 0.8 = 0.10 regression > 0.05 threshold → FAIL
    assert not result.passed


def test_compare_uses_custom_threshold():
    baseline = {"pass_rate": 0.9, "passed": 18, "failed": 2, "total": 20, "per_task": {}}
    evolved = {"pass_rate": 0.85, "passed": 17, "failed": 3, "total": 20, "per_task": {}}
    # 0.9 - 0.85 = 0.05 regression. Threshold 0.1 → pass
    result = compare(baseline, evolved, regression_threshold=0.1)
    assert result.passed
    # Threshold 0.03 → fail
    result2 = compare(baseline, evolved, regression_threshold=0.03)
    assert not result2.passed


def test_compare_handles_error():
    baseline = {"error": "batch_runner not found", "pass_rate": 0.0, "passed": 0, "failed": 0, "total": 0, "per_task": {}}
    evolved = {"pass_rate": 0.0, "passed": 0, "failed": 0, "total": 0, "per_task": {}}
    result = compare(baseline, evolved)
    assert not result.passed
    assert "error" in result.message.lower()


def test_run_fast_subset_nonexistent_repo():
    result = run_fast_subset(Path("/nonexistent"), task_list=["test_task"])
    assert result["error"] is not None
    assert result["pass_rate"] == 0.0


# ── Sub-Project A: benchmark gate fix tests ──


def test_gate_applies_baseline_then_evolved_then_restores():
    calls = []
    _orig = bg.run_fast_subset
    try:
        def _mock_run(hermes_repo, tasks=None, **k):
            if not calls:
                return {"pass_rate": 0.90, "total": 1, "passed": 1, "failed": 0, "per_task": {}}
            return {"pass_rate": 0.90, "total": 1, "passed": 1, "failed": 0, "per_task": {}}
        bg.run_fast_subset = _mock_run
        res = bg.run_benchmark_gate(
            artifact_name="demo",
            hermes_repo=Path("."),
            apply_baseline=lambda: calls.append("baseline"),
            apply_evolved=lambda: calls.append("evolved"),
            restore=lambda: calls.append("restore"),
            mode="tblite",
        )
        assert "baseline" in calls, f"baseline not called: {calls}"
        assert "evolved" in calls, f"evolved not called: {calls}"
        assert calls[-1] == "restore", f"restore not last: {calls}"
    finally:
        bg.run_fast_subset = _orig


def test_gate_detects_real_regression():
    seq = iter([
        {"pass_rate": 0.90, "total": 20, "passed": 18, "failed": 2, "per_task": {}},
        {"pass_rate": 0.70, "total": 20, "passed": 14, "failed": 6, "per_task": {}},
    ])
    _orig = bg.run_fast_subset
    try:
        bg.run_fast_subset = lambda repo, tasks=None, **k: next(seq)
        res = bg.run_benchmark_gate(
            artifact_name="demo", hermes_repo=Path("."),
            apply_baseline=lambda: None, apply_evolved=lambda: None, restore=lambda: None,
            mode="tblite", regression_threshold=0.02,
        )
        assert res.passed is False, f"expected FAIL, got {res}"
        assert res.regression > 0.02
    finally:
        bg.run_fast_subset = _orig


def test_restore_called_even_on_error():
    def boom(*a, **k):
        raise RuntimeError("subset failed")
    _orig = bg.run_fast_subset
    restored = {"v": False}
    try:
        bg.run_fast_subset = boom
        bg.run_benchmark_gate(
            artifact_name="demo", hermes_repo=Path("."),
            apply_baseline=lambda: None, apply_evolved=lambda: None,
            restore=lambda: restored.__setitem__("v", True),
            mode="tblite",
        )
    except RuntimeError:
        pass
    finally:
        bg.run_fast_subset = _orig
    assert restored["v"] is True, "restore not called on error"


def test_missing_batch_runner_skips_gracefully():
    _orig = bg.run_fast_subset
    try:
        bg.run_fast_subset = lambda repo, tasks=None, **k: {
            "error": "batch_runner.py not found",
            "pass_rate": 0.0, "total": 0, "passed": 0, "failed": 0, "per_task": {}
        }
        res = bg.run_benchmark_gate(
            artifact_name="demo", hermes_repo=Path("."),
            apply_baseline=lambda: None, apply_evolved=lambda: None, restore=lambda: None,
            mode="tblite",
        )
        assert getattr(res, "skipped", False) is True
        assert res.passed is True
    finally:
        bg.run_fast_subset = _orig


def test_old_signature_falls_back_to_skip():
    """Backward compat: call with old signature must not crash and should skip."""
    res = bg.run_benchmark_gate(
        skill_name="demo",
        hermes_repo=Path("."),
    )
    assert getattr(res, "skipped", False) is True
    assert res.passed is True
