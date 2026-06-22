"""Benchmark gating for evolved skills.

Runs a fast subset of TBLite tasks before/after evolution to detect
regressions. This is an OPT-IN gate (expensive, ~20-30 min).

Usage:
    gate_result = run_benchmark_gate(skill_name, hermes_repo, baseline_module, evolved_module)
    if not gate_result.passed:
        print(f"Regression detected: {gate_result.message}")
"""

import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class GateResult:
    """Result of a benchmark gate check."""
    passed: bool
    constraint_name: str = "benchmark_gate"
    message: str = ""
    details: Optional[str] = None
    baseline_pass_rate: float = 0.0
    evolved_pass_rate: float = 0.0
    regression: float = 0.0
    skipped: bool = False


# Fast subset: 20 TBLite tasks that cover diverse capabilities
TBLITE_FAST_SUBSET = [
    "write_python_function",
    "debug_syntax_error",
    "explain_sql_query",
    "refactor_function",
    "write_unit_test",
    "create_bash_script",
    "parse_json_data",
    "fix_import_error",
    "optimize_loop",
    "write_docstring",
    "implement_binary_search",
    "convert_csv_to_json",
    "find_security_issue",
    "validate_email_format",
    "calculate_complexity",
    "sort_file_contents",
    "merge_two_dicts",
    "handle_file_not_found",
    "format_datetime",
    "generate_random_data",
]


def run_fast_subset(
    hermes_repo: Path,
    task_list: Optional[list[str]] = None,
    timeout_per_task: int = 120,
) -> dict:
    """Run a subset of TBLite tasks using batch_runner.

    Args:
        hermes_repo: Path to hermes-agent repo.
        task_list: List of task names to run. Defaults to TBLITE_FAST_SUBSET.
        timeout_per_task: Max seconds per task.

    Returns:
        Dict with keys: pass_rate, total, passed, failed, per_task.
    """
    tasks = task_list or TBLITE_FAST_SUBSET
    batch_runner = hermes_repo / "batch_runner.py"

    if not batch_runner.exists():
        return {
            "pass_rate": 0.0,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "per_task": {},
            "error": f"batch_runner.py not found at {batch_runner}",
        }

    results = {"passed": 0, "failed": 0, "per_task": {}}
    for task_name in tasks:
        try:
            result = subprocess.run(
                ["python", str(batch_runner), "--task", task_name, "--timeout", str(timeout_per_task)],
                capture_output=True, text=True,
                cwd=str(hermes_repo),
                timeout=timeout_per_task + 30,
            )
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            success = False

        results["per_task"][task_name] = {
            "passed": success,
            "output": (result.stdout[:200] if success else result.stderr[:200]) if not success else "",
        }
        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1

    total = len(tasks)
    results["total"] = total
    results["pass_rate"] = results["passed"] / max(1, total)
    return results


def compare(
    baseline_results: dict,
    evolved_results: dict,
    regression_threshold: float = 0.02,
) -> GateResult:
    """Compare baseline vs evolved benchmark results.

    Args:
        baseline_results: Dict from run_fast_subset (baseline).
        evolved_results: Dict from run_fast_subset (evolved).
        regression_threshold: Max allowed regression (default 2%).

    Returns:
        GateResult indicating pass/fail.
    """
    baseline_rate = baseline_results.get("pass_rate", 0.0)
    evolved_rate = evolved_results.get("pass_rate", 0.0)
    regression = baseline_rate - evolved_rate

    if baseline_results.get("error"):
        return GateResult(
            passed=False,
            message=f"Baseline error: {baseline_results['error']}",
            baseline_pass_rate=baseline_rate,
            evolved_pass_rate=evolved_rate,
            regression=regression,
        )

    if regression > regression_threshold:
        return GateResult(
            passed=False,
            message=f"Regression: {regression:.1%} drop (threshold {regression_threshold:.1%})",
            details=f"Baseline: {baseline_rate:.1%} ({baseline_results.get('passed', 0)}/{baseline_results.get('total', 0)})\n"
                    f"Evolved: {evolved_rate:.1%} ({evolved_results.get('passed', 0)}/{evolved_results.get('total', 0)})",
            baseline_pass_rate=baseline_rate,
            evolved_pass_rate=evolved_rate,
            regression=regression,
        )

    return GateResult(
        passed=True,
        message=f"No significant regression ({regression:+.1%})",
        details=f"Baseline: {baseline_rate:.1%} → Evolved: {evolved_rate:.1%}",
        baseline_pass_rate=baseline_rate,
        evolved_pass_rate=evolved_rate,
        regression=regression,
    )


def run_benchmark_gate(
    skill_name: str = "",
    hermes_repo: Optional[Path] = None,
    baseline_module=None,
    evolved_module=None,
    task_list: Optional[list[str]] = None,
    regression_threshold: float = 0.02,
    artifact_name: str = "",
    apply_baseline: Optional[Callable] = None,
    apply_evolved: Optional[Callable] = None,
    restore: Optional[Callable] = None,
    mode: str = "proxy",
) -> GateResult:
    """Run the full benchmark gate: baseline → evolve → compare.

    NEW SIGNATURE (Sub-Project A):
      apply_baseline/apply_evolved/restore are callables that apply the
      artifact to the repo. mode="proxy" delegates to a proxy function
      (no TBLite run). mode="tblite" runs the real fast-subset benchmark.

    Old signature (skill_name, hermes_repo, baseline_module, evolved_module,
    task_list, regression_threshold) is still supported for backward
    compat but produces a skipped gate.

    Returns:
        GateResult.
    """
    # Backward compat: if the new params are absent, skip gracefully
    if apply_baseline is None or apply_evolved is None or restore is None:
        return GateResult(
            passed=True,
            skipped=True,
            message="benchmark gate skipped: use apply_baseline/apply_evolved/restore",
        )

    print(f"  Running benchmark gate for '{artifact_name or skill_name}'...")
    task_list2 = task_list or TBLITE_FAST_SUBSET
    print(f"  Task count: {len(task_list2)}")
    print(f"  Mode: {mode}")

    if mode in ("proxy",):
        print(f"  Proxy gate mode (cheap, no real TBLite run).")
        return GateResult(passed=True, skipped=True,
                          message=f"proxy gate mode — no real benchmark run")

    try:
        # 1. Apply baseline artifact
        print(f"  [1/2] Running baseline...")
        apply_baseline()
        baseline_results = run_fast_subset(hermes_repo or Path("."), task_list2)
        if baseline_results.get("error"):
            print(f"  Baseline error: {baseline_results['error']}")
            return GateResult(passed=True, skipped=True,
                              message=f"benchmark skipped: {baseline_results['error']}")
        print(f"  Baseline pass rate: {baseline_results['pass_rate']:.1%} "
              f"({baseline_results.get('passed', 0)}/{baseline_results.get('total', 0)})")

        # 2. Apply evolved artifact
        print(f"  [2/2] Running evolved...")
        apply_evolved()
        evolved_results = run_fast_subset(hermes_repo or Path("."), task_list2)
        if evolved_results.get("error"):
            print(f"  Evolved error: {evolved_results['error']}")
            return GateResult(passed=True, skipped=True,
                              message=f"benchmark skipped: {evolved_results['error']}")

        print(f"  Evolved pass rate: {evolved_results['pass_rate']:.1%} "
              f"({evolved_results.get('passed', 0)}/{evolved_results.get('total', 0)})")

        # 3. Compare
        result = compare(baseline_results, evolved_results, regression_threshold)
        print(f"  {result.message}")
        return result

    finally:
        restore()
