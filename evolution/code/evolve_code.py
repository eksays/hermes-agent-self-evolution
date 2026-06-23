"""Evolve tool source code using DSPy + GEPA — Sub-Project D.

Reads a tool .py file, wraps its source as a CodeModule, runs GEPA with
pytest pass rate as fitness, validates constraints, writes back.

Usage:
    python -m evolution.code.evolve_code --tool file_tools --iterations 5
"""
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console

from evolution.code.code_module import CodeModule
from evolution.core.config import EvolutionConfig, resolve_model, get_hermes_agent_path

console = Console()

# ── Serialization lock ────────────────────────────────────────────────────
# GEPA may evaluate candidates in parallel threads. Only one candidate can
# be written to the tool file at a time, so we serialize pytest runs.
_eval_lock = threading.Lock()


def _extract_candidate_code(trace) -> Optional[str]:
    """Extract the candidate source code from a DSPy trace.

    In DSPy's evaluation pipeline the trace is a list of
    ``(predictor, inputs, outputs)`` tuples.  The predictor's
    ``signature.instructions`` holds the evolved code that GEPA is
    proposing for this candidate.
    """
    if not trace:
        return None
    try:
        for entry in trace:
            predictor = entry[0]
            instructions = getattr(
                getattr(predictor, "signature", None), "instructions", None
            )
            if instructions:
                return instructions
    except (TypeError, IndexError, AttributeError):
        pass
    return None


def _pytest_fitness_with_candidate(
    code_file: Path,
    original_source: str,
    hermes_repo: Path,
    baseline_failures: int,
    candidate_code: Optional[str],
) -> float:
    """Score a candidate by temporarily swapping it into the repo and running pytest.

    Returns a float in [0, 1]:
      - 1.0 if the candidate introduces no new failures (or fixes some).
      - Partial credit proportional to how many baseline failures remain
        vs. how many new ones were introduced.
      - 0.0 if the candidate causes a syntax error or all tests to fail.

    The original file is always restored, even on exceptions.
    """
    if not candidate_code or candidate_code.strip() == original_source.strip():
        # Unchanged candidate — baseline score.
        return _pytest_fitness(hermes_repo, baseline_failures)

    # Quick syntax check before the expensive pytest run.
    try:
        compile(candidate_code, str(code_file), "exec")
    except SyntaxError:
        return 0.0

    with _eval_lock:
        try:
            code_file.write_text(candidate_code, encoding="utf-8")
            return _pytest_fitness(hermes_repo, baseline_failures)
        finally:
            # Always restore the original source.
            code_file.write_text(original_source, encoding="utf-8")


def _pytest_fitness(hermes_repo: Path, baseline_failures: int) -> float:
    """Run pytest and return a fitness score in [0, 1].

    Provides a gradient rather than a binary pass/fail so GEPA can
    distinguish "slightly better" from "same" or "worse":
      - baseline_failures == 0: 1.0 if still 0, else scaled down.
      - baseline_failures > 0: reward for reducing failures; penalize
        for increasing them.
    """
    current = _count_pytest_failures(hermes_repo)
    if baseline_failures == 0:
        # Perfect baseline — any new failure is bad.
        return 1.0 if current == 0 else max(0.0, 1.0 - current * 0.2)
    # Imperfect baseline — reward improvement, penalize regression.
    # Ratio: 1.0 when current == 0, 0.5 at baseline, 0.0 when double.
    ratio = current / baseline_failures
    return max(0.0, min(1.0, 1.0 - ratio * 0.5))


def evolve_code(
    tool_name: str,
    iterations: int = 5,
    optimizer_model: str = "openai/gpt-4o-mini",
    eval_model: str = "openai/gpt-4o-mini",
    hermes_repo: Optional[str] = None,
    dry_run: bool = False,
    write_back: bool = False,
):
    """Evolve a tool's source code via GEPA."""
    repo = Path(hermes_repo) if hermes_repo else get_hermes_agent_path()
    code_file = repo / "tools" / f"{tool_name}.py"
    if not code_file.exists():
        console.print(f"[red]✗ {code_file} not found[/red]")
        sys.exit(1)

    source_text = code_file.read_text(encoding="utf-8")
    console.print(f"  Loaded: {code_file.relative_to(repo)} ({len(source_text):,} chars)")

    if dry_run:
        console.print("[bold green]DRY RUN[/bold green]")
        return

    config = EvolutionConfig(
        iterations=iterations, optimizer_model=optimizer_model,
        eval_model=eval_model, judge_model=eval_model,
    )
    optimizer_model_resolved = resolve_model(optimizer_model)
    eval_model_resolved = resolve_model(eval_model)

    # Baseline test failures
    baseline_failures = _count_pytest_failures(repo)
    console.print(f"  Baseline pytest failures: {baseline_failures}")

    # Pre-flight
    lm = dspy.LM(eval_model_resolved)
    dspy.configure(lm=lm)

    # CodeModule + GEPA
    module = CodeModule(source_text)
    reflection_lm = dspy.LM(optimizer_model_resolved, temperature=1.0)

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        """GEPA metric: write candidate code to disk, run pytest, restore.

        The candidate's evolved source lives in the trace's predictor
        ``signature.instructions``.  We write it to the tool file, run
        the test suite, then unconditionally restore the original.
        """
        candidate_code = _extract_candidate_code(trace)
        return _pytest_fitness_with_candidate(
            code_file, source_text, repo, baseline_failures, candidate_code,
        )

    start = time.time()
    optimizer = dspy.GEPA(metric=metric, max_full_evals=iterations, reflection_lm=reflection_lm)
    # GEPA needs at least 1 example
    dummy_train = [dspy.Example(dummy_input="test").with_inputs("dummy_input")]

    try:
        optimized = optimizer.compile(module, trainset=dummy_train)
    except Exception as e:
        console.print(f"[red]✗ GEPA compilation failed: {e}[/red]")
        sys.exit(1)

    evolved_code = optimized.code
    elapsed = time.time() - start

    # Write back
    if write_back:
        code_file.write_text(evolved_code, encoding="utf-8")
        console.print(f"  [green]✓ Wrote evolved code to {code_file}[/green]")
    else:
        out = Path("output") / "code" / tool_name
        out.mkdir(parents=True, exist_ok=True)
        (out / "evolved.py").write_text(evolved_code, encoding="utf-8")
        console.print(f"  Output saved to {out}/evolved.py")

    console.print(f"  Code evolution completed in {elapsed:.1f}s")


def _count_pytest_failures(repo: Path) -> int:
    import re as _re
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            capture_output=True, text=True,
            cwd=str(repo), timeout=300,
        )
        # Use exit code: 0=all pass, 1=some failed, 2+=error
        if result.returncode == 0:
            return 0
        # Parse "N failed" from summary line for accurate count
        m = _re.search(r"(\d+) failed", result.stdout + result.stderr)
        return int(m.group(1)) if m else (1 if result.returncode == 1 else 0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0


@click.command()
@click.option("--tool", required=True, help="Tool name (e.g. file_tools)")
@click.option("--iterations", default=5, help="GEPA iterations")
@click.option("--optimizer-model", default="openai/gpt-4o-mini")
@click.option("--eval-model", default="openai/gpt-4o-mini")
@click.option("--hermes-repo", default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--write-back", is_flag=True)
def main(tool, iterations, optimizer_model, eval_model, hermes_repo, dry_run, write_back):
    evolve_code(
        tool_name=tool, iterations=iterations,
        optimizer_model=optimizer_model, eval_model=eval_model,
        hermes_repo=hermes_repo, dry_run=dry_run, write_back=write_back,
    )


if __name__ == "__main__":
    main()
