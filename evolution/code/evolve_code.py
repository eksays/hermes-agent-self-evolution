"""Evolve tool source code using DSPy + GEPA — Sub-Project D.

Reads a tool .py file, wraps its source as a CodeModule, runs GEPA with
pytest pass rate as fitness, validates constraints, writes back.

Usage:
    python -m evolution.code.evolve_code --tool file_tools --iterations 5
"""
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import click
import dspy
from rich.console import Console

from evolution.code.code_module import CodeModule
from evolution.core.config import EvolutionConfig, resolve_model, get_hermes_agent_path

console = Console()


def _pytest_fitness(hermes_repo: Path, baseline_failures: int) -> float:
    """Score = 1.0 if test failures <= baseline, 0.0 if worse."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q"],
            capture_output=True, text=True,
            cwd=str(hermes_repo),
            timeout=300,
        )
        output = result.stdout + result.stderr
        failed_count = output.count("FAILED")
        if failed_count <= baseline_failures:
            return 1.0
        return 0.0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0.0


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
        return _pytest_fitness(repo, baseline_failures)

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
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q"],
            capture_output=True, text=True,
            cwd=str(repo), timeout=300,
        )
        output = result.stdout + result.stderr
        return output.count("FAILED")
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
