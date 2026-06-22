"""Daemon mode — periodic auto-evolution with drift detection.

Runs the evolution loop on an interval, checking Monitor metrics for drift.
Only triggers evolution for phases whose metrics are declining.
"""
import time
from pathlib import Path

from rich.console import Console

console = Console()

HISTORY_PATH = Path(__file__).resolve().parent / ".run_history.jsonl"
MONITOR_PATH = Path(__file__).resolve().parent / ".monitor.jsonl"


def run_daemon(interval_minutes=60, hermes_repo=None, force=False,
               iterations=10, write_back=False, threshold_drop=0.05):
    """Run evolution loop periodically, auto-triggering declining phases."""
    from evolution.loop.history import load_history
    from evolution.loop.scheduler import select_pending_phases
    from evolution.loop.evolve_all import _build_phase_funcs, HISTORY_PATH
    from evolution.loop.orchestrator import run_phases
    from evolution.monitor.monitor import Monitor

    monitor = Monitor(MONITOR_PATH)
    console.print(f"[bold cyan]Daemon mode[/bold cyan] — interval: {interval_minutes}m, "
                  f"threshold: {threshold_drop:.0%}")

    try:
        while True:
            history = load_history(HISTORY_PATH)
            latest = history.latest_per_phase()

            pending = select_pending_phases(
                latest, force=force, monitor=monitor
            )

            if pending:
                console.print(f"\n[bold]Phases to evolve: {', '.join(pending)}[/bold]")
                phase_funcs = _build_phase_funcs(
                    hermes_repo, iterations, write_back
                )
                result = run_phases(phase_funcs, phases=pending)

                # Record metrics
                for name, data in result.phases.items():
                    if data.get("status") == "success":
                        from evolution.monitor.monitor import MonitorRecord
                        from datetime import datetime
                        monitor.append(MonitorRecord(
                            timestamp=datetime.now().isoformat(),
                            git_sha="",
                            phase=name,
                            metric_name="fitness",
                            value=data.get("improvement", 0.0),
                            artifact_name=name,
                        ))

                console.print(f"  Result: {result.status}")
            else:
                console.print(f"\n[dim]No declining phases — sleeping {interval_minutes}m[/dim]")

            console.print(f"  Sleeping {interval_minutes} minutes... (Ctrl-C to stop)")
            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Daemon stopped.[/bold yellow]")
