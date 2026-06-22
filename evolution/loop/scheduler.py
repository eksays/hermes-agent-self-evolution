"""Scheduler — decide which evolution phases to run.

Supports presence-based (legacy) and drift-aware (Sub-Project C) scheduling.
"""
from typing import Any

PHASE_ORDER = ["skills", "tools", "guidance", "params"]

PHASE_DEPENDENCIES = {
    "skills": [],
    "tools": [],
    "guidance": [],
    "params": ["tools"],
}


def select_pending_phases(
    latest_per_phase: dict,
    force: bool = False,
    only: list = None,
    monitor: Any = None,
) -> list:
    """Return ordered list of phase names that should be run.

    Priority (highest first):
    1. Phases with a declining trend (if monitor is provided).
    2. Phases never run.
    3. Phases not run in the longest time (fallback).
    4. ``--only`` filter and ``--force`` override as before.
    """
    candidates = PHASE_ORDER if only is None else [p for p in PHASE_ORDER if p in only]
    if force:
        return list(candidates)

    # 1: declining
    declining = []
    # 2: never run
    never = [p for p in candidates if p not in latest_per_phase]
    # 3: not run longest (by recency)
    remaining = [p for p in candidates if p not in never]

    if monitor is not None:
        for p in list(candidates):
            trend = getattr(monitor, "recent_trend", None)
            if trend:
                try:
                    t = monitor.recent_trend(p, p)
                    if t == "declining":
                        declining.append(p)
                        if p in never:
                            never.remove(p)
                        if p in remaining:
                            remaining.remove(p)
                except Exception:
                    pass

    # No monitor → legacy behavior: only phases that have never run
    if monitor is None:
        return never

    # Sort remaining by recency (oldest first = higher priority)
    remaining.sort(key=lambda p: latest_per_phase.get(p, ""))
    return declining + never + remaining
