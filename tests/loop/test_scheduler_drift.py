"""Tests for drift-aware scheduler."""
from evolution.loop.scheduler import select_pending_phases


def test_force_returns_all():
    phases = select_pending_phases({"skills": "1", "tools": "1", "guidance": "1", "params": "1"},
                                   force=True)
    assert len(phases) == 4


def test_never_run_phases_included():
    phases = select_pending_phases({"skills": "1"}, force=False)
    assert "tools" in phases
    assert "guidance" in phases
    assert "skills" not in phases


def test_only_filters():
    phases = select_pending_phases({}, force=False, only=["skills", "tools"])
    assert "skills" in phases
    assert "tools" in phases
    assert "guidance" not in phases


def test_declining_prioritized():
    """Mock monitor returns 'declining' for tools → tools first."""
    class FakeMonitor:
        @staticmethod
        def recent_trend(phase, artifact):
            return "declining" if phase == "tools" else "stable"
    phases = select_pending_phases({"skills": "1", "tools": "1"},
                                   force=False, monitor=FakeMonitor())
    assert phases[0] == "tools"  # declining first
