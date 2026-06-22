"""Tests for Monitor — metric tracking and trend detection."""
from pathlib import Path
from evolution.monitor.monitor import Monitor, MonitorRecord


def test_append_and_load(tmp_path):
    m = Monitor(tmp_path / "metrics.jsonl")
    m.append(MonitorRecord(timestamp="2026-01-01", git_sha="abc", phase="tools",
                           metric_name="accuracy", value=0.85, artifact_name="search_files"))
    records = m.load()
    assert len(records) == 1
    assert records[0].value == 0.85
    assert records[0].phase == "tools"


def test_filter_by_phase_and_artifact(tmp_path):
    m = Monitor(tmp_path / "metrics.jsonl")
    m.append(MonitorRecord(timestamp="2026-01-01", git_sha="a", phase="tools",
                           metric_name="accuracy", value=0.8, artifact_name="read_file"))
    m.append(MonitorRecord(timestamp="2026-01-02", git_sha="b", phase="guidance",
                           metric_name="fitness", value=0.7, artifact_name="memory"))
    assert len(m.load(phase="tools")) == 1
    assert len(m.load(artifact="read_file")) == 1
    assert len(m.load(phase="guidance", artifact="memory")) == 1


def test_recent_trend_declining(tmp_path):
    m = Monitor(tmp_path / "metrics.jsonl")
    for i, val in enumerate([0.9, 0.8, 0.7, 0.6]):
        m.append(MonitorRecord(timestamp=f"2026-01-0{i+1}", git_sha="x",
                               phase="tools", metric_name="accuracy",
                               value=val, artifact_name="search"))
    assert m.recent_trend("tools", "search") == "declining"


def test_recent_trend_improving(tmp_path):
    m = Monitor(tmp_path / "metrics.jsonl")
    for i, val in enumerate([0.5, 0.65, 0.75, 0.85]):
        m.append(MonitorRecord(timestamp=f"2026-01-0{i+1}", git_sha="x",
                               phase="tools", metric_name="accuracy",
                               value=val, artifact_name="search"))
    assert m.recent_trend("tools", "search") == "improving"


def test_recent_trend_stable(tmp_path):
    m = Monitor(tmp_path / "metrics.jsonl")
    for i in range(4):
        m.append(MonitorRecord(timestamp=f"2026-01-0{i+1}", git_sha="x",
                               phase="tools", metric_name="accuracy",
                               value=0.78, artifact_name="search"))
    assert m.recent_trend("tools", "search") == "stable"
