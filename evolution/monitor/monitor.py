"""Monitor — append-only metric tracking for evolution phases.

Records per-phase metrics (accuracy, fitness, fail rate) with git SHA and
timestamp. Supports trend detection for drift-aware scheduling.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class MonitorRecord:
    timestamp: str
    git_sha: str
    phase: str
    metric_name: str  # "accuracy" | "fitness" | "fail_rate"
    value: float
    artifact_name: str


Trend = Literal["improving", "stable", "declining"]


class Monitor:
    """Append-only metric store with trend detection."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: MonitorRecord):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "timestamp": record.timestamp,
                    "git_sha": record.git_sha,
                    "phase": record.phase,
                    "metric_name": record.metric_name,
                    "value": record.value,
                    "artifact_name": record.artifact_name,
                })
                + "\n"
            )

    def load(self, phase: str | None = None,
             artifact: str | None = None) -> list[MonitorRecord]:
        if not self.path.exists():
            return []
        records: list[MonitorRecord] = []
        with open(self.path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if phase and d.get("phase") != phase:
                    continue
                if artifact and d.get("artifact_name") != artifact:
                    continue
                records.append(MonitorRecord(**d))
        return records

    def recent_trend(self, phase: str, artifact: str,
                     window: int = 5) -> Trend:
        """Return trend based on last `window` records for phase+artifact."""
        records = self.load(phase=phase, artifact=artifact)
        if len(records) < 3:
            return "stable"
        recent = records[-window:]
        values = [r.value for r in recent]
        slope = (values[-1] - values[0]) / max(1, len(values) - 1)
        if slope > 0.02:
            return "improving"
        if slope < -0.02:
            return "declining"
        return "stable"
