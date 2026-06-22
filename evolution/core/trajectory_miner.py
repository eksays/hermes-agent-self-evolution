"""TrajectoryMiner — Stage 1: parse Hermes sessions into tool-call episodes.

Pure parsing, no API calls. Defensive against tool-call format variation.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from evolution.core.external_importers import SECRET_PATTERNS

_CORRECTION_RE = re.compile(
    r"\b(use|try|switch to|instead|should have used|don'?t use)\b", re.IGNORECASE
)
_ERROR_RE = re.compile(r"\b(error|failed|traceback|exception|not found)\b", re.IGNORECASE)


@dataclass
class ToolEpisode:
    task: str
    used_tool: str
    used_params: dict = field(default_factory=dict)
    session_success: bool = True
    had_user_correction: bool = False
    had_retry_or_error: bool = False
    session_id: str = ""
    source: str = "hermes"


def _contains_secret(text: str) -> bool:
    return bool(SECRET_PATTERNS.search(text or ""))


def _extract_tool_call(msg: dict):
    """Return (name, params) from a message, or (None, {}). Handles 3 shapes."""
    calls = msg.get("tool_calls")
    if isinstance(calls, list) and calls:
        call = calls[0]
        # shape A: {"function": {"name", "arguments": json-string}}
        if isinstance(call.get("function"), dict):
            fn = call["function"]
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return name, (args if isinstance(args, dict) else {})
        # shape B: {"name", "input": {...}}
        if call.get("name"):
            args = call.get("input", {}) or call.get("arguments", {})
            return call["name"], (args if isinstance(args, dict) else {})
    return None, {}


def _cap_params(params: dict) -> dict:
    return {k: (str(v)[:500]) for k, v in (params or {}).items()}


class TrajectoryMiner:
    SESSION_DIR = Path.home() / ".hermes" / "sessions"

    @staticmethod
    def extract_episodes(limit: int = 0, session_dir: Path | None = None) -> list[ToolEpisode]:
        """Extract tool-call episodes from Hermes session files."""
        directory = Path(session_dir) if session_dir else TrajectoryMiner.SESSION_DIR
        if not directory.exists():
            return []

        episodes: list[ToolEpisode] = []
        files = sorted(
            directory.glob("*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        for sf in files:
            try:
                data = json.loads(sf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            msgs = data.get("messages", [])
            if not msgs:
                continue
            session_id = data.get("session_id", sf.stem)

            # session-level error/success heuristic
            session_text = " ".join(str(m.get("content", "")) for m in msgs[-3:])
            session_success = not _ERROR_RE.search(session_text or "")

            for i, msg in enumerate(msgs):
                if msg.get("role") != "user":
                    continue
                task = (msg.get("content") or "").strip()
                if not task or len(task) < 10:
                    continue

                # find next assistant tool call before next user turn
                used_tool, used_params = None, {}
                correction, error = False, False
                for j in range(i + 1, len(msgs)):
                    role = msgs[j].get("role")
                    if role == "user":
                        if _CORRECTION_RE.search(msgs[j].get("content") or ""):
                            correction = True
                        break
                    if role in ("tool", "function"):
                        if _ERROR_RE.search(msgs[j].get("content") or ""):
                            error = True
                    if used_tool is None:
                        name, params = _extract_tool_call(msgs[j])
                        if name:
                            used_tool, used_params = name, params
                if not used_tool:
                    continue

                if _contains_secret(task) or any(
                    _contains_secret(str(v)) for v in used_params.values()
                ):
                    continue

                episodes.append(ToolEpisode(
                    task=task[:2000],
                    used_tool=used_tool,
                    used_params=_cap_params(used_params),
                    session_success=session_success,
                    had_user_correction=correction,
                    had_retry_or_error=error,
                    session_id=session_id,
                ))
                if limit and len(episodes) >= limit:
                    return episodes
        return episodes
