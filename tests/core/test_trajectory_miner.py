"""Tests for TrajectoryMiner — parse sessions into tool-call episodes."""
import json
from evolution.core.trajectory_miner import TrajectoryMiner, ToolEpisode


def _write_session(tmp_path, name, messages, **extra):
    data = {"session_id": name, "messages": messages, **extra}
    (tmp_path / f"{name}.json").write_text(json.dumps(data))


def test_parses_tool_calls_field(tmp_path):
    """Shape A: tool_calls with function.name + function.arguments (string)."""
    _write_session(tmp_path, "s1", [
        {"role": "user", "content": "find all python files importing os"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search_files", "arguments": '{"query": "import os"}'}}
        ]},
        {"role": "assistant", "content": "Done, found 3 files."},
    ])
    eps = TrajectoryMiner.extract_episodes(session_dir=tmp_path)
    assert len(eps) == 1
    assert eps[0].used_tool == "search_files"
    assert eps[0].task.startswith("find all python")
    assert eps[0].used_params.get("query") == "import os"


def test_parses_name_input_shape(tmp_path):
    """Shape B: tool_calls with name + input dict."""
    _write_session(tmp_path, "s2", [
        {"role": "user", "content": "read config file lines 1-20"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "read_file", "input": {"path": "config.py"}}
        ]},
        {"role": "tool", "name": "read_file", "content": "ok"},
        {"role": "assistant", "content": "Here are the lines."},
    ])
    eps = TrajectoryMiner.extract_episodes(session_dir=tmp_path)
    assert eps[0].used_tool == "read_file"
    assert eps[0].used_params.get("path") == "config.py"


def test_detects_user_correction(tmp_path):
    """User says 'no, use X instead' → had_user_correction=True."""
    _write_session(tmp_path, "s3", [
        {"role": "user", "content": "search the codebase for TODO"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "terminal", "arguments": '{"cmd": "grep TODO"}'}}
        ]},
        {"role": "user", "content": "no, use search_files instead"},
    ])
    eps = TrajectoryMiner.extract_episodes(session_dir=tmp_path)
    assert eps[0].had_user_correction is True


def test_detects_tool_error(tmp_path):
    """Error/exception in tool response → had_retry_or_error=True."""
    _write_session(tmp_path, "s4", [
        {"role": "user", "content": "open the file"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "read_file", "input": {"path": "missing.py"}}
        ]},
        {"role": "tool", "name": "read_file", "content": "Error: file not found (traceback)"},
    ])
    eps = TrajectoryMiner.extract_episodes(session_dir=tmp_path)
    assert eps[0].had_retry_or_error is True


def test_redacts_secrets(tmp_path):
    """Task containing API key → episode dropped."""
    _write_session(tmp_path, "s5", [
        {"role": "user", "content": "use my key sk-ant-api03-AAAABBBBCCCCDDDD to call the api"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"name": "web_search", "input": {"q": "x"}}
        ]},
    ])
    eps = TrajectoryMiner.extract_episodes(session_dir=tmp_path)
    assert eps == []


def test_malformed_session_is_safe(tmp_path):
    """Parse error, empty messages → [], never crash."""
    (tmp_path / "bad.json").write_text("{not valid json")
    (tmp_path / "empty.json").write_text(json.dumps({"messages": []}))
    eps = TrajectoryMiner.extract_episodes(session_dir=tmp_path)
    assert eps == []
