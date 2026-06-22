"""Tests for ToolLabeler — Success + LLM-judge tool labeling."""
from unittest.mock import patch
from evolution.tools.tool_labeler import ToolLabeler
from evolution.core.trajectory_miner import ToolEpisode
from evolution.core.config import EvolutionConfig

TOOLS = [("search_files", "search file contents"), ("terminal", "run shell commands"),
         ("read_file", "read a file")]


def _episode(**kw):
    base = dict(task="search codebase for TODO", used_tool="terminal",
                used_params={"cmd": "grep TODO"}, session_success=True)
    base.update(kw)
    return ToolEpisode(**base)


def test_optimal_becomes_correct_tool():
    lab = ToolLabeler(EvolutionConfig())
    with patch.object(lab, "_judge_one",
                      return_value={"optimal": True, "better_tool": None,
                                    "alt_tools": [], "difficulty": "easy", "kind": "clear"}):
        out = lab.label([_episode(used_tool="search_files")], TOOLS)
    assert out[0].correct_tool == "search_files"
    assert out[0].kind == "clear"


def test_non_optimal_uses_better_tool_and_is_confusable():
    lab = ToolLabeler(EvolutionConfig())
    with patch.object(lab, "_judge_one",
                      return_value={"optimal": False, "better_tool": "search_files",
                                    "alt_tools": [], "difficulty": "medium", "kind": "confusable"}):
        out = lab.label([_episode(used_tool="terminal")], TOOLS)
    assert out[0].correct_tool == "search_files"
    assert "terminal" in out[0].alt_tools
    assert out[0].kind == "confusable"


def test_user_correction_overrides_judge():
    lab = ToolLabeler(EvolutionConfig())
    ep = _episode(used_tool="terminal", had_user_correction=True,
                  task="search for TODO; no, use search_files instead")
    with patch.object(lab, "_judge_one",
                      return_value={"optimal": True, "better_tool": None,
                                    "alt_tools": [], "difficulty": "easy", "kind": "clear"}):
        out = lab.label([ep], TOOLS)
    assert out[0].correct_tool == "search_files"
    assert out[0].kind == "confusable"


def test_params_attached_when_optimal():
    lab = ToolLabeler(EvolutionConfig())
    ep = _episode(used_tool="read_file", used_params={"path": "a.py"})
    with patch.object(lab, "_judge_one",
                      return_value={"optimal": True, "better_tool": None,
                                    "alt_tools": [], "difficulty": "easy", "kind": "clear"}):
        out = lab.label([ep], TOOLS, evolve_params=True)
    assert out[0].correct_params.get("path") == "a.py"
