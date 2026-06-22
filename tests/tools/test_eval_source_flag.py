"""Tests for --eval-source flag in tool description evolution."""
from unittest.mock import MagicMock
from evolution.tools import evolve_tool_descriptions as mod


def test_synthetic_source_calls_generate():
    builder = MagicMock()
    mod._build_dataset(builder, all_tools=[("a", "b")], eval_source="synthetic",
                       evolve_params=False, config=MagicMock(min_real_examples=12))
    builder.generate.assert_called_once()
    builder.build_from_sessions.assert_not_called()


def test_sessions_source_calls_build_from_sessions():
    builder = MagicMock()
    mod._build_dataset(builder, all_tools=[("a", "b")], eval_source="sessions",
                       evolve_params=False, config=MagicMock(min_real_examples=12))
    builder.build_from_sessions.assert_called_once()
    builder.generate.assert_not_called()
