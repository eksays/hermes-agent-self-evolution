"""Tests for build_from_sessions — real-data dataset with synthetic fallback."""
from unittest.mock import patch
from evolution.tools.tool_dataset import ToolDatasetBuilder, ToolSelectionExample
from evolution.core.config import EvolutionConfig

TOOLS = [("search_files", "search"), ("terminal", "shell"), ("read_file", "read")]


def _examples(n, kind="clear"):
    return [ToolSelectionExample(task=f"task {i}", correct_tool="search_files", kind=kind)
            for i in range(n)]


def test_uses_real_data_when_enough():
    b = ToolDatasetBuilder(EvolutionConfig())
    with patch("evolution.tools.tool_dataset.TrajectoryMiner.extract_episodes", return_value=["e"]), \
         patch("evolution.tools.tool_dataset.ToolLabeler.label", return_value=_examples(20)):
        ds = b.build_from_sessions(TOOLS, min_real=12)
    assert len(ds.all_examples) >= 12
    assert len(ds.train) >= 1 and len(ds.holdout) >= 1


def test_falls_back_to_synthetic_on_cold_start():
    b = ToolDatasetBuilder(EvolutionConfig())
    mock_synth = type("D", (), {"all_examples": _examples(20)})()
    with patch("evolution.tools.tool_dataset.TrajectoryMiner.extract_episodes", return_value=[]), \
         patch("evolution.tools.tool_dataset.ToolLabeler.label", return_value=_examples(2)), \
         patch.object(b, "generate", return_value=mock_synth):
        ds = b.build_from_sessions(TOOLS, min_real=12)
    assert len(ds.all_examples) >= 12  # topped up with synthetic
