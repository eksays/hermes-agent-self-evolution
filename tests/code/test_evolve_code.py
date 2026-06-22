"""Tests for CodeEvolver — code evolution orchestrator."""
from unittest.mock import patch, MagicMock
from pathlib import Path
import evolution.code.evolve_code as mod


def test_count_pytest_failures_parses_output():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = "FAILED tests/test_x.py::test_a - AssertionError\n1 failed, 5 passed\n"
        mock_run.return_value.stderr = ""
        result = mod._count_pytest_failures(Path("."))
        assert result == 1


def test_count_pytest_failures_no_failures():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "10 passed\n"
        mock_run.return_value.stderr = ""
        result = mod._count_pytest_failures(Path("."))
        assert result == 0


@patch("evolution.code.evolve_code.get_hermes_agent_path")
@patch("evolution.code.evolve_code.dspy.GEPA")
def test_dry_run_does_not_call_gepa(mock_gepa, mock_path):
    mock_path.return_value = Path(".")
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.read_text", return_value="def x(): pass"):
        mod.evolve_code(tool_name="test", dry_run=True, hermes_repo=".")
        mock_gepa.assert_not_called()
