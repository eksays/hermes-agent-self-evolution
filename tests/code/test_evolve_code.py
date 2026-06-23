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


# ── New tests for the fixed metric ────────────────────────────────────────

def test_extract_candidate_code_from_trace():
    """_extract_candidate_code reads instructions from the trace tuple."""
    mock_pred = MagicMock()
    mock_pred.signature.instructions = "def evolved(): pass"
    trace = [(mock_pred, {"dummy_input": "test"}, MagicMock())]
    assert mod._extract_candidate_code(trace) == "def evolved(): pass"


def test_extract_candidate_code_returns_none_for_empty_trace():
    assert mod._extract_candidate_code(None) is None
    assert mod._extract_candidate_code([]) is None


def test_pytest_fitness_gradient_baseline_zero():
    """With baseline=0, fitness should be 1.0 when current=0."""
    with patch.object(mod, "_count_pytest_failures", return_value=0):
        assert mod._pytest_fitness(Path("."), 0) == 1.0


def test_pytest_fitness_gradient_baseline_zero_degraded():
    """With baseline=0, fitness should decrease with new failures."""
    with patch.object(mod, "_count_pytest_failures", return_value=2):
        score = mod._pytest_fitness(Path("."), 0)
        assert 0.0 < score < 1.0  # 1.0 - 2*0.2 = 0.6


def test_pytest_fitness_gradient_improvement():
    """Reducing failures from baseline should score above 0.5."""
    with patch.object(mod, "_count_pytest_failures", return_value=1):
        score = mod._pytest_fitness(Path("."), 4)
        # ratio=0.25, score=1.0 - 0.25*0.5 = 0.875
        assert score > 0.5


def test_pytest_fitness_gradient_regression():
    """Doubling failures from baseline should score 0.0."""
    with patch.object(mod, "_count_pytest_failures", return_value=8):
        score = mod._pytest_fitness(Path("."), 4)
        # ratio=2.0, score=max(0.0, 1.0 - 2.0*0.5) = 0.0
        assert score == 0.0


def test_fitness_with_candidate_writes_and_restores(tmp_path):
    """The metric must write candidate code, run pytest, then restore original."""
    code_file = tmp_path / "tool.py"
    original = "def original(): pass"
    candidate = "def evolved(): pass"
    code_file.write_text(original, encoding="utf-8")

    # Track what was on disk when _pytest_fitness ran
    observed_contents = []

    def fake_fitness(repo, baseline):
        observed_contents.append(code_file.read_text(encoding="utf-8"))
        return 0.8

    with patch.object(mod, "_pytest_fitness", side_effect=fake_fitness):
        score = mod._pytest_fitness_with_candidate(
            code_file, original, tmp_path, baseline_failures=0,
            candidate_code=candidate,
        )

    # Metric ran against the candidate code
    assert observed_contents[0] == candidate
    # Original is restored after metric
    assert code_file.read_text(encoding="utf-8") == original
    assert score == 0.8


def test_fitness_with_candidate_syntax_error_returns_zero(tmp_path):
    """Candidate code with syntax errors should score 0 without running pytest."""
    code_file = tmp_path / "tool.py"
    original = "def original(): pass"
    code_file.write_text(original, encoding="utf-8")

    with patch.object(mod, "_pytest_fitness") as mock_fitness:
        score = mod._pytest_fitness_with_candidate(
            code_file, original, tmp_path, baseline_failures=0,
            candidate_code="def broken(: pass",  # invalid syntax
        )
    # Should not have called pytest at all
    mock_fitness.assert_not_called()
    assert score == 0.0
    # File should be untouched
    assert code_file.read_text(encoding="utf-8") == original


def test_fitness_with_candidate_restores_on_exception(tmp_path):
    """Original file must be restored even if pytest raises."""
    code_file = tmp_path / "tool.py"
    original = "def original(): pass"
    candidate = "def evolved(): pass"
    code_file.write_text(original, encoding="utf-8")

    with patch.object(mod, "_pytest_fitness", side_effect=RuntimeError("boom")):
        try:
            mod._pytest_fitness_with_candidate(
                code_file, original, tmp_path, baseline_failures=0,
                candidate_code=candidate,
            )
        except RuntimeError:
            pass

    # File must be restored regardless
    assert code_file.read_text(encoding="utf-8") == original
