"""Tests for MultiJudge — judge panel with agreement scoring."""
from unittest.mock import MagicMock, patch
from evolution.core.multi_judge import MultiJudge, PanelVerdict
from evolution.core.fitness import FitnessScore
from evolution.core.config import EvolutionConfig


def _fake_score(value):
    return FitnessScore(correctness=value, procedure_following=value, conciseness=value)


def test_median_and_agreement_high_when_judges_agree():
    cfg = EvolutionConfig()
    mj = MultiJudge(cfg, judge_models=["m1", "m2", "m3"])
    with patch.object(mj, "_score_one", side_effect=[
        _fake_score(0.80), _fake_score(0.82), _fake_score(0.78)
    ]):
        v = mj.score("task", "expected", "output", "skill")
    assert 0.78 <= v.median_score <= 0.82
    assert v.agreement > 0.8
    assert v.low_confidence is False
    assert len(v.individual_scores) == 3


def test_low_agreement_flagged():
    cfg = EvolutionConfig()
    mj = MultiJudge(cfg, judge_models=["m1", "m2", "m3"], agreement_threshold=0.6)
    with patch.object(mj, "_score_one", side_effect=[
        _fake_score(0.10), _fake_score(0.90), _fake_score(0.50)
    ]):
        v = mj.score("task", "expected", "output", "skill")
    assert v.low_confidence is True


def test_single_model_falls_back_low_confidence():
    cfg = EvolutionConfig()
    mj = MultiJudge(cfg, judge_models=["only-one"])
    with patch.object(mj, "_score_one", side_effect=[_fake_score(0.70)]):
        v = mj.score("task", "expected", "output", "skill")
    assert v.low_confidence is True
    assert len(v.individual_scores) == 1
