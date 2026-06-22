"""Tests for OverfitGuard — Tier 0 overfitting detection."""
import pytest
from evolution.core.overfit_guard import check_overfit, OverfitVerdict


def test_real_results_case_is_severe_overfit():
    # The actual RESULTS.md case: val 0.9256, holdout 0.439
    v = check_overfit(train_score=0.92, val_score=0.9256, holdout_score=0.439)
    assert v.is_overfit is True
    assert v.severity == "severe"
    assert round(v.val_holdout_gap, 3) == 0.487
    assert "overfit" in v.reason.lower()


def test_healthy_candidate_not_overfit():
    v = check_overfit(train_score=0.80, val_score=0.78, holdout_score=0.76)
    assert v.is_overfit is False
    assert v.severity == "none"
    assert v.trust_penalty < 0.16


def test_holdout_regression_vs_baseline_flags_overfit():
    # Small gap, but holdout is worse than baseline -> still rejected
    v = check_overfit(train_score=0.70, val_score=0.68, holdout_score=0.60,
                      baseline_holdout=0.65)
    assert v.is_overfit is True
    assert v.holdout_regression > 0.02


def test_severity_bands():
    # Gap <0.15 = none
    assert check_overfit(0.9, 0.90, 0.80).severity == "none"     # gap 0.10
    # 0.15 to <0.30 = mild
    assert check_overfit(0.9, 0.90, 0.70).severity == "mild"     # gap 0.20
    # 0.30 to <0.45 = moderate
    assert check_overfit(0.9, 0.90, 0.50).severity == "moderate" # gap 0.40
    # >=0.45 = severe
    assert check_overfit(0.9, 0.95, 0.40).severity == "severe"   # gap 0.55


def test_trust_penalty_monotonic():
    low = check_overfit(0.8, 0.80, 0.78).trust_penalty
    high = check_overfit(0.9, 0.95, 0.40).trust_penalty
    assert high > low
