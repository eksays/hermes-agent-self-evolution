"""Tests for TrustReport — honest before/after verdict."""
from evolution.core.trust_report import build_trust_report, render_text
from evolution.core.overfit_guard import check_overfit


def test_results_case_is_not_trusted():
    overfit = check_overfit(0.92, 0.9256, 0.439)
    r = build_trust_report(
        artifact_name="github-code-review",
        train_score=0.92, val_score=0.9256, holdout_score=0.439,
        baseline_holdout=0.471, overfit=overfit, panel=None, benchmark=None, diff="",
    )
    assert r.verdict in ("REJECT", "CAUTION")
    assert r.verdict != "TRUSTED"
    assert any("overfit" in w.lower() for w in r.warnings)


def test_healthy_case_trusted():
    overfit = check_overfit(0.80, 0.78, 0.76)
    r = build_trust_report(
        artifact_name="demo", train_score=0.80, val_score=0.78, holdout_score=0.76,
        baseline_holdout=0.70, overfit=overfit, panel=None, benchmark=None, diff="",
    )
    assert r.verdict == "TRUSTED"
    assert r.trust_score >= 0.6


def test_text_render_shows_all_three_splits():
    overfit = check_overfit(0.80, 0.78, 0.76)
    r = build_trust_report(
        artifact_name="demo", train_score=0.80, val_score=0.78, holdout_score=0.76,
        baseline_holdout=None, overfit=overfit, panel=None, benchmark=None, diff="",
    )
    text = render_text(r)
    assert "train" in text.lower() and "val" in text.lower() and "holdout" in text.lower()
    assert "0.76" in text
