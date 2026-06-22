"""Integration test: OverfitGuard + TrustReport wired together."""
from evolution.core.overfit_guard import check_overfit
from evolution.core.trust_report import build_trust_report, render_text


def test_pipeline_helper_builds_report_for_overfit_case():
    # Simulates what evolve_skill does after holdout eval.
    overfit = check_overfit(0.92, 0.9256, 0.439, baseline_holdout=0.471)
    report = build_trust_report(
        artifact_name="github-code-review",
        train_score=0.92, val_score=0.9256, holdout_score=0.439,
        baseline_holdout=0.471, overfit=overfit, panel=None, benchmark=None,
        diff="(diff)",
    )
    assert report.verdict != "TRUSTED"
    rendered = render_text(report)
    assert "github-code-review" in rendered
