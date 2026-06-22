"""OverfitGuard — Tier 0 (free) overfitting detection.

Pure arithmetic, no API calls. Flags candidates whose validation success did
not transfer to the holdout set, or that regressed vs the baseline on holdout.
"""
from dataclasses import dataclass


@dataclass
class OverfitVerdict:
    is_overfit: bool
    val_holdout_gap: float
    severity: str            # "none" | "mild" | "moderate" | "severe"
    holdout_regression: float
    reason: str
    trust_penalty: float     # 0.0-1.0


def _severity(gap: float) -> str:
    if gap < 0.15:
        return "none"
    if gap < 0.30:
        return "mild"
    if gap < 0.45:
        return "moderate"
    return "severe"


def check_overfit(
    train_score: float,
    val_score: float,
    holdout_score: float,
    baseline_holdout: float | None = None,
    max_gap: float = 0.15,
    regression_tolerance: float = 0.02,
) -> OverfitVerdict:
    gap = val_score - holdout_score
    holdout_regression = (
        (baseline_holdout - holdout_score) if baseline_holdout is not None else 0.0
    )

    gap_overfit = gap > max_gap
    regression_overfit = holdout_regression > regression_tolerance
    is_overfit = gap_overfit or regression_overfit

    severity = _severity(gap)
    trust_penalty = min(1.0, max(0.0, gap))

    reasons = []
    if gap_overfit:
        reasons.append(
            f"Validation {val_score:.3f} but holdout {holdout_score:.3f} "
            f"(gap {gap:.3f} > {max_gap}) — likely overfit to validation set."
        )
    if regression_overfit:
        reasons.append(
            f"Holdout {holdout_score:.3f} is worse than baseline "
            f"{baseline_holdout:.3f} (regression {holdout_regression:.3f})."
        )
    if not reasons:
        reasons.append(
            f"Healthy: val {val_score:.3f} / holdout {holdout_score:.3f} "
            f"(gap {gap:.3f} within {max_gap})."
        )

    return OverfitVerdict(
        is_overfit=is_overfit,
        val_holdout_gap=gap,
        severity=severity,
        holdout_regression=holdout_regression,
        reason=" ".join(reasons),
        trust_penalty=trust_penalty,
    )
