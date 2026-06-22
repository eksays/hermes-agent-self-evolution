"""TrustReport — combine all validation signals into one honest verdict."""
from dataclasses import dataclass, field

from evolution.core.overfit_guard import OverfitVerdict


@dataclass
class TrustReport:
    artifact_name: str
    train_score: float
    val_score: float
    holdout_score: float
    baseline_holdout: float | None
    overfit: OverfitVerdict
    panel: object | None
    benchmark: object | None
    diff: str
    trust_score: float
    verdict: str
    warnings: list[str] = field(default_factory=list)


def build_trust_report(artifact_name, train_score, val_score, holdout_score,
                       baseline_holdout, overfit, panel, benchmark, diff,
                       trust_score_min=0.6):
    """Build a TrustReport from all validation signals."""
    warnings = []

    agreement = getattr(panel, "agreement", 1.0) if panel else 1.0
    bench_ok = True
    if benchmark is not None:
        bench_ok = getattr(benchmark, "passed", True)
        if getattr(benchmark, "skipped", False):
            warnings.append("Benchmark gate was skipped — regression not verified.")

    trust_score = (
        holdout_score
        * (1 - overfit.trust_penalty)
        * (agreement if panel else 1.0)
        * (1.0 if bench_ok else 0.5)
    )
    trust_score = max(0.0, min(1.0, trust_score))

    force_caution = False
    if overfit.is_overfit:
        warnings.append(f"Overfit detected ({overfit.severity}): {overfit.reason}")
        if overfit.severity in ("moderate", "severe"):
            force_caution = True
    if panel is not None and getattr(panel, "low_confidence", False):
        warnings.append("Judge panel had low agreement — review manually.")
        force_caution = True
    if benchmark is not None and not bench_ok and not getattr(benchmark, "skipped", False):
        warnings.append("Benchmark regression detected.")
        force_caution = True

    if trust_score >= trust_score_min and not force_caution:
        verdict = "TRUSTED"
    elif trust_score >= 0.4 or force_caution:
        verdict = "CAUTION"
    else:
        verdict = "REJECT"

    if overfit.severity == "severe" and trust_score < 0.4:
        verdict = "REJECT"

    return TrustReport(
        artifact_name=artifact_name, train_score=train_score, val_score=val_score,
        holdout_score=holdout_score, baseline_holdout=baseline_holdout,
        overfit=overfit, panel=panel, benchmark=benchmark, diff=diff,
        trust_score=trust_score, verdict=verdict, warnings=warnings,
    )


def render_text(report: TrustReport) -> str:
    """Render the trust report as a human-readable string."""
    icon = {"TRUSTED": "✅", "CAUTION": "⚠️", "REJECT": "❌"}.get(report.verdict, "?")
    lines = [
        f"=== Trust Report: {report.artifact_name} ===",
        f"Verdict: {report.verdict} {icon}  (trust_score={report.trust_score:.3f})",
        "",
        f"  Train   : {report.train_score:.3f}",
        f"  Val     : {report.val_score:.3f}",
        f"  Holdout : {report.holdout_score:.3f}"
        + (f"  (baseline {report.baseline_holdout:.3f})" if report.baseline_holdout is not None else ""),
        f"  Val-Holdout gap: {report.overfit.val_holdout_gap:.3f} ({report.overfit.severity})",
    ]
    if report.panel is not None:
        lines.append(f"  Judge agreement: {getattr(report.panel, 'agreement', 0):.3f}")
    if report.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in report.warnings:
            lines.append(f"    - {w}")
    return "\n".join(lines)


def render_html(report: TrustReport) -> str:
    """Render the trust report as a simple HTML snippet."""
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.3f}</td></tr>"
        for k, v in [("Train", report.train_score), ("Val", report.val_score),
                     ("Holdout", report.holdout_score)]
    )
    warn = "".join(f"<li>{w}</li>" for w in report.warnings)
    return (
        f"<h2>Trust Report: {report.artifact_name}</h2>"
        f"<p><b>{report.verdict}</b> (trust_score={report.trust_score:.3f})</p>"
        f"<table border=1>{rows}</table>"
        f"<ul>{warn}</ul>"
    )
