"""MultiJudge — Tier 2 judge panel with agreement measurement.

Runs the existing LLMJudge under several models and aggregates with a median
(robust to outliers). Reports agreement so low-confidence scores can be flagged
for human review. Only used on top-3 finalists, so cost is bounded.
"""
import statistics
from dataclasses import dataclass, field

from evolution.core.config import EvolutionConfig
from evolution.core.fitness import LLMJudge


@dataclass
class PanelVerdict:
    median_score: float
    mean_score: float
    agreement: float
    individual_scores: list[float]
    judge_models: list[str]
    low_confidence: bool
    feedback: list[str] = field(default_factory=list)


class MultiJudge:
    def __init__(self, config: EvolutionConfig,
                 judge_models: list[str] | None = None,
                 agreement_threshold: float = 0.6):
        self.config = config
        # Distinct models only (contamination control).
        models = judge_models if judge_models is not None else list(
            getattr(config, "multi_judge_models", []) or []
        )
        self.judge_models = list(dict.fromkeys(models))  # dedupe, keep order
        if not self.judge_models:
            self.judge_models = [getattr(config, "eval_model", "openai/gpt-4o-mini")]
        self.agreement_threshold = agreement_threshold
        self._judge = LLMJudge(config)

    def _score_one(self, model: str, task_input: str, expected_behavior: str,
                   agent_output: str, skill_text: str):
        """Score with one model. Overridable / patchable in tests."""
        prev = self.config.eval_model
        self.config.eval_model = model
        try:
            return self._judge.score(task_input, expected_behavior,
                                     agent_output, skill_text)
        finally:
            self.config.eval_model = prev

    def score(self, task_input: str, expected_behavior: str,
              agent_output: str, skill_text: str) -> PanelVerdict:
        scores: list[float] = []
        feedback: list[str] = []
        for model in self.judge_models:
            fs = self._score_one(model, task_input, expected_behavior,
                                 agent_output, skill_text)
            scores.append(fs.composite)
            if getattr(fs, "feedback", ""):
                feedback.append(fs.feedback)

        median = statistics.median(scores)
        mean = statistics.mean(scores)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        agreement = max(0.0, min(1.0, 1 - stdev / 0.5))

        low_confidence = len(scores) < 2 or agreement < self.agreement_threshold

        return PanelVerdict(
            median_score=median,
            mean_score=mean,
            agreement=agreement,
            individual_scores=scores,
            judge_models=list(self.judge_models),
            low_confidence=low_confidence,
            feedback=feedback,
        )
