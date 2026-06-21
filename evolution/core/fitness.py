"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import re

import dspy
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        lm = dspy.LM(self.config.eval_model)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def _words(text: str) -> set:
    """Lowercase tokenization stripping punctuation for overlap matching."""
    return set(re.findall(r'\w+', text.lower()))


def skill_fitness_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: Optional[str] = None,
    pred_trace=None,
) -> float:
    """DSPy-compatible metric function for GEPA skill optimization.

    Hybrid metric: uses keyword overlap + instruction-following heuristics.
    Fast, deterministic, and provides rich gradient for GEPA.
    """
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""
    task = getattr(example, "task_input", "") or ""
    skill_text = getattr(example, "skill_text", "") or ""

    if not agent_output.strip():
        return 0.0

    score = 0.0
    output_words = _words(agent_output)

    # 1. Non-empty base (0-0.15)
    score += 0.15

    # 2. Keyword overlap with expected behavior (0-0.4)
    expected_words = _words(expected)
    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score += 0.4 * overlap
    else:
        score += 0.15  # Partial if no expected behavior

    # 3. Task relevance (0-0.15)
    task_words = _words(task)
    if task_words:
        overlap = len(task_words & output_words) / len(task_words)
        score += 0.15 * overlap

    # 4. Procedure following (0-0.2)
    # Check if key action words from skill appear in output
    if skill_text.strip():
        skill_lower = skill_text.lower()
        # Extract numbered steps and bullet point keywords
        step_keywords = set()
        for match in re.finditer(r'(?:^|\n)\s*(?:\d+\.|[-*])\s+(.+?)(?:\n|$)', skill_lower):
            line = match.group(1).strip()
            # Take first 3 significant words from each step
            words = [w for w in re.findall(r'\b[a-z]{3,}\b', line)][:3]
            step_keywords.update(words)
        if step_keywords:
            overlap = len(step_keywords & output_words) / len(step_keywords)
            score += 0.2 * overlap
    else:
        score += 0.1  # Partial if no skill text

    # 5. Length normalization penalty (0-0.1)
    # Very short (<10 words) or very long (>500 words) responses get penalized
    word_count = len(output_words)
    if word_count < 5:
        score -= 0.15
    elif word_count < 10:
        score -= 0.05
    elif word_count > 500:
        score -= 0.1
    elif word_count > 300:
        score -= 0.05

    return max(0.0, min(1.0, score))


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
