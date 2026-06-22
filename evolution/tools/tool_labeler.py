"""ToolLabeler — Stage 2: label correct tool from real episodes (Success + LLM-judge).

Reuses the _parse_scoring_json helper from evolution.core.external_importers.
"""
import re

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.external_importers import _parse_scoring_json

_TOOL_NAME_RE = re.compile(r"\buse\s+([a-z_]+)\b", re.IGNORECASE)


class ToolLabeler:
    """Label correct_tool from ToolEpisode using a judge LLM."""

    class JudgeToolChoice(dspy.Signature):
        """Given a task and the tool the agent actually used in a real session,
        decide whether that tool was the OPTIMAL choice among the available tools.
        Return a JSON object:
        {"optimal": bool, "better_tool": str or null, "alt_tools": [str],
         "difficulty": "easy|medium|hard", "kind": "clear|confusable|no_tool"}."""
        task: str = dspy.InputField()
        used_tool: str = dspy.InputField()
        available_tools: str = dspy.InputField(desc="name: description, one per line")
        judgement: str = dspy.OutputField(desc="JSON object")

    def __init__(self, config: EvolutionConfig, model: str | None = None):
        self.config = config
        self.model = model or config.judge_model
        self.judge = dspy.ChainOfThought(self.JudgeToolChoice)

    def _judge_one(self, task: str, used_tool: str, overview: str) -> dict:
        """One judge call. Patchable in tests."""
        lm = dspy.LM(self.model)
        with dspy.context(lm=lm):
            result = self.judge(task=task, used_tool=used_tool, available_tools=overview)
        return _parse_scoring_json(result.judgement) or {}

    def _correction_tool(self, task: str) -> str | None:
        m = _TOOL_NAME_RE.search(task or "")
        return m.group(1).lower() if m else None

    def label(self, episodes: list, all_tools: list[tuple[str, str]],
              max_examples: int = 100,
              evolve_params: bool = False) -> list:
        from evolution.tools.tool_dataset import ToolSelectionExample  # avoid circular import
        """Label episodes and build ToolSelectionExamples.

        Args:
            episodes: ToolEpisode list from TrajectoryMiner.
            all_tools: List of (tool_name, description) pairs (all available tools).
            max_examples: Maximum examples to produce.
            evolve_params: If True, attach correct_params from the episode.

        Returns:
            List of ToolSelectionExample.
        """
        overview = "\n".join(f"{n}: {d}" for n, d in all_tools)
        tool_names = {n.lower() for n, _ in all_tools}

        # hard cases first (corrections/errors), then the rest
        episodes = sorted(
            episodes,
            key=lambda e: (e.had_user_correction or e.had_retry_or_error),
            reverse=True,
        )

        out: list[ToolSelectionExample] = []
        errors = 0
        for ep in episodes[: max_examples * 2]:
            if not ep.task:
                continue
            try:
                j = self._judge_one(ep.task, ep.used_tool, overview)
            except Exception:
                errors += 1
                continue
            if not j:
                errors += 1
                continue

            optimal = bool(j.get("optimal", False)) and ep.session_success
            better = (j.get("better_tool") or "").strip()
            alts = [a for a in (j.get("alt_tools") or []) if a]
            kind = j.get("kind", "clear")

            if optimal:
                correct = ep.used_tool
            elif better and better in tool_names:
                correct = better
                alts = [ep.used_tool] + alts
                kind = "confusable"
            else:
                correct = ep.used_tool

            # user correction beats the judge
            if ep.had_user_correction:
                ct = self._correction_tool(ep.task)
                if ct and ct in tool_names:
                    if ct != correct:
                        alts = [correct] + [a for a in alts if a != ct]
                        correct = ct
                    kind = "confusable"

            ex = ToolSelectionExample(
                task=ep.task, correct_tool=correct,
                alt_tools=list(dict.fromkeys(alts)),
                difficulty=j.get("difficulty", "medium"), kind=kind,
            )
            if evolve_params and optimal:
                ex.correct_params = dict(ep.used_params or {})
            out.append(ex)
            if len(out) >= max_examples:
                break
        return out
