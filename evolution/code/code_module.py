"""CodeModule — wrap tool source code as a DSPy module for GEPA evolution.

The full source text becomes the DSPy Predictor's instructions. GEPA can
rephrase, improve comments, add docstrings, rearrange error handling —
all at the text level. No function signatures or code structure is removed.
"""
import dspy


class CodeModule(dspy.Module):
    """Wrap source code text as a GEPA-optimizable module."""

    def __init__(self, source_text: str):
        super().__init__()
        self.predictor = dspy.Predict(
            signature="dummy_input: str -> dummy_output: str",
            instructions=source_text,
        )

    @property
    def code(self) -> str:
        """Return the evolved source code from the predictor instructions."""
        return self.predictor.signature.instructions

    def forward(self, dummy_input: str = "") -> dspy.Prediction:
        return self.predictor(dummy_input=dummy_input)
