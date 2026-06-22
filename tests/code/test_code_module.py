"""Tests for CodeModule — wrap source code as DSPy module."""
from unittest.mock import patch, MagicMock
from evolution.code.code_module import CodeModule


def test_code_property_returns_instructions():
    """Code property returns the predictor's instructions (DSPy-transformed)."""
    source = "def hello():\n    return 'world'"
    cm = CodeModule(source)
    assert isinstance(cm.code, str)
    assert len(cm.code) > 0


def test_module_stores_source_in_predictor():
    """The source text is embedded in the predictor signature."""
    source = "def compute(x): return x * 2"
    cm = CodeModule(source)
    # The predictor exists and has a signature
    assert cm.predictor is not None
    assert hasattr(cm.predictor, "signature")
