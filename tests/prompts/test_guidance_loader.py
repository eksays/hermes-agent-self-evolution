"""Tests for the guidance block loader."""

import textwrap
from pathlib import Path

from evolution.prompts.targets import TARGET_GUIDANCE, LABEL_OPEN, LABEL_CLOSE, BLOCK_START, BLOCK_END


def test_target_guidance_count():
    assert len(TARGET_GUIDANCE) == 6


def test_target_guidance_all_strings():
    assert all(isinstance(t, str) and len(t) > 0 for t in TARGET_GUIDANCE)


def test_label_markers_are_distinct():
    assert LABEL_OPEN != LABEL_CLOSE
    assert len(LABEL_OPEN) == 2
    assert len(LABEL_CLOSE) == 2


def test_block_markers_are_distinct():
    assert BLOCK_START != BLOCK_END
    assert BLOCK_START != LABEL_OPEN
    assert BLOCK_END != LABEL_CLOSE


# ---------------------------------------------------------------------------
# Guidance loader helpers (Task 2)
# ---------------------------------------------------------------------------

def _write(tmp_path, name, content):
    f = Path(tmp_path) / name
    f.write_text(textwrap.dedent(content), encoding="utf-8")
    return f


def test_read_inline_guidance(tmp_path):
    from evolution.prompts.guidance_loader import read_guidance_from_source
    src = _write(tmp_path, "prompt_builder.py", '''
        MEMORY_GUIDANCE = (
            "Keep memory compact. "
            "Do not save temporary state."
        )
    ''')
    text = read_guidance_from_source(src, "MEMORY_GUIDANCE")
    assert text is not None
    assert "Keep memory compact" in text
    assert "Do not save temporary state" in text


def test_read_single_line_guidance(tmp_path):
    from evolution.prompts.guidance_loader import read_guidance_from_source
    src = _write(tmp_path, "prompt_builder.py", '''
        SKILLS_GUIDANCE = "After complex tasks, save the approach as a skill."
    ''')
    text = read_guidance_from_source(src, "SKILLS_GUIDANCE")
    assert text == "After complex tasks, save the approach as a skill."


def test_read_missing_guidance_returns_none(tmp_path):
    from evolution.prompts.guidance_loader import read_guidance_from_source
    src = _write(tmp_path, "prompt_builder.py", '''
        OTHER_CONSTANT = "something"
    ''')
    assert read_guidance_from_source(src, "MEMORY_GUIDANCE") is None
