"""Tests for the guidance block loader."""

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
