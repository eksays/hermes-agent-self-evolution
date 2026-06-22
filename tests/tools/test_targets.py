"""Tests for the Phase 2 target tools constants."""

from evolution.tools.targets import TARGET_TOOLS, LABEL_OPEN, LABEL_CLOSE


def test_target_tools_is_the_confusable_cluster():
    assert TARGET_TOOLS == [
        "search_files",
        "read_file",
        "terminal",
        "web_search",
        "browser_navigate",
        "write_file",
    ]


def test_label_markers_are_distinct():
    assert LABEL_OPEN != LABEL_CLOSE
    assert "[[" in LABEL_OPEN
