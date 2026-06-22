"""Tests for the tool description loader/writer."""

import textwrap
from pathlib import Path

from evolution.tools.tool_loader import read_description_from_source


def _write(tmp_path, name, content):
    """Helper: write a temporary Python source file."""
    f = tmp_path / name
    f.write_text(textwrap.dedent(content), encoding="utf-8")
    return f


def test_read_inline_description(tmp_path):
    """Extract a description that is an inline string literal."""
    src = _write(tmp_path, "file_tools.py", '''
        READ_FILE_SCHEMA = {
            "name": "read_file",
            "description": "Read a text file with line numbers.",
            "parameters": {"type": "object", "properties": {}},
        }
    ''')
    desc = read_description_from_source(src, "read_file")
    assert desc == "Read a text file with line numbers."


def test_read_constant_description(tmp_path):
    """Extract a description stored in a module-level variable."""
    src = _write(tmp_path, "terminal_tool.py", '''
        TERMINAL_TOOL_DESCRIPTION = "Run a shell command on the VM."
        TERMINAL_SCHEMA = {
            "name": "terminal",
            "description": TERMINAL_TOOL_DESCRIPTION,
            "parameters": {"type": "object", "properties": {}},
        }
    ''')
    desc = read_description_from_source(src, "terminal")
    assert desc == "Run a shell command on the VM."


def test_read_missing_tool_returns_none(tmp_path):
    """Requesting a tool not present in the file returns None."""
    src = _write(tmp_path, "file_tools.py", '''
        OTHER_SCHEMA = {"name": "other", "description": "x", "parameters": {}}
    ''')
    assert read_description_from_source(src, "read_file") is None
