"""Read, list, and write system-prompt guidance blocks from hermes-agent.

Guidance blocks live in ``agent/prompt_builder.py`` as module-level string
constants (e.g. ``MEMORY_GUIDANCE = ("..." )``). We parse statically with
:py:mod:`ast` so hermes-agent code is never executed.  Reuses patterns from
Phase 2's ``tool_loader.py`` but adapted for assignment-from-tuple/string.
"""

import ast
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level AST helpers
# ---------------------------------------------------------------------------

def _string_constants(tree: ast.Module) -> dict:
    """Map module-level NAME -> str value for simple and parenthesized string assigns."""
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            value = _resolve_constant_string(node.value)
            if value is not None:
                consts[target.id] = value
    return consts


def _resolve_constant_string(node) -> Optional[str]:
    """Resolve an AST node to a string if it's a literal or tuple of literals."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Tuple):
        parts = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                parts.append(elt.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(part.value)
            else:
                return None
        return "".join(parts)
    return None


# ---------------------------------------------------------------------------
# Public API  (Task 2)
# ---------------------------------------------------------------------------

def read_guidance_from_source(source_path: Path, constant_name: str) -> Optional[str]:
    """Extract the value of a module-level string constant by name."""
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    consts = _string_constants(tree)
    return consts.get(constant_name, None)
