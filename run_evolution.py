"""Runner — sets up DSPy with 9router, then invokes evolution phases.

Usage:
    python run_evolution.py [phase]

    phase: 1|skill, 2|tool, 3|guidance, 4|param, or "all" (default)
"""
import os
import sys

# ── 1. Set 9router credentials BEFORE any imports ────────────────────────────
API_BASE = "http://localhost:20128/v1"
API_KEY = "sk-2783c61019b4473f-vnfvuc-1859b2fb"

os.environ["OPENAI_API_BASE"] = API_BASE
os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["LITELLM_LOG"] = "ERROR"

# Force ascii/stripped output for Rich to avoid Windows cp1252 emoji crash
os.environ["TERM"] = "xterm-256color"
os.environ["COLORTERM"] = "truecolor"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["FORCE_COLOR"] = "0"
os.environ["NO_COLOR"] = "1"
os.environ["RICH_FORCE_EMOJI"] = "0"

HERMES_REPO = "E:/File/Project AI Agent/Hermes Agent/hermes-agent"
EVAL_MODEL = "openai/cc/claude-haiku-4-5-20251001"
OPTIMIZER_MODEL = "openai/cc/claude-sonnet-4-6"


def _no_emoji(text: str) -> str:
    """Strip Unicode emoji/checkmarks that crash Windows cp1252."""
    return text.replace("✓", "OK").replace("✗", "X")


def _patch_rich_console():
    """Monkeypatch rich.console.Console.print to strip unsupported Unicode."""
    import rich.console
    _orig_print = rich.console.Console.print

    def _safe_print(self, *args, **kwargs):
        safe_args = tuple(
            _no_emoji(str(a)) for a in args
        )
        try:
            return _orig_print(self, *safe_args, **kwargs)
        except UnicodeEncodeError:
            # Last resort: force plain text with no styles
            kwargs.pop("style", None)
            kwargs.pop("highlight", None)
            for a in safe_args:
                sys.stdout.write(str(a).replace("\n", "\n") + "\n")
            sys.stdout.flush()
            return

    rich.console.Console.print = _safe_print


def run_phase1():
    """Optimize github-code-review skill."""
    from evolution.skills.evolve_skill import evolve

    evolve(
        skill_name="github-code-review",
        iterations=3,
        eval_source="synthetic",
        optimizer_model=OPTIMIZER_MODEL,
        eval_model=EVAL_MODEL,
        hermes_repo=HERMES_REPO,
        dry_run=False,
    )


def run_phase2():
    """Evolve tool descriptions."""
    from evolution.tools.evolve_tool_descriptions import evolve_tools

    return evolve_tools(
        hermes_repo=HERMES_REPO,
        iterations=3,
        optimizer_model=OPTIMIZER_MODEL,
        eval_model=EVAL_MODEL,
        dry_run=False,
        write_back=False,
    )


def run_phase4():
    """Evolve parameter descriptions."""
    from evolution.tools.evolve_tool_descriptions import evolve_tools

    return evolve_tools(
        hermes_repo=HERMES_REPO,
        iterations=3,
        optimizer_model=OPTIMIZER_MODEL,
        eval_model=EVAL_MODEL,
        dry_run=False,
        write_back=False,
        evolve_params=True,
    )


def run_phase3():
    """Evolve system prompt guidance blocks."""
    from evolution.prompts.evolve_guidance import evolve_guidance

    return evolve_guidance(
        hermes_repo=HERMES_REPO,
        iterations=3,
        optimizer_model=OPTIMIZER_MODEL,
        eval_model=EVAL_MODEL,
        dry_run=False,
        write_back=False,
    )


if __name__ == "__main__":
    _patch_rich_console()

    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    if phase in ("1", "skill", "all"):
        print("=" * 60)
        print("PHASE 1: SKILL EVOLUTION (github-code-review)")
        print("=" * 60)
        run_phase1()
        print("[OK] Phase 1 complete")

    if phase in ("2", "tool", "all"):
        print("=" * 60)
        print("PHASE 2: TOOL DESCRIPTION EVOLUTION")
        print("=" * 60)
        run_phase2()
        print("[OK] Phase 2 complete")

    if phase in ("3", "guidance", "all"):
        print("=" * 60)
        print("PHASE 3: GUIDANCE EVOLUTION")
        print("=" * 60)
        run_phase3()
        print("[OK] Phase 3 complete")

    if phase in ("4", "param", "all"):
        print("=" * 60)
        print("PHASE 4: PARAMETER DESCRIPTION EVOLUTION")
        print("=" * 60)
        run_phase4()
        print("[OK] Phase 4 complete")

    print("=" * 60)
    print(f"[OK] {'ALL' if phase == 'all' else phase.upper()} PHASE(S) COMPLETE")
    print("=" * 60)
