"""Meta-tests: Claude Code and Codex read the same project rules and the same skills.

Claude Code reads ``CLAUDE.md`` and ``.claude/skills/``; Codex reads ``AGENTS.md`` and
``.agents/skills/``. Both Codex paths are symlinks to the Claude ones, so the two agents can never
drift apart. These tests fail if someone replaces a symlink with a copy that later diverges.
"""

from pathlib import Path

import pytest

# Repository root: this file lives in <root>/tests/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Claude Code's instruction file and skill directory (the single sources of truth).
CLAUDE_INSTRUCTIONS_PATH = PROJECT_ROOT / "CLAUDE.md"
CLAUDE_SKILLS_DIRECTORY = PROJECT_ROOT / ".claude" / "skills"

# Codex's instruction file and repo-scoped skill directory (official Codex skill location).
CODEX_INSTRUCTIONS_PATH = PROJECT_ROOT / "AGENTS.md"
CODEX_SKILLS_DIRECTORY = PROJECT_ROOT / ".agents" / "skills"


def skip_unless_full_checkout() -> None:
    """Skip outside a full repo checkout.

    The cluster code snapshot and the Docker image ship only src/, tests/, configs/ and scripts/
    (``STAGED_REPO_PATHS``), so the agent files are absent there by design.
    """
    if not CLAUDE_INSTRUCTIONS_PATH.is_file():
        pytest.skip("needs the full repo checkout (CLAUDE.md)")


def test_agents_md_has_the_same_content_as_claude_md():
    """AGENTS.md exists and is byte-identical to CLAUDE.md (it is a symlink to it)."""
    skip_unless_full_checkout()
    # Step 1: Codex must find the file at all.
    assert CODEX_INSTRUCTIONS_PATH.exists(), "AGENTS.md is missing: recreate with `ln -s CLAUDE.md AGENTS.md`"
    # Step 2: same bytes, whether it is still a symlink or was replaced by a copy.
    assert CODEX_INSTRUCTIONS_PATH.read_bytes() == CLAUDE_INSTRUCTIONS_PATH.read_bytes(), (
        "AGENTS.md differs from CLAUDE.md: edit CLAUDE.md only and keep AGENTS.md a symlink to it"
    )


def test_every_claude_skill_is_visible_to_codex_with_identical_content():
    """Every ``.claude/skills/<name>/SKILL.md`` has a byte-identical twin under ``.agents/skills/``."""
    skip_unless_full_checkout()
    # Step 1: discover the skills from the source directory rather than listing names by hand.
    claude_skill_files = sorted(CLAUDE_SKILLS_DIRECTORY.glob("*/SKILL.md"))
    assert claude_skill_files, "no skills found under .claude/skills/"
    # Step 2: each one must resolve under the Codex directory with the same bytes.
    for claude_skill_file in claude_skill_files:
        skill_relative_path = claude_skill_file.relative_to(CLAUDE_SKILLS_DIRECTORY)
        codex_skill_file = CODEX_SKILLS_DIRECTORY / skill_relative_path
        assert codex_skill_file.is_file(), (
            f"{skill_relative_path} is not visible to Codex: keep `.agents/skills` a symlink to `../.claude/skills`"
        )
        assert codex_skill_file.read_bytes() == claude_skill_file.read_bytes(), (
            f"{skill_relative_path} differs between .claude/skills and .agents/skills"
        )
