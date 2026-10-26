"""The cluster code snapshot and the image must hold every repo file the test suite reads.

build_env.sbatch runs the suite on the staged snapshot (STAGED_REPO_PATHS) and the Dockerfile runs
it in the image; job 222810 failed because runs/RUN_FOLDER_GLOSSARY.md was in neither.
"""

import re

from tests.unit.test_slurm_submit import PROJECT_ROOT, load_submit_module

# Files outside the always-shipped trees (src, tests, configs, scripts) that tests read by path.
FILES_READ_BY_TESTS = ("runs/RUN_FOLDER_GLOSSARY.md", ".claude/hooks/readonly_bash_guard.py")


def test_every_file_read_by_tests_is_staged_copied_and_not_docker_ignored():
    """Each such file is in STAGED_REPO_PATHS, COPYed by the Dockerfile and whitelisted in .dockerignore."""
    staged_paths = load_submit_module().STAGED_REPO_PATHS
    dockerfile_text = (PROJECT_ROOT / "docker" / "Dockerfile.splatad").read_text()
    copied_sources = re.findall(r"^COPY\s+(\S+)\s", dockerfile_text, flags=re.MULTILINE)
    dockerignore_lines = (PROJECT_ROOT / ".dockerignore").read_text().splitlines()
    for relative_path in FILES_READ_BY_TESTS:
        assert relative_path in staged_paths
        assert relative_path in copied_sources
        assert f"!{relative_path}" in dockerignore_lines
        assert (PROJECT_ROOT / relative_path).is_file()
