"""Environment naming from the recipe files, node sizing on mixed nodes, and the gres exceptions.

The recipe-hash test uses a real throwaway git repository with real files (not the project's own
recipe list as oracle): the environment name must change when, and only when, a recipe file's
committed content changes.
"""

import math
import subprocess
from pathlib import Path
from typing import Dict

import pytest

from tests.unit.test_slurm_submit import load_submit_module

RECIPE_FILE_NAMES = ["recipe_a.txt", "recipe_b.txt", "recipe_c.txt"]
GIT_IDENTITY_ARGUMENTS = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]


def git_in(repo_dir: Path, *arguments: str) -> str:
    """Run git in ``repo_dir`` and return stdout (raises on failure).

    Args:
        repo_dir: Repository directory.
        arguments: git arguments after the identity options.

    Returns:
        Stripped stdout.
    """
    return subprocess.run(["git", *GIT_IDENTITY_ARGUMENTS, *arguments], cwd=repo_dir, check=True,
                          capture_output=True, text=True).stdout.strip()


def commit_files(repo_dir: Path, file_texts: Dict[str, str]) -> str:
    """Write the files, commit them and return the commit SHA.

    Args:
        repo_dir: Repository directory.
        file_texts: relative path -> new text.

    Returns:
        Full SHA of the new commit.
    """
    for relative_path, text in file_texts.items():
        (repo_dir / relative_path).write_text(text)
    git_in(repo_dir, "add", "-A")
    git_in(repo_dir, "commit", "-q", "-m", "change")
    return git_in(repo_dir, "rev-parse", "HEAD")


def test_environment_name_changes_iff_a_recipe_file_changes(tmp_path, monkeypatch):
    """Editing any one recipe file renames the env; editing a non-recipe file or re-committing does not."""
    submit_module = load_submit_module()
    git_in(tmp_path, "init", "-q")
    monkeypatch.setattr(submit_module, "PROJECT_ROOT", tmp_path)
    config = {"environment_recipe_files": RECIPE_FILE_NAMES}
    baseline_texts = {name: f"content of {name}\n" for name in RECIPE_FILE_NAMES}
    baseline_texts["unrelated.txt"] = "not part of the recipe\n"
    baseline_sha = commit_files(tmp_path, baseline_texts)
    baseline_name = submit_module.environment_name(config, baseline_sha)
    assert baseline_name.startswith("splatad-")
    # An unrelated file changing leaves the name alone.
    unrelated_sha = commit_files(tmp_path, {"unrelated.txt": "changed\n"})
    assert submit_module.environment_name(config, unrelated_sha) == baseline_name
    # Each recipe file in turn: one-byte change gives a name differing from baseline and from the others.
    names_seen = {baseline_name}
    for recipe_name in RECIPE_FILE_NAMES:
        changed_sha = commit_files(tmp_path, {recipe_name: baseline_texts[recipe_name] + "x"})
        changed_name = submit_module.environment_name(config, changed_sha)
        assert changed_name not in names_seen, recipe_name
        names_seen.add(changed_name)
        # Restore, so the next iteration changes exactly one file relative to baseline.
        restored_sha = commit_files(tmp_path, {recipe_name: baseline_texts[recipe_name]})
        assert submit_module.environment_name(config, restored_sha) == baseline_name


def test_environment_name_reads_the_committed_content_not_the_work_tree(tmp_path, monkeypatch):
    """An uncommitted edit of a recipe file does not change the name of the commit."""
    submit_module = load_submit_module()
    git_in(tmp_path, "init", "-q")
    monkeypatch.setattr(submit_module, "PROJECT_ROOT", tmp_path)
    config = {"environment_recipe_files": RECIPE_FILE_NAMES}
    commit_sha = commit_files(tmp_path, {name: name for name in RECIPE_FILE_NAMES})
    committed_name = submit_module.environment_name(config, commit_sha)
    (tmp_path / RECIPE_FILE_NAMES[0]).write_text("dirty edit")
    assert submit_module.environment_name(config, commit_sha) == committed_name


def test_recipe_hash_distinguishes_a_forged_file_boundary():
    """One file whose content embeds 'next-path NUL NUL content' must not collide with two files.

    Without a length prefix the streams of {"a": b"pq\\0\\0z"} and {"a": b"p", "q": b"z"} are both
    "a NUL NUL pq NUL NUL z"; the prefix makes them differ.
    """
    submit_module = load_submit_module()
    assert (submit_module.compute_recipe_hash({"a": b"pq\0\0z"})
            != submit_module.compute_recipe_hash({"a": b"p", "q": b"z"}))


def test_derive_node_resources_heterogeneous_same_type_fits_every_node():
    """Two A6000 nodes of different size: the request must fit the smaller on both axes."""
    submit_module = load_submit_module()
    # Node 1 has few CPUs but much RAM; node 2 has many CPUs but little RAM, so min per axis differs
    # from either whole node (a first-row or last-row or max implementation gets one axis wrong).
    sinfo_text = ("HOSTNAMES MEMORY CPUS GRES\n"
                  "n1 200000 16 gpu:rtx_a6000:1\n"
                  "n2 100000 32 gpu:rtx_a6000:1\n"
                  "n3 999999 64 gpu:rtx_a4500:1\n")
    mem_fraction = 0.5
    cpus, memory_gb = submit_module.derive_node_resources(sinfo_text, "rtx_a6000", mem_fraction)
    assert cpus == 16
    assert memory_gb == math.floor(mem_fraction * 100000 / 1024)


def test_derive_node_resources_ignores_other_gpu_types_even_if_smaller():
    """A smaller node of another GPU type never lowers the request."""
    submit_module = load_submit_module()
    sinfo_text = ("HOSTNAMES MEMORY CPUS GRES\n"
                  "n1 100000 20 gpu:rtx_a6000:1\n"
                  "n2 1000 1 gpu:rtx_a4500:1\n")
    assert submit_module.derive_node_resources(sinfo_text, "rtx_a6000", 1.0) == (20, math.floor(100000 / 1024))


def test_gres_overrides_exist_only_for_render_only_and_smoke_a4500_profiles():
    """Exactly render_only and the smoke_a4500* profiles override gres, and each override is sm_86."""
    submit_module = load_submit_module()
    config = submit_module.load_slurm_config()
    sm86_types = {"rtx_a6000", "rtx_a4500"}
    overriding_profiles = {name for name, settings in config["profiles"].items() if settings.get("gres")}
    assert overriding_profiles
    assert all(name.startswith("smoke_a4500") for name in overriding_profiles)
    assert "smoke_a4500" in overriding_profiles
    for gres_spec in [config["gres"], config["render_only"]["gres"],
                      *(config["profiles"][name]["gres"] for name in overriding_profiles)]:
        assert submit_module.gres_type_from_spec(gres_spec) in sm86_types
    # A non-overriding profile resolves to the default, an overriding one to its own.
    assert submit_module.resolve_job_gres(config, config["profiles"]["smoke"]) == config["gres"]
    assert submit_module.resolve_job_gres(config, config["render_only"]) == config["render_only"]["gres"]
