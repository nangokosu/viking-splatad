# CLAUDE.md — viking-splatad

Goal: render the Viking Hill forest with SplatAD (neurad-studio). Plan: `~/.claude/plans/start-a-new-claude-cosmic-cascade.md`.

## Rules (in addition to ~/CLAUDE.md, which still applies: heavy comments, descriptive names, no hardcoded values)
1. Every conversion/calibration function is decorated `@transform("T<n>", "<description>")` (see `src/viking_splatad/registry.py`).
2. Every registered transform has at least one unit test marked `@pytest.mark.covers("T<n>")`. `tests/test_traceability.py` enforces this statically.
3. Unit tests use synthetic known-answer input; expected values are computed in the test from first principles.
4. Converter output must satisfy the SplatAD input contract (`contract/splatad_contract.py`); never hand data to the GPU box without a passing contract report.
5. Radar never enters the SplatAD dataset (SplatAD asserts no radars); it is a sidecar for alignment diagnostics.
6. Ground-truth cuboids stay separate from SplatAD inputs (`data/gt/`), used only for occupancy evaluation.
7. SplatAD training/rendering needs CUDA, so it runs only on the CS Lab Slurm cluster (see "GPU training on Slurm" below). Everything else (tests, conversion, non-CUDA GPU work) runs on the Mac, which may use its own GPU (MPS).
8. Network fetches use an anonymous identifier (e.g. `viking-splatad/0.1`); no personal info in headers, URLs or bodies.
9. Run Python with the project venv: `uv run pytest`.
10. **Reproducibility (core goal):** every SplatAD run uses the image from `docker/Dockerfile.splatad` (run with Apptainer on Slurm, see `docker/README.md`). New dependencies go into the Dockerfile and `docker/constraints.txt`, never installed by hand on the cluster. Every upstream is pinned: base image by digest, git by full SHA, pip by constraints.

## GPU training on Slurm (every run)
Only for SplatAD (CUDA-only). Other GPU work uses the Mac's GPU.
Source: https://support.cs.toronto.edu/systems/slurmresource.html
1. **Mac first:** `uv run pytest` and `scripts/06_validate_contract.py` both pass.
2. **Log in:** `ssh comps0.cs.toronto.edu` (comps0–3). Submit jobs from there. Never compute there.
3. **Check nodes:** `slurm_report -g`, `sinfo -p gpunodes -O NodeHost,Memory,CPUs`. One GPU per node, so request all of its CPUs and most of its RAM. Read the sizes from `sinfo`. Never guess them.
4. **Stage data:** `ls -1 /scratch/scratch-space/ | egrep expires`. Use the directory that expires last. The job must finish before that date.
5. **Submit with `sbatch`:** always `--partition=gpunodes --gres=gpu:1 -c <C> --mem=<M>G -t <≤5-0>`. The defaults (1 core, 1 GB, 12 h) are too small. A limit over 5 days never schedules. Never use `--get-user-env`. Checkpoint often: jobs are killed at the time or memory limit.
6. **Interactive (`srun ... -t 60 --pty bash`) for short debugging only.** Exit right away. Claimed-but-idle resources still lower priority.
7. **Monitor:** `squeue -u $USER`, then `sacct` after the job ends. `scancel <id>` if it's stuck.
8. **Clean up:** copy outputs off scratch before it expires.

## Verification workflow (independent test agent)
- Transformation/calibration tests are run by the project agent `transform-test-runner` (`.claude/agents/`), which has its own context and is READ-ONLY (no Edit/Write; a PreToolUse hook, `.claude/hooks/readonly_bash_guard.py`, blocks any shell command that could write).
- After modifying anything under `src/`, `tests/`, `scripts/` or `configs/`, the main agent calls `transform-test-runner` with what changed (files and/or transform ids). The main agent owns all fixes; the test agent only reports.
- The agent runs only affected tests (`scripts/affected_tests.py`: changed modules + transitive importers + traceability) and the full suite only when asked or when `pyproject.toml`/`configs/`/`conftest.py` changed.
- **Tests are the spec; the burden is on the module.** When a test fails, the main agent fixes the module. An existing test may be modified only when the main agent is absolutely convinced the test itself is wrong (e.g. its expected value contradicts first-principles math or the documented SplatAD contract), and then the reason must be stated explicitly in the commit message and to the user. Never loosen a tolerance, drop an assertion, skip, or delete an existing test to get green. Adding new tests is always fine.
- The test agent audits this: `scripts/affected_tests.py` prints "existing tests modified" (git M/D status vs HEAD, removed assertion lines, changed tolerance lines) and the agent flags every hit as SUSPICIOUS. Subagents that build modules are told the same rule: add tests, never edit existing ones.
- The test agent itself never edits tests or modules; it reports and the main agent acts.
