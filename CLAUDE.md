# CLAUDE.md — viking-splatad

Goal: answer the team's research question (`docs/proposal/`): *does adding 4D radar to LiDAR+camera SplatAD improve tree-trunk reconstruction when tall grass occludes the trunks?*

Current scope: **Scene A**, a region with tree trunks, low grass in May and high grass in June. It covers 3 consecutive bag chunks per run (May 7–9, June 20–22), chosen by guess and check. June gets two runs:
- **B:** stock SplatAD with no photometric loss. The camera still colours the initial Gaussians.
- **R:** the same as B, plus radar seed Gaussians and a label-free radar opacity loss near the seeds. Radar is **never rendered**.

Extensions, each run only with the user's approval: the photometric loss (B+cam, R+cam) and the radar depth loss.

May Scene A is trained with stock SplatAD and checked on held-out metrics. **No May ↔ June mapping:** no ICP and no cross-run alignment.

Plan: `docs/PLAN.md`. SplatAD internals: `docs/research/splatad_extension_points.md`. Phase-1 findings and the T1–T33 table: `~/.claude/plans/start-a-new-claude-cosmic-cascade.md`.

## Rules (in addition to ~/CLAUDE.md, which still applies: heavy comments, descriptive names, no hardcoded values)
1. Every conversion/calibration function is decorated `@transform("T<n>", "<description>")` (see `src/viking_splatad/registry.py`).
2. Every registered transform has at least one unit test marked `@pytest.mark.covers("T<n>")`. `tests/test_traceability.py` enforces this statically.
3. Unit tests use synthetic known-answer input; expected values are computed in the test from first principles.
4. Converter output must satisfy the SplatAD input contract (`contract/splatad_contract.py`); never hand data to the GPU box without a passing contract report.
5. Radar is never a SplatAD *sensor*: `radars=()` stays empty because SplatAD asserts there are no radars. Radar is never rendered. It may enter SplatAD **only** as the `radar_seeds` metadata:
   - built offline in the map frame (`radar_init/`, `scripts/07_build_radar_seeds.py`);
   - transformed into the dataparser frame only by `VikingHillDataParser`, for the train split only;
   - gated by the contract.

   The seeds serve two purposes: extra Gaussians at init, and the anchors of the label-free radar losses (the opacity gate, and the optional depth rays). Raw radar is otherwise only a sidecar for alignment diagnostics.
6. Ground-truth cuboids stay separate from SplatAD inputs (`data/gt/`) and are used only for evaluation (occupancy, trunk mask). They never become seeds, seed filters or loss targets.
7. SplatAD training/rendering needs CUDA, so it runs only on the CS Lab Slurm cluster (see "GPU training on Slurm" below). Everything else (tests, conversion, non-CUDA GPU work) runs on the Mac, which may use its own GPU (MPS).
8. Network fetches use an anonymous identifier (e.g. `viking-splatad/0.1`); no personal info in headers, URLs or bodies.
9. Run Python with the project venv: `uv run pytest`. Set the venv up with `uv sync --extra dev`, because pytest is in the `dev` extra. Every import must be declared in `pyproject.toml`: a plain `uv sync` removes anything installed by hand.
10. **Reproducibility (core goal):** `docker/Dockerfile.splatad` is the single recipe and the source of every pin: git by full SHA, pip by `docker/constraints.txt`, toolchain versions. The CS Lab GPU nodes have no container runtime and block user namespaces (probed 2026-10-25), so every SplatAD run on Slurm uses the **native environment** that `scripts/slurm/build_env.sbatch` builds from that recipe (`scripts/slurm/submit_run.py build-env`). The environment is named by a hash of the committed recipe files, so a run always uses the environment its recipe describes. The Docker image is the reference build and is used on the Mac for CPU checks (`scripts/slurm/preflight_dataparser.py`). New dependencies go into the Dockerfile and `docker/constraints.txt` (plus `scripts/slurm/cluster_env.yaml` for conda-level tools), never installed by hand on the cluster.

## GPU training on Slurm (every run)
Only for SplatAD (CUDA-only). Other GPU work uses the Mac's GPU.
Source: https://support.cs.toronto.edu/systems/slurmresource.html
1. **Mac first:** `uv run pytest` and `scripts/06_validate_contract.py` both pass (`submit_run.py gates`), and the CPU preflight in the image shows sane lidar-on-image overlays.
2. **Log in:** `ssh comps0.cs.toronto.edu` (comps0–3). Submit jobs from there. Never compute there.
3. **Check nodes:** `slurm_report -g`, `sinfo -p gpunodes -O NodeHost,Memory,CPUs`. One GPU per node, so request all of its CPUs and most of its RAM. Read the sizes from `sinfo`. Never guess them.
4. **Stage data:** `submit_run.py stage --session <s>` copies data and code into `stage_root` (`configs/slurm.yaml`), `~/viking-splatad` in NFS home. Jobs only **read** home: compute jobs that write fast to NFS home are SIGKILLed with no error (debug jobs 222528–222535, 2026-10-25). So the environment and training outputs live on the scratch dir that expires last (`scripts/slurm/scratch_lib.sh`), and each job copies its results back to home rate-limited (`copy_back_bwlimit_kib`). A job must finish before its scratch dir expires; `submit_run.py` and the jobs check this. When the env's scratch dir expires, run `build-env` again.
5. **Submit with `sbatch`** (via `submit_run.py submit`): always `--partition=gpunodes --gres=gpu:rtx_a6000:1 -c <C> --mem=<M>G -t <≤5-0>`; the environment is compiled for sm_86 (A6000). The defaults (1 core, 1 GB, 12 h) are too small. A limit over 5 days never schedules. Never use `--get-user-env`, and never `--export=...,K=V`: on this cluster both make Slurm fetch the login environment, which fails, and the job is held (`user_env_retrieval_failed_requeued_held`). Pass job variables as `env K=V sbatch ...` (`submit_run.py` does this). Checkpoint often: jobs are killed at the time or memory limit.
6. **Interactive (`srun ... -t 60 --pty bash`) for short debugging only.** Exit right away. Claimed-but-idle resources still lower priority.
7. **Monitor:** `squeue -u $USER`, then `sacct` after the job ends. `scancel <id>` if it's stuck.
8. **Clean up:** `submit_run.py fetch <run_id>` copies the results the job already copied to home onto the Mac. Anything else on scratch is lost when the scratch dir expires.

## Session summary (read at start, overwrite at end)
- **Start of session:** the last session's summary is imported below. Treat it as the current state and verify file/flag names it mentions before relying on them.
- **End of session** (when the user wraps up, or after the last task of the session): **overwrite** `docs/SESSION_SUMMARY.md`; never append. It contains only:
  1. the date;
  2. what was done, with commit SHAs;
  3. the current state (data on disk, open issues);
  4. the next steps, in order.
  Keep it short, current-state only, with no history of earlier sessions (git has that). Commit it with the session's last commit.

@docs/SESSION_SUMMARY.md

## Verification workflow (independent test agent)
- Transformation/calibration tests are run by the project agent `transform-test-runner` (`.claude/agents/`), which has its own context and is READ-ONLY (no Edit/Write; a PreToolUse hook, `.claude/hooks/readonly_bash_guard.py`, blocks any shell command that could write).
- After modifying anything under `src/`, `tests/`, `scripts/` or `configs/`, the main agent calls `transform-test-runner` with what changed (files and/or transform ids). The main agent owns all fixes; the test agent only reports.
- The agent runs only affected tests (`scripts/affected_tests.py`: changed modules + transitive importers + traceability) and the full suite only when asked or when `pyproject.toml`/`configs/`/`conftest.py` changed.
- **Tests are the spec; the burden is on the module.** When a test fails, the main agent fixes the module. An existing test may be modified only when the main agent is absolutely convinced the test itself is wrong (e.g. its expected value contradicts first-principles math or the documented SplatAD contract), and then the reason must be stated explicitly in the commit message and to the user. Never loosen a tolerance, drop an assertion, skip, or delete an existing test to get green. Adding new tests is always fine.
- The test agent audits this: `scripts/affected_tests.py` prints "existing tests modified" (git M/D status vs HEAD, removed assertion lines, changed tolerance lines) and the agent flags every hit as SUSPICIOUS. Subagents that build modules are told the same rule: add tests, never edit existing ones.
- The test agent itself never edits tests or modules; it reports and the main agent acts.
