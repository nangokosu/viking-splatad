# CLAUDE.md — viking-splatad

Goal: answer the team's research question (`docs/proposal/`): *does adding 4D radar to LiDAR+camera SplatAD improve tree-trunk reconstruction when tall grass occludes the trunks?*

Current scope: **Scene A**, a region with tree trunks, low grass in May and high grass in June. It covers 3 consecutive bag chunks per run (May 7–9, June 20–22), chosen by guess and check. June gets two runs:
- **B:** stock SplatAD with no photometric loss. The camera still colours the initial Gaussians.
- **R:** the same as B, plus radar seed Gaussians and a label-free radar opacity loss near the seeds. Radar is **never rendered**.

Extensions, each run only with the user's approval: the photometric loss (B+cam, R+cam) and the radar depth loss.

May Scene A is trained with stock SplatAD and checked on held-out metrics. **No May ↔ June mapping:** no ICP and no cross-run alignment.

Plan: `docs/PLAN.md`. SplatAD internals: `docs/research/splatad_extension_points.md`. Phase-1 findings and the T1–T33 table: `docs/research/phase1_findings.md`.

`AGENTS.md` is a symlink to this file, so Claude Code and Codex read the same rules. Edit `CLAUDE.md` only; `tests/test_agent_instructions_sync.py` fails if the two ever differ. The skills live in `.claude/skills/`; `.agents/skills` is a symlink to it for Codex.

## General code rules (every language, every contributor and agent)
1. **Comments:** a docstring per function (purpose, params, return, in the language's doc-comment convention); inline step-by-step comments through the logic; the math or algorithm explained where used (formulas, invariants, complexity). Precise and concise: no restating obvious code, no filler.
2. **Names:** unambiguous and descriptive of the role. No single-letter or vague names (`i`, `s`, `tmp`, `x`), loop indices included: `row_index`, not `i`.
3. **Values:** every number is computed, measured or derived from config/data, never hardcoded or guessed. When a hardcoded value is truly unavoidable, a comment justifies it.
4. **Bash:** simple single or `&&`/`;`-chained commands rather than `for`/`while`/`if` constructs when an equivalent simpler form exists.
5. **Personal information:** never put an email, name, account ID, cluster username, password or token into repo files or outbound requests (headers, URLs, query strings, bodies). Each user's Slurm login stays on their own machine (see "GPU training on Slurm", step 2). `tests/test_no_personal_info.py` scans tracked files for the running user's own identifiers.
6. **Machine-specific settings:** never commit settings that only apply to one person's machine (absolute paths such as `additionalDirectories` entries, personal permission allowlist entries, local tool or editor config) unless the user explicitly asks. Keep them uncommitted or in a gitignored local file (e.g. `.claude/settings.local.json`). Before committing a settings file, read its diff and drop the machine-specific lines.

## Project rules
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
9. **Tests always run in the reference image**, never on the host Python, so that results are reproducible and no test is skipped for a missing package (the Mac venv skips the torch, tensorboard and neurad tests). Use `scripts/test_in_image.sh` (full suite) or `scripts/test_in_image.sh python scripts/affected_tests.py --run` (affected tests). It starts `viking-splatad:cu118` with the repo mounted read-only and hands the container the host's login, home, git and ssh config so `tests/test_no_personal_info.py` checks the real user. A skip other than a test's own "nothing to check" skip (today only `test_traceability.py`'s "0 transforms pending") is a defect to fix, not to accept. Other Python (conversion, scripts) uses the project venv: `uv run ...`. Set the venv up with `uv sync --locked --extra dev` (`.python-version` pins the Python minor and `uv.lock` the libraries, so everyone gets the same numpy/scipy/opencv), because pytest is in the `dev` extra. Every import must be declared in `pyproject.toml`: a plain `uv sync` removes anything installed by hand.
    **Extraction:** run `scripts/03_extract.py` with the session YAML's pinned `window.chunk_files`. Never pass `--chunks` and never reuse an old output directory: the camera/lidar parameters are derived from the first chunk processed and cached in `derived.json`, so a different chunk set silently changes the dataset.
10. **Reproducibility (core goal):** `docker/Dockerfile.splatad` is the single recipe and the source of every pin: git by full SHA, pip by `docker/constraints.txt`, toolchain versions. The CS Lab GPU nodes have no container runtime and block user namespaces (probed 2026-10-25), so every SplatAD run on Slurm uses the **native environment** that `scripts/slurm/build_env.sbatch` builds from that recipe (`scripts/slurm/submit_run.py build-env`). The environment is named by a hash of the committed recipe files, so a run always uses the environment its recipe describes. The Docker image is the reference build, and it is **always** what the Mac uses for the CPU preflight (`scripts/slurm/preflight_dataparser.py`, `src/` bind-mounted) and for the tests (rule 9), never the host Python. After any change to the recipe files, rebuild the image (see the `building-splatad-cluster-env` skill) so the reference build matches what the cluster builds. **SplatAD's run-to-run noise is accepted** (user's decision 2026-10-29): we do not change gsplat, tcnn or SplatAD kernels to make them deterministic. Measured on the smoke window (3000 steps), identical-seed runs agree to <1% on PSNR/SSIM/LPIPS but differ by 10–340% on chamfer and relative-depth metrics, on the same node as much as across nodes. So no B-versus-R (or any) comparison is read without repeats: compare the gap between arms to the spread within an arm. New dependencies go into the Dockerfile and `docker/constraints.txt` (plus `scripts/slurm/cluster_env.yaml` for conda-level tools), never installed by hand on the cluster.
11. **Eval is accompanied by its picture:** every semantic-occupancy evaluation (`scripts/13_eval_semantic_occupancy.py`) draws its pooled test sweeps as three voxel maps (overlay, recall, precision) with `scripts/15_view_pose_pyvista.py` by default (opt-out `--no-viz`; a failed drawing is reported and recorded, never silent). PyVista is declared in the isolated `viz/pyproject.toml` (the exception to rule 9's root-`pyproject.toml` declaration) and the script is launched as a subprocess, so the root `pyproject.toml` stays unchanged. A new evaluation metric gets a matching view in the same change. The viewer calls the evaluation's own functions (`src/` viz code imports `src/` eval code, never the reverse; script 13 only launches script 15), so the picture uses the metric's scored boxes and inside test. See `docs/research/pyvista_viewer.md`.

## GPU training on Slurm (every run)
Only for SplatAD (CUDA-only). Other GPU work uses the Mac's GPU.
Source: https://support.cs.toronto.edu/systems/slurmresource.html
1. **Mac first:** `scripts/test_in_image.sh` and `scripts/06_validate_contract.py` both pass (`submit_run.py gates`), and the CPU preflight in the image shows sane lidar-on-image overlays.
2. **Log in:** `ssh comps0.cs.toronto.edu` (comps0–3). Submit jobs from there. Never compute there. Each user logs in with their own CS account. The username lives only in that user's `~/.ssh/config` (`User`) or in the `VIKING_SLURM_USER` environment variable, and never in repo files. Passwords are never stored anywhere (no repo files, env files, scripts or `sshpass`): use an ssh key or a ControlMaster connection. The one-time setup is in the `running-splatad-render-jobs` skill.
3. **Check nodes:** `slurm_report -g`, `sinfo -p gpunodes -O NodeHost,Memory,CPUs`. One GPU per node, so request all of its CPUs and most of its RAM. Read the sizes from `sinfo`. Never guess them.
4. **Stage data:** `submit_run.py stage --session <s>` copies data and code into the stage root on the MSCAC workdisk: `<workdisk_root>/<user>/<stage_root>` (`configs/slurm.yaml`), currently `/w/20262/<user>/viking-splatad`. Code, data, the environment (`envs/`) and run outputs (`runs/<run_id>/`) all live there. The workdisk is persistent (no expiry) and as fast as scratch (same file servers). **Never put project files in home or scratch.** Home has a 10 GB quota (MSCAC), and a full home made job writes fail with no error (jobs 222534 and 222633, explained by CS Lab support on 2026-10-26). Scratch expires every 15 days.
5. **Submit with `sbatch`** (via `submit_run.py submit`): always `--partition=gpunodes --gres=gpu:rtx_a6000:1 -c <C> --mem=<M>G -t <≤5-0>`; the environment is compiled for sm_86 (A6000). Exceptions (user's choice 2026-10-27): render-only jobs (`render_only.gres`) and the `smoke_a4500` profile request `gpu:rtx_a4500:1`, also sm_86. A different GPU type comes only from a `gres` key in `configs/slurm.yaml`, only for an sm_86 GPU, and is sized from that type's own `sinfo` rows. The defaults (1 core, 1 GB, 12 h) are too small. A limit over 5 days never schedules. Never use `--get-user-env`, and never `--export=...,K=V`: on this cluster both make Slurm fetch the login environment, which fails, and the job is held (`user_env_retrieval_failed_requeued_held`). Pass job variables by exporting them inside the batch script, as CS Lab support recommends (2026-10-26): `submit_run.py` writes a per-job `job.sbatch` with `export K=V` lines that then runs the committed script. Checkpoint often: jobs are killed at the time or memory limit.
6. **Interactive (`srun ... -t 60 --pty bash`) for short debugging only.** Exit right away. Claimed-but-idle resources still lower priority.
7. **Monitor:** `squeue -u $USER`, then `sacct` after the job ends. `scancel <id>` if it's stuck.
8. **Clean up:** `submit_run.py fetch <run_id>` copies a run's results (the latest checkpoint only) from the workdisk onto the Mac. Delete cancelled, failed or unwanted runs with `submit_run.py clean [<run_id> ...]` (dry run first, then `--yes`; see the `cleaning-up-splatad-runs` skill), never by hand.

## Run tracking (every session start)
At the start of every session, before other work, run `uv run python scripts/slurm/submit_run.py runs`. It is read-only: it reconciles `runs/submitted.tsv`, the cluster run folders, your `squeue` and the fetched copies in `runs/`. For each run that needs attention it prints the status, its jobs (state, pending reason, Slurm's start estimate), the furthest `job_info.txt` milestone and the next action. Report the result to the user in a few lines, then act on each status:
- `active`: start one background wait on all active jobs (the loop in `running-splatad-render-jobs`, "Fetch"), since a wait from an earlier session is gone.
- `fetch`: fetch it in the background, without asking (standing instruction).
- `failed` / `stale-ledger`: read the Slurm log, diagnose and report, then clean it up (`cleaning-up-splatad-runs`) without asking (standing instruction, 2026-10-29).
- `resumable` / `plot-recovery`: diagnose and report. Resume or recover only with the user's approval.
- `done` runs are hidden (`--all` lists them).

If the cluster cannot be reached, say so and carry on. Never infer from a failed ssh that the queue is empty.

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
- **Agents without the `transform-test-runner` subagent (e.g. Codex):** run `scripts/test_in_image.sh python scripts/affected_tests.py --run` yourself, read its "existing tests modified" audit the same way, and follow the same rules (tests are the spec; add tests, never weaken existing ones).
