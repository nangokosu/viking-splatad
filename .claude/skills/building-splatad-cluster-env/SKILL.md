---
name: building-splatad-cluster-env
description: Use when the SplatAD native environment on the CS Lab Slurm cluster is missing, expired or stale - submit_run.py says the env is not built or will not outlive the job, a recipe file changed (Dockerfile.splatad, constraints.txt, cluster_env.yaml, build_env.sbatch, pyproject.toml), a scratch dir expired, a build job died or failed to compile/link, or the Mac reference Docker image needs rebuilding.
---

# Building the SplatAD cluster environment

## Overview
CS Lab GPU nodes have **no container runtime** and block user namespaces, so SplatAD runs in a native env that `scripts/slurm/build_env.sbatch` replays from `docker/Dockerfile.splatad`. The env is named `splatad-<12-hex hash of the recipe files>`, lives on scratch, and is recorded by a marker in home. Never install anything by hand on the cluster.

## Rebuild or not?

| Change | Rebuild env? |
|---|---|
| `src/`, `tests/`, `configs/` (except recipe list), data, profiles | **No** — jobs put `code/<sha>/src` first on `PYTHONPATH` |
| Any file in `environment_recipe_files` (`configs/slurm.yaml`) | **Yes** — new hash, new env |
| Env's scratch dir expires (dirs last ≤15 days; `submit` refuses if the env would expire mid-job) | **Yes** |
| Docker image on the Mac | Only for the Mac reference/CPU preflight; cluster never uses it |

## Run (Mac, repo root)
```bash
git status                     # recipe changes must be COMMITTED (build stages HEAD via git archive)
VIKING_SLURM_USER=<user> uv run python scripts/slurm/submit_run.py build-env
```
Prints the job id and log path `~/viking-splatad/envs/logs/splatad-<hash>-<job>.out`. Re-running prints `environment already built` when the marker is valid; an expired marker with the same hash is rebuilt automatically in the newest scratch dir. The build's pytest runs HEAD's `src/`, so a broken commit fails the build.

**Monitor:** `submit_run.py status <job>`, and `ssh <user>@comps0.cs.toronto.edu "grep -nE 'Successfully|passed|ERROR|Traceback|complete' <log> | tail"`.

**Duration:** ~35 min on an A6000 node (job 222558: 17:28→18:03 UTC) plus queue. Order: micromamba env → torch → tiny-cuda-nn (long compile) → neurad-studio/gsplat/pandaset wheels → ~300-package install (silent up to ~10 min; not stuck) → splatad kernels → repo + pytest pin → viser client (yarn errors here) → checks. Silence >20 min with the job RUNNING: check the node (`srun --jobid=<job> --overlap -n1 top -bn1 | head`) before cancelling.

**Done when** the log ends with `== environment complete: ...` after `GPU import check OK` and a pytest summary, and `~/viking-splatad/envs/splatad-<hash>.marker` exists. Then:
```bash
scp '<user>@comps0.cs.toronto.edu:viking-splatad/envs/splatad-<hash>-locks/*' docker/cluster-locks/
```
(quote the glob — zsh expands it locally otherwise), check the log's `pip drift` block (expected: only the `pytest` final-step pin and `packaging` from conda), grep the locks for the username/home path, commit locks + a note in `docker/README.md`. Locks are records, not recipe inputs: committing them does not change the hash.

## Symptom → cause → fix (all seen on this cluster)

| Symptom | Cause | Fix |
|---|---|---|
| Job PENDING `user_env_retrieval_failed_requeued_held` | `sbatch --export=...,K=V` or `--get-user-env` | Pass vars as `env K=V sbatch ...` (submit_run.py does). `scancel` the held job |
| Job dies in ~15 s, sacct FAILED/KILLED, no error in log | Fast writes to NFS home are SIGKILLed | Heavy writes to scratch only (`scratch_lib.sh`); copy back with `rsync --bwlimit` |
| `cannot find -lcuda` linking tiny-cuda-nn | No driver lib at link time | `LIBRARY_PATH=$ENV_PREFIX/lib/stubs` (link-time only, never in activate script) |
| pip `ResolutionImpossible` on pytest | Global pin vs neurad `[dev]` pytest==7.1.2 | Keep pytest only as `# final-step-pin:` in constraints.txt |
| viser `ESOCKETTIMEDOUT` / `tsc: not found` | yarn downloads time out on nodes | Harmless: build continues; only the interactive viewer needs it; runs use `--vis tensorboard` |
| `Error unknown MAMBA_EXE ... must be mamba or micromamba` early in log | Versioned binary name | Harmless; build continues |
| `No space left on device` during build | Scratch is shared and was 99% full (41 GB free, 2026-10-25); env ~15-20 GB | `srun ... df -h /scratch/scratch-space/expires-*`; delete your old `viking-splatad-<user>` trees in older dirs |
| `ls /scratch` fails on comps0 | Scratch is mounted only on compute nodes | Inspect scratch from inside a job (`srun --jobid=<id> --overlap -n1 ...`) |

Kernels compile for the node's GPU (`nvidia-smi compute_cap`); jobs request `gpu:rtx_a6000:1` (sm_86). Do not switch GPU type without rebuilding.

## Mac reference image
```bash
docker build --platform linux/amd64 -f docker/Dockerfile.splatad --build-arg CUDA_ARCHITECTURES=86 -t viking-splatad:cu118 .
```
Build from a clean worktree, in the background (hours under emulation if `constraints.txt` or the patcher changed; minutes if only `src/`, `tests/`, `scripts/`, `configs/`). Record the image id and commit in `docker/README.md`. Until rebuilt, bind-mount `src/` for CPU preflights.

## Red flags
- "I'll just `pip install` it on the node" → add it to the Dockerfile + constraints.txt, rebuild.
- "Write the env to home, it's persistent" → the job gets killed.
- "Recompile because I changed the dataparser" → no rebuild needed.
