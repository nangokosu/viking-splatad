# Last session summary (2026-10-25)

## Done
- **First SplatAD smoke run on Slurm: passes.**
  - Job 222568, run `may_chunks0_1_smoke-smoke-20261025-182824-b8104e1`, commit `b8104e1`, A6000.
  - Parking-lot window 52–67 s: 135 train + 15 test frames, 3000 iterations.
  - Train took 536 s, rendering 72 s.
  - Renders line up with the real frames for both train and held-out poses, with no geometry errors. PSNR is 20.05 train / 19.88 test. The surfaces are speckled because the model is undertrained.
  - The rendered ground-truth images match the raw extracted frames (53.7 dB).
- **Native cluster env built.**
  - Job 222558 built `splatad-a1e9255584ab` in about 35 min, and 227 tests passed on the node.
  - Fixes on the way:
    - jobs held when variables were passed with `--export=...,K=V`; they now go through `env K=V sbatch` (`e09cc6a`);
    - jobs that write fast to NFS home are SIGKILLed, so the env and run outputs now go to scratch and are copied back rate-limited (`0bb8ad1`);
    - tiny-cuda-nn failed to link with `-lcuda`; the stub libcuda is now on the link path (`1d44127`).
  - Lock files committed in `docker/cluster-locks/` (`81e2b1f`).
- **Render-step fix:** `ns-render --downscale-factor` asserts with viking-data, so the sbatch now omits it when the factor is 1 (`b8104e1`).
- **Project skills:**
  - `.claude/skills/building-splatad-cluster-env`
  - `.claude/skills/running-splatad-render-jobs` (submit, monitor and fetch only)
  - Both were tested with fresh agents (`97c3bbf`, `cf0e825`).
- **`scripts/slurm/compare_renders.py`:** per-split PSNR plus render-vs-real comparison sheets, with 5 tests (`97c3bbf`). No skill refers to it.
- **Tests:** full suite 233 passed, 3 skipped.

## Current state
- **Cluster:**
  - Env `splatad-a1e9255584ab` on scratch dir `expires-2026-Nov-07`, so rebuild after that date (`submit_run.py build-env`).
  - Stage root `~/viking-splatad` holds `data/may_chunks0_1_smoke`, `code/<sha>` and `runs/`.
- **Mac:**
  - The fetched run is in `runs/` (gitignored).
  - Preflight overlays are in `reports/may_chunks0_1_smoke/`.
  - The Docker image `viking-splatad:cu118` predates `c713a02`, so bind-mount `src/` for CPU checks.
- **Open issues:**
  - The viser web client doesn't build on the nodes (yarn timeout). Only the interactive viewer needs it.
  - The test split is every 10th frame interleaved with training frames, so it only tests interpolation.
  - Lidar renders were not checked.

## Next steps
1. Optionally, a 30k-iteration run of the same window, to confirm the speckle clears with training.
2. Full May Scene A with stock SplatAD, then June B and R (`docs/PLAN.md`).
3. Before evals, decide the held-out protocol (a held-out stretch of the drive vs interleaved frames).
4. Rebuild the Mac reference image from a clean worktree, in the background.
