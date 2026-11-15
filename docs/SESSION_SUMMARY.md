# Last session summary (2026-11-15)

## Done
Branch `radar-init-loss-merge` (not pushed): Neel's radar opacity loss merged with Dorsa's radar initialization, giving three variants per session (stock `splatad`, `splatad-radar-init`, `splatad-radar-init-lidar`).

- `7dce3fb` merge; `553d650`, `fe84c70` tests first; `ca83f2c` docs; `702a6b8`, `e83a866` implementation.
- `6d81e8e`, `6ba4709`, `a0ea119`: the two code reviews (Slurm wiring, radar modules) and their fixes.
- `1420416`: painting tries the K nearest train frames (`paint_frames_per_camera: 2`); footprint is half a cell (one cell wide).
- `476b757`: seed `features_rest` random like stock; seed opacity from stock `init_opacities`.
- `cf9a5f0`:
  - Profiles `smoke_a4500_scene_split_radar_init` and `_radar_init_loss` on `may_chunks0_1_smoke`, the stock `smoke_a4500_scene_split` settings plus `radar_variant`. The `smoke_a4500` prefix is needed for the A4500 `gres`.
  - Fix: `load_verified_anchors` accepts a dataset named by its folder (`may_chunks0_1_smoke`, extracted as `may_short_grass`).
- Full suite in the image: 1362 passed, 1 allowed skip.
- **May chunks 0-1 radar smokes, successful and fetched:**
  - Runs `…radar_init-20261115-142917-cf9a5f0` and `…radar_init_loss-20261115-143151-cf9a5f0`.
  - Same frames as the stock smoke (`membership_sha256` 415ff0704f00).
  - Final eval at step 2950:

    | Method | PSNR | Chamfer | Median chamfer | Rel. depth | Train time |
    |---|---|---|---|---|---|
    | Stock | 19.65 | 0.968 | 1.382 | 0.212 | 827 s |
    | Radar init | 21.14 | 0.942 | 0.867 | 0.084 | 832 s |
    | Radar init + loss | 20.52 | 0.885 | 1.161 | 0.075 | 1432 s |

  - Single runs, non-trunk scene: wiring check only, not a result.

## Current state
- **Radar artifacts (gitignored):**

  | Artifact | Anchors | Painted |
  |---|---|---|
  | May Scene A `may_scene_a_radar_init` | 359,387 | 0.717 |
  | June Scene A `june_scene_a_radar_init` | 201,471 | 0.618 |
  | Chunks 0-1 `smoke_a4500_scene_split_radar_init` | 23,961 | 0.755 |

  The chunks 0-1 clock fit is looser on its 15 s window: offset 55 ms, residual 32 ms.
- **Cluster workdisk at its ~100 GB per-user quota.** It counts the user's other projects (~45 GB). Being full silently killed four smoke jobs (no log error, 0-byte checkpoint).
  - Freed so far: `pip-cache` and the cluster copy of the stock chunks 0-1 smoke (the Mac copy is kept).
  - The two new runs then took ~4 GB; only ~4 GB is free now.
- Cluster login: key authorized; the passphrase is in Keychain (`UseKeychain`).
- **Risks:**
  - Every anchor becomes a Gaussian (200k-360k on Scene A).
  - The May loss acts on lidar batches only.
  - Run-to-run noise means repeats are needed.
- Neel's scripts 17/18 accept only B vs L on June.
- Carried over: `pyvista-pose-viewer` not merged; T16 elevation bias undecided.

## Next steps
1. Free workdisk space (user's call) before the six full Scene A runs.
2. With approval: the six full runs (`<month>_scene_a`, `_radar_init`, `_radar_init_loss`); gates and stage per session first.
3. Repeats per variant; evaluation with scripts 13 and 15.
4. Push the branch and open a PR to main (user's call); tell Dorsa and Neel which of their pieces changed.
