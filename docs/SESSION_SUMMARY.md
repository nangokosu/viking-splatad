# Last session summary (2026-11-16)

## Done
Branch `radar-init-loss-merge` (not pushed).
- Fetched the three radar runs (May init + loss 250273, June init cam 250333, June init + loss cam 250413) and evaluated all six Scene A runs with script 13 (pictures in each run's `eval/viz/`). Single runs only. June (tall grass, the claim): trunks found 73 / 72 / 76 of 175 (stock / init / init + loss), trunk voxel IoU 0.162 / 0.160 / 0.164; rocks 3 / 3 / 3 of 21. May (expected neutral): trunks 106 / 92 / 102 of 189, IoU 0.188 / 0.172 / 0.186. No effect can be read without repeats, and the stock runs were on older code (`852a745`).
- `c74d6e4`: `submit_run.py clean --cluster-only <run> ... [--yes]` deletes only cluster run folders. It runs only when an rsync dry run shows the Mac copy is complete (macOS openrsync's 9-character itemize codes are handled; it refuses when in doubt), and it re-checks `squeue` in the same ssh call as the delete. The skill `cleaning-up-splatad-runs` documents it.
- `17bb7a1`: profile `june_scene_a_cam` (stock June with the camera loss; `june_scene_a` is lidar-only now), the baseline for the June `*_cam` radar arms.
- Cluster workdisk 61 GB -> 37 GB: removed `data/may_chunks0_1_smoke` (by hand; re-stage with `stage --session may_chunks0_1_smoke`) and the cluster copies of the 8 fetched runs (`--cluster-only`). Those runs still exist on the Mac.

## Current state
Repeat campaign: 5 runs per arm, 30 jobs, all on code `17bb7a1`, submitted 2026-11-16 ~13:40 UTC, round-robin. Only 2 A6000 nodes (gpunode2, gpunode3), so expect at least ~2.7 days (measured run times: ~3.2 h stock and init, ~6.5-6.9 h init + loss). Run ids are in `runs/submitted.tsv`.

| Arm (profile) | Jobs |
|---|---|
| May stock (`may_scene_a`) | 250781, 250807, 250813, 250819, 250825 |
| May radar init (`may_scene_a_radar_init`) | 250799, 250809, 250815, 250821, 250827 |
| May init + loss (`may_scene_a_radar_init_loss`) | 250805, 250811, 250817, 250823, 250829 |
| June stock cam (`june_scene_a_cam`) | 250798, 250808, 250814, 250820, 250826 |
| June radar init cam (`june_scene_a_radar_init_cam`) | 250800, 250810, 250816, 250822, 250828 |
| June init + loss cam (`june_scene_a_radar_init_loss_cam`) | 250806, 250812, 250818, 250824, 250830 |

- User's instruction for this campaign: after each successful fetch, delete that run from the cluster (`clean --cluster-only --yes <run>`); the Mac keeps the only copy. A detached watcher does fetch + cluster-only clean every 10 min (session scratchpad `campaign_watcher.py`, log `campaign_watcher.log`). It stops if the Mac sleeps or reboots. Then do it by hand from `submit_run.py runs`.
- Mac disk: 111 GB free before the campaign; 30 runs add ~75 GB.
- The user's uncommitted FiftyOne removal (plus `.claude/settings.json`) is still uncommitted; stash it around every gated submit (`git stash push`, then `git stash pop --index`).
- Radar seed caveats unchanged: no ground/height/lidar-proximity filter; 28% (May) and 38% (June) unpainted; radar-vs-odometry velocity residual ~15x the noise model.

## Next steps
1. Run `submit_run.py runs`. Fetch each finished campaign run and then clean it with `--cluster-only` (the watcher does this while it runs). Diagnose and clean failed runs.
2. When all 30 are in, run script 13 per session over the repeats. Compare the gap between arms with the spread within each arm, for trunks and rocks, June first.
3. If radar init alone still hurts May, check where the seeds sit relative to lidar surfaces and how many survive training.
4. Push the branch and open a PR to main (user's call); tell Dorsa and Neel what changed.
