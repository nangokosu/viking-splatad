# Last session summary (2026-11-15, night; runs submitted 2026-11-16 UTC)

## Done
Branch `radar-init-loss-merge` (not pushed). Earlier work of the day: Neel's radar opacity loss merged with Dorsa's radar init into three variants per session (stock `splatad`, `splatad-radar-init`, `splatad-radar-init-lidar`); time limits 14 h May / 16 h June (`a0abfe1`, `269c590`).

This session:
- Diagnosed the bad June radar-init renders: the June radar profiles were lidar-only (`train_lidar_only: true`), while the only stock June run (`june_scene_a-20261101-211347-852a745`) trained with the camera loss, so they were no valid pair (test PSNR 8.5 vs 15.9). The May pair is valid and close (PSNR 17.89 vs 18.26; chamfer 1.67 vs 0.98 is within the run-to-run noise until repeated).
- Deleted the lidar-only June radar-init run (249571) and cancelled and deleted the lidar-only June init + loss run (250274).
- `ec90a71`: profile `june_scene_a_radar_init_cam` (June core with the camera loss on, radar init). `b981e0c`: profile `june_scene_a_radar_init_loss_cam` (same, init + loss). Both read the existing lidar-only init artifact (201,471 anchors); the existing lidar-only June profiles and their tests are unchanged. Gates passed (1366 passed, 1 allowed skip; split identical to the reference).

## Current state
Jobs on the cluster (all with the camera loss on):

| Job | Run | Code | Limit |
|---|---|---|---|
| 250273 | `may_short_grass-may_scene_a_radar_init_loss-20261116-004300-269c590` | `269c590` | 14 h |
| 250333 | `june_tall_grass-june_scene_a_radar_init_cam-20261116-023330-ec90a71` | `ec90a71` | 16 h |
| 250413 | `june_tall_grass-june_scene_a_radar_init_loss_cam-20261116-025344-b981e0c` | `b981e0c` | 16 h |

All three are in `runs/submitted.tsv`. 250273 started ~19:43 EST on 2026-11-15 and ends roughly 8 h later; the two June jobs were submitted after midnight UTC.

Fetched and ready to evaluate: May stock (`may_scene_a`, 2026-11-01), May radar init (`may_scene_a_radar_init-...-1ed8131`), June stock (B+cam, 2026-11-01).

Other state:
- Radar seeds (May 359,387, June 201,471) have no ground, height, power or lidar-proximity filter, only a 42 m range cut, a static-Doppler test and a persistence test of 3-4 sweeps. 28% (May) and 38% (June) are unpainted grey. Radar-vs-odometry velocity residual RMS is ~14-16 (noise model expects ~1): possible calibration or timing error. These are suspects for any May gap, not confirmed.
- No stock June lidar-only baseline exists (needed only if the lidar-only design is wanted as its own arm).
- Uncommitted work of the user, not part of the runs: the FiftyOne removal (staged deletions of script 14, its doc and test; edits to `registry.py`, `viz/`, `tests/fixtures/transform_manifest.json`, `test_viewer_export.py`, `evaluation_caveats.md`). `submit_run.py gates` needs a clean tree, so it has to be stashed before each submit (`git stash push`, then `git stash pop --index`); a plain stash also takes your own uncommitted edits, so commit those first.
- Cluster workdisk: watch the ~100 GB per-user quota. Always run `submit_run.py stage --session <s>` after a new commit and before `submit`.

## Next steps
1. Run `submit_run.py runs`. When 250273, 250333 and 250413 end, fetch them; if one failed, diagnose from the Slurm log, then clean it.
2. Compare each radar arm with its stock run on the same objective (May: stock vs init vs init + loss; June: B+cam stock vs init vs init + loss).
3. Repeats per variant (chamfer varies 10-340% between identical runs); evaluation with scripts 13 and 15.
4. If the May gap persists, check seed quality locally: how many seeds survive training and where they sit relative to lidar surfaces.
5. Push the branch and open a PR to main (user's call); tell Dorsa and Neel which of their pieces changed.
