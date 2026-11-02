# Last session summary (2026-11-01)

## Done
- Branch `intermittent-splits`: the test split is now the standard 3DGS hold-out instead of two contiguous blocks.
  - `342b1ad`: T71 `shared_interleaved_train_test_indices` holds out every 1/(1 − f)-th lidar sweep (neurad LINSPACE complement); a camera frame is test iff its nearest sweep is, so camera and lidar share the held-out instants. T72 `restrict_test_to_render_subset` renders an evenly spaced fraction r of those instants; the rest are guard. Session YAMLs: `split.method: shared_interleaved`, `train_fraction: 0.9`, `test_render_fraction: 0.1`. Wired through `submit_run.py`, the sbatch scripts, the dataparser, scripts/09 (split record) and scripts/12.
  - `852a745`: Slurm profile `smoke_a4500_scene_split` (A4500, the Scene A f = 0.9 and r = 0.1); the `smoke` / `smoke_a4500` profiles keep 0.8 / 0.2 because existing tests pin them. Glossary: the eval during training uses every held-out frame (T65).
  - Full suite in the image: 638 passed, 1 allowed skip; no existing test modified.
- All earlier runs (smoke, May A, June A, and the 80/20 smoke 226449 cancelled at the user's request) were cleaned from the cluster, the Mac and the ledger.
- Re-trained at `852a745`, all successful (`split_render_check: match`, `overfitting_plot: written`, `ADD_MISSING_POINTS: True`) and fetched to `runs/`:

  | Run | Job | Train (s) | Camera train/test/guard | Lidar train/test/guard |
  |---|---|---|---|---|
  | `may_chunks0_1_smoke-smoke_a4500_scene_split-20261101-205035-852a745` | 226469 | 827 | 135/2/13 | 135/2/13 |
  | `may_short_grass-may_scene_a-20261101-211146-852a745` | 226539 | 11809 | 2160/25/216 | 2171/25/216 |
  | `june_tall_grass-june_scene_a-20261101-211347-852a745` | 226540 | 11381 | 1938/22/193 | 1946/22/194 |

## Current state
- Branch `intermittent-splits` is not merged into `main` or pushed.
- The June run is Scene A arm **B** (stock SplatAD, no photometric loss). Arm R (radar seeds + opacity loss) has not been run on the new split.
- Run-to-run noise is accepted (CLAUDE.md rule 10), so B vs R needs repeats per arm.
- Carried over: the T16 elevation-mapping bias (per-ring median vs. far-field fit) is undecided; the viser client does not build on the nodes.

## Next steps
1. Review the May and June overfitting curves (`runs/<run>/overfitting/`) and held-out metrics.
2. Merge or push `intermittent-splits` (user's call).
3. June arm R on the same split and seed, then repeats of B and R to measure within-arm spread.
4. Evaluation per PLAN §5 (June frame check, `05_extract_cuboids.py`, P1 T45–T47, P2 T56–T57).
