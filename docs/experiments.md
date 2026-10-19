# Experiments

Every run is one config in `configs/experiments/`. Its outputs go to `reports/experiments/<id>/` together with the config hash and the git SHA. The table below is filled in by `scripts/09_compare_runs.py`; never type numbers in by hand.

Scene A is a region with tree trunks, low grass in May and high grass in June: 2 consecutive bag chunks per run, chosen by guess and check (plan §1). May and June are never aligned to each other.

## Scene A choice
_Not chosen yet._ For each run, record the chunk indices and offsets, plus the contact sheet that shows the grass height and the trunks.

| id | session | method | radar seeds | radar loss | purpose |
|---|---|---|---|---|---|
| may_smoke | may_short_grass (chunk 1, on disk) | `splatad` | – | – | Slurm/Docker smoke run only. |
| may_A | may_short_grass (Scene A chunks) | `splatad` | – | – | SplatAD out of the box on forest. Held-out PSNR/SSIM/LPIPS, lidar metrics, rendered point cloud. |
| june_B | june_tall_grass (Scene A chunks) | `splatad` | – | – | Model 1: lidar + camera under high vegetation. |
| june_R | june_tall_grass (same chunks, split, seed and steps) | `splatad-radar` | yes | opacity loss (depth loss optional, later) | Model 2: radar init + radar opacity loss. |

## Results
_Not run yet._

June metrics are computed at held-out June poses, against June cuboids (Tree_trunk, Bush_or_small_tree, Rock):
- **Object detection rate per class:** does the rendered lidar put points on each in-range cuboid?
- **Trunk precision and false-positive trunk clusters.**
- **Built-in lidar metrics:** depth, Chamfer and ray-drop.
- **Camera metrics:** only if `camera_loss: on` for both runs.
- **Figures:** BEV and occupancy figures B | R | GT lidar.
