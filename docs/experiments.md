# Experiments

Every run is one config in `configs/experiments/`. Its outputs go to `reports/experiments/<id>/` together with the config hash and the git SHA. The table below is filled in by `scripts/09_compare_runs.py`; never type numbers in by hand.

| id | session | method | radar seeds | radar loss | purpose |
|---|---|---|---|---|---|
| may_check | may_short_grass (forest window, chunk 1) | `splatad` | – | – | Pipeline readiness check. Pass = held-out RGB and lidar renders visibly show the forest. |
| june_B | june_tall_grass (forest window) | `splatad` | – | – | Baseline: lidar + camera. |
| june_R | june_tall_grass (same window, split and seed) | `splatad-radar` | yes | yes (binary V; continuous V is a stretch) | Radar init plus the radar geometric regularizer. |

## Results
_Not run yet._

The metrics are:
- **Camera:** PSNR, SSIM and LPIPS, both full frame and inside the trunk mask.
- **Lidar:** depth, Chamfer and ray-drop.
- **Rendered lidar point cloud:** occupancy grid, visualised against the trunk cuboids. The cuboids are used for evaluation only.
