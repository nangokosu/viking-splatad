# Experiments

Every run is one config in `configs/experiments/`. Its outputs go to `reports/experiments/<id>/` together with the config hash and the git SHA. The table below is filled in by `scripts/09_compare_runs.py`; never type numbers in by hand.

Scene A is a region with tree trunks, low grass in May and high grass in June: 2 consecutive bag chunks per run, chosen by guess and check (plan §1). May and June are never aligned to each other.

## Scene A choice
Chosen 2026-10-19 with `scripts/02_scene_quicklook.py`; the contact sheets and track maps are in `reports/scene_a/`.

| run | chunks | duration / path | trunks within 20 m of track | what the camera shows |
|---|---|---|---|---|
| May | `short_grass__ros2_7`, `_8` | 84.3 s / 69.9 m, 83.2 s / 76.0 m | 45, 72 | short grass, bare birch and pine trunks fully visible, gravel path, rocks |
| June | `tall_grass__ros2_19`, `_20` | 69.2 s / 29.4 m, 70.8 s / 32.1 m | 20, 37 | knee-to-waist-high grass and flowers hiding the trunk bases; same birch stand and gravel path |

- **Where the runs overlap** (`reports/scene_a/may_short_grass_chunk7_track_map.png` shows all four tracks):
  - May 8 and June 20 drive the same gravel path through the birch stand, around x ∈ [−55, −20], y ≈ −50 in the label map.
  - May 7 runs parallel to June 19, about 5–8 m north.
- **June frame check (§1 step 5):** the June 19 and 20 tracks pass right beside the June rock cuboids, and those rocks appear in the camera at the matching times. So the June `/icp_odom` frame is consistent with the June label map, judged by eye. A quantitative check (GT lidar on the cuboids) is part of the June evaluation.
- **Rejected guesses:**
  - June 18: tall grass, but open meadow with few trunks.
  - May 19 and 20: about 70 m south of the June scene.

| id | session | method | radar seeds | radar loss | purpose |
|---|---|---|---|---|---|
| may_smoke | `data/intermediate/may_chunks0_1_smoke` (May chunks 0+1, already extracted) | `splatad` | – | – | Slurm/Docker smoke run only. |
| may_A | may_short_grass (Scene A chunks) | `splatad` | – | – | SplatAD out of the box on forest. Held-out PSNR/SSIM/LPIPS, lidar metrics, rendered point cloud. |
| june_B | june_tall_grass (Scene A chunks) | `splatad`, lidar-only | – | – | Core Model 1: lidar under high vegetation. |
| june_R | june_tall_grass (same chunks, split, seed and steps) | `splatad-radar`, lidar-only | yes | opacity loss | Core Model 2: radar init + radar opacity loss. |
| june_B_cam | as june_B | `splatad` (camera on) | – | – | Extension (approval needed): photometric loss added to B. |
| june_R_cam | as june_R | `splatad-radar` (camera on) | yes | opacity loss | Extension (approval needed): photometric loss added to R. |
| june_R_depth | as june_R | `splatad-radar`, lidar-only | yes | opacity + depth loss | Extension (approval needed): radar depth loss (T44). |

## Results
_Not run yet._

June metrics are computed at held-out June poses, against June cuboids (Tree_trunk, Bush_or_small_tree, Rock):
- **Object detection rate per class:** does the rendered lidar put points on each in-range cuboid?
- **Trunk precision and false-positive trunk clusters.**
- **Built-in lidar metrics:** depth, Chamfer and ray-drop.
- **Figures:** BEV and occupancy figures B | R | GT lidar.
- **Extension runs only, camera metrics:** PSNR/SSIM/LPIPS, full frame and inside the trunk-above-grass mask, as a "radar does not hurt" check (PLAN §4.3).
