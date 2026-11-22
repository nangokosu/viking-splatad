# SuperSplat export measurements (2026-11-21)

Measurements behind the SuperSplat PLY export (`scripts/22_export_supersplat_ply.py`, `src/viking_splatad/export/`). Each script reuses the repo's export functions, takes its inputs as CLI arguments (repo-relative defaults) and writes `<name>_output.json` next to itself. The run is `may_short_grass-may_scene_a-20261116-135432-17bb7a1` (fetched, in `runs/`).

The saved `*_output.json` files were regenerated on 2026-11-21 with the current scripts. To regenerate them, in this order:
```
uv run python docs/research/supersplat_export/crop_real_may.py
uv run python docs/research/supersplat_export/ply_sanity.py
VIKING_WRITABLE_DIR=docs/research/supersplat_export scripts/test_in_image.sh python docs/research/supersplat_export/time_bake.py
VIKING_WRITABLE_DIR=docs/research/supersplat_export scripts/test_in_image.sh python docs/research/supersplat_export/sh_oversampling_residual.py
```

Scripts that import torch or gsplat run in the reference image. The repo is mounted read-only there, so the output folder is mounted writable:
`VIKING_WRITABLE_DIR=docs/research/supersplat_export scripts/test_in_image.sh python docs/research/supersplat_export/<script>.py`

- `sh_oversampling_residual.py` (image) measures the held-out RMS residual of the degree-3 SH fit on random degree-4 colour functions, against the number of fit directions (factor x 16). It justifies `DIRECTION_OVERSAMPLING_FACTOR` in `spherical_harmonics.py`. Output: `sh_oversampling_residual_output.json`. Relative residual: factor 1: 0.297060, 2: 0.126339, 4: 0.125443, 8: 0.125390, 16 and above: 0.125387. The floor is the degree-4 energy the fit cannot represent; it is reached by 4x.
- `crop_real_may.py` (`uv run`) computes the observed-train-lidar crop box (T86) and how many Gaussians it keeps at the run's `mcmc_min_opacity`, plus a 0.1/99.9 percentile box for comparison only. Output: `crop_real_may_output.json`. The observed box keeps 2,435,942 of 5,000,000 Gaussians (48.7%); the percentile box keeps 2,245,491 (44.9%).
- `time_bake.py` (image) times `bake_gaussian_colours` on 100,000 seeded random Gaussians with the real run's decoder, one direction each. Output: `time_bake_output.json`. Timing is machine-dependent (this run: amd64 image emulated on a Mac, CPU only): median of 5 repeats after one warm-up: 0.72 s at chunk 4096, 0.32 s at chunk 65536.
- `ply_sanity.py` (`uv run`) reads `renders/supersplat.ply` back and reports the row count, finiteness and percentiles of degree-0 colour, opacity and scale. Output: `ply_sanity_output.json`. 2,435,942 rows, 14 fields, all finite; degree-0 colour red median 0.3085, opacity median 0.0541.

## Export logs of the 30k radar init + loss runs

Two saved stdout logs of `scripts/22_export_supersplat_ply.py` on the 30k presentation runs (code `d51a4d6`; amd64 image emulated on the Mac, CPU only, so slower than a native node):

- `export_log_may_short_grass-may_scene_a_radar_init_loss_30k-20261119-132048-d51a4d6.txt`
- `export_log_june_tall_grass-june_scene_a_radar_init_loss_cam_30k-20261119-133010-d51a4d6.txt`

Each shows the per-step seconds (`load_checkpoint`, `map_frame_geometry`, `crop`, `bake_colours_and_sh_fit`, `quaternions_and_write`), the kept and dropped counts, the PLY size and the peak RSS:

| Run | Total (s) | Kept of 5,000,000 | PLY size | Peak RSS |
|---|---|---|---|---|
| May | 184.16 | 4,801,958 of 5,000,000 | 256.45 MiB | 4468 MiB |
| June | 174.46 | 4,839,422 of 5,000,000 | 258.45 MiB | 4515 MiB |

These runs keep almost all of their Gaussians after the crop (kept counts from the logs), against 2,435,942 of 5,000,000 for the 20k May stock run (`crop_real_may.py` above), so their PLYs are about twice as large. Every run has the MCMC cap of 5,000,000 Gaussians (`mcmc_cap_max`, `docs/research/splatad_extension_points.md` 1.4), so these two bound the export time of any run; the time cap in `scripts/slurm/splatad_run.sbatch` now uses the viewer-fix logs below.

## Colour accuracy (not in this folder)

The colour diagnostic is `scripts/23_check_supersplat_colours.py`, with its output in `reports/supersplat_colour_check/`. Against the photo on test frame 1203, the median / p90 absolute colour error is 0.148 / 0.559 for the PLY degree-0 colour, 0.120 / 0.622 for the camera-direction bake, and 0.051 / 0.247 for SplatAD's own render. Per-Gaussian decoding therefore cannot reproduce SplatAD's blended decoding. The user chose (2026-11-21) to keep the bake as an approximate viewer product.

## Viewer fix (2026-11-22): corridor crop and SuperSplat viewer frame

The first PLYs looked broken in SuperSplat. They were not: `render_ply_view.py` (image) renders a PLY with the standard 3DGS image model, using gsplat's covariance and projection functions plus a CPU front-to-back compositor (gsplat's own needs CUDA), from the run's median train camera. Rendered from that camera, both box-cropped, map-frame PLYs showed the scene (`render_ply_view_{june,may}_output*`: median abs error vs the photo 0.113 June, 0.147 May). The cause was how SuperSplat placed them. SuperSplat rotates every PLY 180 degrees about z at load and its engine is y-up, so it expects up = -y; the map frame has up = +z and is far from the origin. On top of that, the T86 box spanned about ±200 m (the lidar's range), so a sparse shell of background Gaussians set the viewer's framing.

The fix is two transforms, now the defaults of script 22:
- **T88, viewer frame:** `p_viewer = M (p_map - o)` with `M = [[1,0,0],[0,0,-1],[0,1,0]]` and `o` the mean train sensor position. `supersplat_frame.json` records the frame, origin, rotation, crop and the PLY size.
- **T87, corridor crop:** a Gaussian is kept if its horizontal distance to the nearest train sensor position is at most the corridor width, and its z lies within the corridor's z bounds. The width is the p99.9 horizontal lidar range. The z bounds are the 0.001 / 99.999 percentiles of the lidar z inside the corridor.

How the constants were chosen:
- **First measurements:** `corridor_crop_measure.py` (`uv run`) gave the lidar range percentiles and the kept counts per width (`corridor_crop_measure_{june,may}_output.json`).
- **Width p99.9:** `corridor_width_render_sweep.py` (image) rendered the box-cropped May PLY masked to each width (`corridor_width_render_sweep_may_output.json/.png`). Holes in the train view were 3.9% at p95, 1.1% at p99 and 0.73% at p99.9, against 0.004% for the box. At p99.9 the error vs the photo is the same as the box, and the kept x/y extent is 153 x 112 m instead of 416 x 341 m. p95 cut the distant trees seen from the path (black holes).
- **z tail 0.001%:** `corridor_z_measure.py` (`uv run`, every sweep) measured the z bounds (`corridor_z_measure_{june,may}_output.json`). Raw min/max came out -76.0 / 72.5 m (May) and -1.5 / 43.6 m (June), set by stray returns. The 0.001% tails give -0.92 / 22.49 m and 0.15 / 21.78 m, which is above the ~22 m tree tops, and cut 3,208 / 3,073 Gaussians.

Two of these scripts need the old box-cropped, map-frame PLY as input: `corridor_crop_measure.py` and `corridor_width_render_sweep.py`. To rerun them, first export with `--crop-mode box --frame map --overwrite`. `corridor_z_measure.py` reads the current viewer-frame PLY through its sidecar.

Final exports with the defaults (`export_log_viewer_*.txt`, the base of the job step's time cap):

| Run | Total (s) | Kept of 5,000,000 | Corridor width | z bounds | PLY size |
|---|---|---|---|---|---|
| May 30k radar init + loss | 183.35 | 4,782,636 | 37.09 m | -0.92 .. 22.49 m | 255.42 MiB |
| June 30k radar init + loss cam | 191.08 | 4,811,023 | 32.22 m | 0.15 .. 21.78 m | 256.94 MiB |

`render_ply_view_{june,may}_viewer_output*` render these final viewer-frame PLYs from the same train cameras, moved into the viewer frame with the sidecar, so the test also covers the rotated quaternions. Median / p90 error vs the photo: June 0.114 / 0.390, May 0.145 / 0.480.

The section "Export logs of the 30k radar init + loss runs" above describes the earlier box, map-frame export. Its logs are kept as a record.
