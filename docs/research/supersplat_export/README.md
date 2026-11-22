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

These runs keep almost all of their Gaussians after the crop (kept counts from the logs), against 2,435,942 of 5,000,000 for the 20k May stock run (`crop_real_may.py` above), so their PLYs are about twice as large. Every run has the MCMC cap of 5,000,000 Gaussians (`mcmc_cap_max`, `docs/research/splatad_extension_points.md` 1.4), so these two bound the export time of any run; the larger total (184.16 s) is the base of the time cap in `scripts/slurm/splatad_run.sbatch`.

## Colour accuracy (not in this folder)

The colour diagnostic is `scripts/23_check_supersplat_colours.py`, with its output in `reports/supersplat_colour_check/`. Against the photo on test frame 1203, the median / p90 absolute colour error is 0.148 / 0.559 for the PLY degree-0 colour, 0.120 / 0.622 for the camera-direction bake, and 0.051 / 0.247 for SplatAD's own render. Per-Gaussian decoding therefore cannot reproduce SplatAD's blended decoding. The user chose (2026-11-21) to keep the bake as an approximate viewer product.
