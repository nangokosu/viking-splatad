# Last session summary (2026-11-22)

## Done
Branch `radar-init-loss-merge` (not pushed). One commit: "SuperSplat export: viewer frame and driven-path corridor crop" (the commit carrying this file), on top of 0dbf4f6.
- **The PLYs looked broken in SuperSplat; their content was fine.**
  - `docs/research/supersplat_export/render_ply_view.py` renders a PLY with the 3DGS image model from a train camera, and the old PLYs showed the scene that way.
  - The cause was how SuperSplat placed them. SuperSplat expects up = -y (it rotates 180° about z at load); the map frame has up = +z and is far from the origin. The T86 box spanned ±200 m, so a sparse background shell set the viewer's framing.
- **Fix, now the defaults of `scripts/22_export_supersplat_ply.py`:**
  - **T88 viewer frame:** `p' = M (p_map - o)`, with `M = [[1,0,0],[0,0,-1],[0,1,0]]` and `o` the mean train sensor position.
  - **T89 / T90:** the inverse for points, and directions into the viewer frame.
  - **T87 corridor crop:** width = p99.9 horizontal lidar range; z = the 0.001 / 99.999 percentiles of the corridor lidar.
  - **Sidecar:** `renders/supersplat_frame.json` is always written (frame, origin, rotation, crop, PLY size). Script 23 replays the crop from it and refuses a PLY without one.
  - **Old behaviour still available:** `--frame map`, `--crop-mode box`.
  - **Measurements:** in `docs/research/supersplat_export/README.md`, section "Viewer fix". Corridor widths compared by render: p95 left 3.9% holes, p99.9 0.73% with the same error as the box.
- **Re-exported:** May 30k radar init + loss (4,782,636 Gaussians, 255 MiB) and June 30k radar init + loss cam (4,811,023, 257 MiB). The old May stock 20k PLY was deleted (user's request, to save disk).
- **Job step (`splatad_run.sbatch`):**
  - The time cap now comes from the new logs: 191.08 s rounded up to 192 s, ×4.
  - Temp PLY/sidecar files are removed after a failed or killed export.
  - The skills and the glossary describe the viewer frame and the sidecar.

## Current state
- **The two PLYs are not yet checked in SuperSplat by the user.** The render check (train camera, viewer frame) matches the old box render: June median abs error vs the photo 0.114, May 0.145.
- **The job step has not run on the cluster yet.** Stage the code (`submit_run.py stage`) before the next job.
- **Old test edits need reasons in the PR:** the `affected_tests.py` audit still flags earlier branch commits (7dce3fb, fe84c70, 46765ea). This session's existing-test edits:
  - CLI flags added so old tests keep testing the map frame / box: `--frame map`, `--crop-mode box`.
  - `STUB_MEASURED_SECONDS` 185 -> 192, mirroring the sbatch.
  - Added `covers` markers.
- **Research scripts that need the old map-frame box PLY:** `corridor_crop_measure.py` and `corridor_width_render_sweep.py` (and `ply_sanity.py` / `crop_real_may.py` for the May stock run). Re-export with `--crop-mode box --frame map` before rerunning them.
- **Not in this commit:** the user's uncommitted `.claude/settings.json`, `reports/gaussian_occupancy/` and `scripts/21_compare_sessions.py`.

## Next steps
1. Open the two PLYs in SuperSplat. If they still open badly, say how (orientation, framing, holes).
2. Before the next cluster job: `submit_run.py stage`, then check that `job_info.txt` shows `supersplat_export: written` and that `renders/supersplat_frame.json` exists.
3. Export other runs when wanted: `VIKING_WRITABLE_DIR=runs/<run>/renders scripts/test_in_image.sh python scripts/22_export_supersplat_ply.py --run-dir runs/<run>`.
4. Run script 13 over the repeat campaign: compare the gap between arms with the spread within arms, June first.
5. Push the branch and open a PR to main (user's call), stating the reasons for the old test removals.
