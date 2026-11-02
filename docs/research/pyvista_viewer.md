# PyVista pose viewer (visual aid of the semantic-occupancy evaluation)

`scripts/13_eval_semantic_occupancy.py` draws its poses automatically after it has written its report.
`scripts/15_view_pose_pyvista.py` is the viewer; run it directly to look at a pose interactively or to put runs side by side.

```
uv sync --project viz --locked                      # once; PyVista lives in the isolated viz/ project
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id>          # report + pictures
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> --viz-radius-meters 15
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> --viz-classes tree_trunk
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> --no-viz # report only
uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<B> --run runs/<R> --pose 000529
uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<run_id> --output-dir <dir> --classes tree_trunk --view top --show-points
```

- **Three pooled maps per run** (`<eval root>/eval/viz/pooled_overlay.png`, `_recall.png`, `_precision.png`; no `--output-dir` opens a window per map). Pooled = the report's pooling: one occupied-voxel grid over all test sweeps and one scoring neighbourhood, so the caption's recall and precision are the report's voxel recall and precision for the drawn classes (a test checks TP/FN/FP against `evaluate_run`). `--pose 000529` draws one sweep instead (`<stem>_<map>.png`), an illustration that is not the report's numbers; `--show-points` needs `--pose`. Several `--run` folders give side-by-side panels with a linked camera; they are refused unless their `split.json` `membership_sha256` and the GT `cuboids.json` hash agree.
  - **Every map shows the whole rendered cloud** (all occupied voxels within the radius), at the same faint opacity (0.03, the lowest that still reads), so the three pictures show the cloud identically.
  - **overlay:** GT voxels grey (opaque-ish) under the whole render in light blue, no correct/incorrect split, so the GT always shows through.
  - **recall:** GT voxels: green = the render also occupies the voxel (hit, solid), grey = missed. Every other render voxel is faint green for context.
  - **precision:** render voxels: solid green = on a GT voxel of a selected class; faint red = not on GT, whether inside the metric's neighbourhood (a false positive) or outside it (not scored). No GT voxel is drawn.
  - Dark wireframes mark the scored cuboids. The caption gives the run, pose, number of returned points, point set, radius, classes and the map's own count (recall = hit / GT voxels; precision = green / render voxels).
- **Each run on its own returned test points.** The render voxels come from that run's rendered test-split lidar sweeps (`renders/test/lidar/<stem>.npz`, the stems of `split.json`): never train sweeps, never another run. The sweeps of one run are pooled (union of occupied voxels) exactly as in the report; `--pose` shows a single sweep alone.
- **Occupancy voxels, drawn within a radius.** The drawing is cut to `--radius-meters` (horizontal, around any pooled sweep's lidar; default `eval.viz_radius_meters`, 25 m, the user's choice 2026-11-02); the caption counts cover the whole scored region, so they equal the report even when the picture is cut. The counts (caption ratios, report) use only the voxels inside the metric's scoring neighbourhood; render voxels outside it (ground, grass, far air) are drawn faint but never counted (user's choice 2026-11-02: show the whole cloud in all maps), so the faint red in the precision map mixes the metric's false positives with unscored voxels; the caption gives the outside count.
- **Class filter.** `--classes tree_trunk,rock` (scripts/13: `--viz-classes`; config default `eval.viz_classes`, null = all scored classes) filters GT, hits, the neighbourhood, the render voxels and the wireframes together. Per-class cells are unioned; a voxel correct for one class and incorrect for another is drawn green.
- **Same definitions as the metric.** Scored boxes: `eval.detection.scored_region_cuboids`. Neighbourhood (each box grown horizontally by its class margin, restricted to the sweep's range and vertical FOV): `visible_neighbourhood_mask`, the function `evaluate_run` calls. Occupied voxels: T68 `points_to_map_voxels` + `occupied_voxel_grid`.
- **Point set:** ungated (T67: every valid ray with accumulation >= the threshold; the one the metric binds to, so the maps agree with the report) unless `--gated` (SplatAD's ray-drop-filtered `pred_points`); the caption names it. `--show-points` adds the real returns (grey) and rendered returns (red inside a scored cuboid, blue otherwise) to the overlay (`--all-points` skips the neighbourhood crop). `--view top` looks straight down.
- **Trigger:** after the JSON and markdown are on disk, script 13 runs the viewer as `uv run --project viz python scripts/15_...` from the repo root (timeout in `VIZ_TIMEOUT_SECONDS`). A failure prints a warning with the fix command, is recorded as `"viz": "failed: ..."` in `semantic_occupancy.json`, and never changes the metrics or the exit code.
- **Why a subprocess into `viz/`:** the root `pyproject.toml` is a recipe file (copied into `docker/Dockerfile.splatad`); adding PyVista there would change the cluster environment hash and force an image and cluster rebuild.
- GT cuboids and voxels are display and evaluation only (CLAUDE.md rule 6). Read-only on the run folders.
- Side-by-side pictures of two runs are made by calling script 15 with both runs; script 13 itself launches one picture set per run.
