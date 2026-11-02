# PyVista pose viewer (visual aid of the semantic-occupancy evaluation)

`scripts/13_eval_semantic_occupancy.py` draws its poses automatically after it has written its report.
`scripts/15_view_pose_pyvista.py` is the viewer; run it directly to look at a pose interactively or to put runs side by side.

```
uv sync --project viz --locked                      # once; PyVista lives in the isolated viz/ project
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id>          # report + pictures
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> --viz-radius-meters 15
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> --no-viz # report only
uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<B> --run runs/<R> --pose 000529
uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<run_id> --output-dir <dir> --view top --show-points
```

- **One panel per run, one picture per pose** (`<eval root>/eval/viz/<stem>.png`; no `--output-dir` opens a window). Several `--run` folders give side-by-side panels with a linked camera; they are refused unless their `split.json` `membership_sha256` and the GT `cuboids.json` hash agree.
- **Occupancy voxels, within a radius.** Within `--radius-meters` (horizontal, around that sweep's lidar; default `eval.viz_radius_meters` in `configs/eval.yaml`, 25 m, the user's choice 2026-11-02) the panel shows the metric's confusion cells on the GT grid:
  green = occupied by the render and by the GT (true positive), red = rendered only (false positive), grey translucent = GT only (missed), dark wireframes = the scored cuboids. The caption gives the three counts and their IoU.
- **Same definitions as the metric.** Scored boxes: `eval.detection.scored_region_cuboids`. Neighbourhood (each box grown horizontally by its class margin, restricted to the sweep's range and vertical FOV): `visible_neighbourhood_mask`, the function `evaluate_run` calls. Occupied voxels: T68 `points_to_map_voxels` + `occupied_voxel_grid`. The report pools all test sweeps, so one pose's counts illustrate the report and are not its numbers.
- **Point set:** ungated (the binding one) unless `--gated`. `--show-points` adds the real returns (grey) and rendered returns (red inside a scored cuboid, blue otherwise; `--all-points` skips the neighbourhood crop). `--view top` looks straight down.
- **Trigger:** after the JSON and markdown are on disk, script 13 runs the viewer as `uv run --project viz python scripts/15_...` from the repo root (timeout in `VIZ_TIMEOUT_SECONDS`). A failure prints a warning with the fix command, is recorded as `"viz": "failed: ..."` in `semantic_occupancy.json`, and never changes the metrics or the exit code.
- **Why a subprocess into `viz/`:** the root `pyproject.toml` is a recipe file (copied into `docker/Dockerfile.splatad`); adding PyVista there would change the cluster environment hash and force an image and cluster rebuild.
- GT cuboids and voxels are display and evaluation only (CLAUDE.md rule 6). Read-only on the run folders.
- Side-by-side pictures of two runs are made by calling script 15 with both runs; script 13 itself launches one picture set per run.
