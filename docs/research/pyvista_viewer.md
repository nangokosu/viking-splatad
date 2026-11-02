# PyVista pose viewer (visual aid of the semantic-occupancy evaluation)

`scripts/13_eval_semantic_occupancy.py` draws its poses automatically after it has written its report.
`scripts/15_view_pose_pyvista.py` is the viewer; run it directly to look at a pose interactively or to put runs side by side.

```
uv sync --project viz --locked                      # once; PyVista lives in the isolated viz/ project
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id>          # report + pictures
uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> --no-viz # report only
uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<B> --run runs/<R> --pose 000529
uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<run_id> --output-dir <dir> --all-points
```

- **One panel per run, one picture per pose** (`<eval root>/eval/viz/<stem>.png`; no `--output-dir` opens a window). Several `--run` folders give side-by-side panels with a linked camera; they are refused unless their `split.json` `membership_sha256` and the GT `cuboids.json` hash agree.
- **Content (map frame):** real returns grey, rendered returns red inside a scored GT cuboid and blue in its halo, scored cuboids as green wireframes. The title gives the point set (ungated, the binding one, unless `--gated`), the inside count and the total rendered.
- **Same numbers as the metric.** Scored boxes come from `eval.detection.scored_region_cuboids`, the same call `evaluate_run` makes; the inside test is `points_in_any_cuboid`; the threshold is the one the evaluation used. Only points in each box grown horizontally by its class margin (the T46 neighbourhood) are drawn, so the view is framed on the trunks; `--all-points` draws the whole scan.
- **Trigger:** after the JSON and markdown are on disk, script 13 runs the viewer as `uv run --project viz python scripts/15_...` from the repo root (timeout in `VIZ_TIMEOUT_SECONDS`). A failure prints a warning with the fix command, is recorded as `"viz": "failed: ..."` in `semantic_occupancy.json`, and never changes the metrics or the exit code.
- **Why a subprocess into `viz/`:** the root `pyproject.toml` is a recipe file (copied into `docker/Dockerfile.splatad`); adding PyVista there would change the cluster environment hash and force an image and cluster rebuild.
- GT cuboids are display and evaluation only (CLAUDE.md rule 6). Read-only on the run folders.
- Side-by-side pictures of two runs are made by calling script 15 with both runs; script 13 itself launches one picture set per run.
