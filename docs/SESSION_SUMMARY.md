# Last session summary (2026-11-02)

## Done
- Git: the PR merging `intermittent-splits` into `main` was accepted on GitHub (`a1d161f`). Local `main` fast-forwarded to it and the local `intermittent-splits` branch deleted (`git branch -d`, merged). `origin/intermittent-splits` was left alone.
- Branch `pyvista-pose-viewer` (from `a1d161f`, not pushed), commit `1303108`: a PyVista per-pose viewer that always accompanies the semantic-occupancy evaluation (CLAUDE.md rule 11, `docs/research/pyvista_viewer.md`).
  - `scripts/15_view_pose_pyvista.py`: per pose and run, a map-frame panel with real returns, rendered returns (red inside a scored GT cuboid, blue in its halo), cuboid wireframes, cropped to the T46 neighbourhood (`--all-points` for everything). Several `--run` folders give linked side-by-side panels, refused unless `membership_sha256` and the GT cuboids hash agree.
  - `scripts/13` launches it after the report is written, as a subprocess into the isolated `viz/` project (`uv run --project viz`), `--no-viz` to skip. A failure is recorded as `"viz"` in `semantic_occupancy.json` and warned, and never changes the metrics or exit code. PyVista is only in `viz/pyproject.toml`/`viz/uv.lock`, so the root `pyproject.toml` (a recipe file) is untouched: no image or cluster env rebuild.
  - `eval.detection.scored_region_cuboids` is the single definition of the scored boxes, used by both scripts; `viz/pyvista_scene.py` holds the pure numpy helpers.
  - Tests are all new (no existing test modified): synthetic, real smoke run + May GT (`real_data`, run with `-m real_data`), the launch with a fake runner, and the script-15 logic. Full suite in the image: 657 passed, 1 allowed skip (`test_traceability`), 6 deselected (3 of them these real_data tests, run explicitly and passing).
  - Checked end to end on the smoke run: the report is written, then one PNG per test sweep appears in `runs/<run>/eval/viz/`.

## Current state
- `pyvista-pose-viewer` is not merged or pushed. `viz/` must be synced once: `uv sync --project viz --locked`.
- `real_data` tests skip on a clone without the fetched smoke run (`runs/`) and `data/gt/may_short_grass` (both gitignored).
- Known limits of the viewer (from the whole-branch review, accepted): script 13 launches one viewer per run, so no automatic B-vs-R figure (call script 15 with both `--run` folders); the viewer timeout kills only the `uv` child, not a hung grandchild; script 15 loads scripts 13's helpers by file path (they could move into `src/eval`); only the repo root is stripped from the recorded `viz` text.
- The June run is still Scene A arm **B** only; arm R has not been run on the 90/10 split. Run-to-run noise is accepted (rule 10), so B vs R needs repeats per arm.
- Carried over: the T16 elevation-mapping bias (per-ring median vs. far-field fit) is undecided; the viser client does not build on the nodes.

## Next steps
1. Look at the May and June pictures (`scripts/13 runs/<run>`) and the overfitting curves (`runs/<run>/overfitting/`).
2. Merge or push `pyvista-pose-viewer` (user's call).
3. June arm R on the same split and seed, then repeats of B and R to measure within-arm spread; compare B and R with `scripts/15 --run <B> --run <R>`.
4. Evaluation per PLAN §5 (June frame check, `05_extract_cuboids.py`, P1 T45–T47, P2 T56–T57).
