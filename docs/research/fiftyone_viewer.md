# FiftyOne viewer (read-only)

For a simpler per-pose look at the occupancy evaluation (no database), see `pyvista_viewer.md`.

Browse a fetched run's renders next to the real data, with the GT cuboids, in FiftyOne.

```
uv sync --project viz --locked
uv run --project viz python scripts/14_build_fiftyone_dataset.py runs/<run_id> --session <session> --launch
```

- One group per rendered test lidar sweep. Slices: `lidar` (3D scene: rendered cloud + real cloud + GT cuboids, all in the **map frame**), `render_rgb`, `gt_rgb` (nearest camera frame in time). Per-scan metrics from `lidar_metrics.json` are sample fields.
- GT cuboids are display labels only (CLAUDE.md rule 6). T69: quaternion -> FiftyOne intrinsic-XYZ Euler. T70: sweep points -> map frame.
- Own environment in `viz/` (own `uv.lock`): FiftyOne needs `opencv-python-headless<5`, which would change the root lock's opencv pin. Never add it to the root `pyproject.toml`.
- Local only (bundled MongoDB); cluster nodes cannot run it. May and June are separate datasets (no cross-run alignment).
- Not shown: Gaussians (`gaussians.npz`, ~5M), radar seeds (not built yet), depth (not rendered), cuboid projection onto images (camera pose convention unconfirmed; FiftyOne's default projector also ignores lens distortion).
- Alignment check (May smoke run): share of real-lidar points inside any GT box is 25.8% at the true pose and 17.9% / 12.4% when shifted 3 m / 6 m along x.
- FiftyOne is a viewer only: with SplatAD's run-to-run noise it cannot support B-vs-R conclusions without repeats.
