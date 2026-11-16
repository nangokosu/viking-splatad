# P2 estimator diagnostic (2026-11-16)

Why T81 replaced T57 as the binding P2 estimator (PLAN §5, `evaluation_caveats.md` §5).

- `seen_cell_diagnostic.py` compares, over the seen trunk GT cells of one run, T57's centre-sampled occupancy with the opacity of the Gaussians whose means lie in each cell.
- `seen_cell_diagnostic_output.json` is its output on `may_short_grass-may_scene_a-20261116-133928-17bb7a1`.

The factorized archive comes from `scripts/10_export_gaussians.py --factorized`, run in the reference image. It is about 0.6 GB per run, so it is not kept.
