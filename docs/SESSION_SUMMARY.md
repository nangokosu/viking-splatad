# Last session summary (2026-11-02)

## Done
Test-suite robustness audit (three read-only `code-reviewer` reviews), then hardening on branch `pyvista-pose-viewer` (not pushed):
- `671eae6` new tests only: mutation-resistant transform tests (random rigid transforms, independent expected values; 79 of 82 mutants caught, the 3 survivors are equivalent), contract negative and boundary cases, gates that block, recipe hash and node sizing, split disjointness, `evaluate_run` refusals, evaluation boundaries, synthetic pipeline content.
- `1869c14` enforcement hardening:
  - `scripts/affected_tests.py` now exits 3 on any SUSPICIOUS finding (`--allow-suspicious` opts out, only after stating the reason), compares against the merge-base with `main`, and compares tests per function by AST.
  - Covering tests must be genuine (`scripts/covers_rules.py`); `tests/fixtures/transform_manifest.json` pins the registry.
  - `tests/conftest.py` enforces strict markers and a skip/deselect allowlist (`tests/fixtures/test_exclusions_allowlist.json`, regenerate with `scripts/write_exclusions_allowlist.py` after adding a gpu or real_data test).
  - The bash guard checks arguments and fails closed.
  - `pyproject.toml` is untouched (no image or cluster env rebuild).
- `b6eaa6b` six contract-validator fixes in `splatad_contract.py` (NaN point time, empty sweep, non-finite xyz, pose rotation orthonormality, 4-column sweep, truncated `poses.npz`); the 17 tests in `tests/contract/test_contract_suspected_bugs.py` were the spec.
- Full suite in the image before the commits: 1093 passed, 1 allowed skip (`test_traceability`), 8 deselected. Not re-run after splitting into three commits.

## Current state
- `pyvista-pose-viewer` is not merged or pushed. `viz/` must be synced once: `uv sync --project viz --locked`.
- The new pose-rotation tolerance (`sqrt(float32 eps)`) has not been run on a real dataset; `scripts/06_validate_contract.py` on the smoke and June datasets is the check.
- Traceability accepts a transform with at least one genuine marker; 46 extra markers on adjacent or `real_data`-gated tests (T4, T24, T27, T21, T28, T56, T58-T60, T62, T65, T66, T71, T72, T5) are tolerated.
- Strict markers rely on seeding the private `config._inicache` (pytest 9.1.1); a pytest upgrade could break it, and `tests/unit/test_conftest_strictness.py` would show it.
- Rules 5 and 6 (radar seeds train-only, GT never a seed or loss target) are only partly testable: `scripts/07_build_radar_seeds.py` and `radar_init/` do not exist on this branch.
- Not audited: viewer and scene tests, render and render-only job tests, Slurm wiring tests, run cleanup and tracking, overfitting, fiftyone helpers, writer, MCAP reader, GT scripts, block and shared-split tests.
- `real_data` tests (8, deselected) and `test_pipeline_real.py` (one 5 s chunk, 8-12 Hz band) never gate by default.
- The June run is still Scene A arm **B** only; arm R has not been run on the 90/10 split. Run-to-run noise is accepted (rule 10), so B vs R needs repeats per arm.
- Carried over: the T16 elevation-mapping bias is undecided; the viser client does not build on the nodes; known limits of the viewer (one viewer per run, timeout kills only the `uv` child, script 15 loads script 13's helpers by file path).

## Next steps
1. Run `scripts/06_validate_contract.py` on the smoke and June datasets; re-run the full suite in the image on the committed state.
2. Look at the smoke maps, then on approval redraw May and June (`scripts/13 runs/<run>`) and the overfitting curves.
3. Merge or push `pyvista-pose-viewer` (user's call).
4. Audit the un-audited third of the suite; decide whether to require every `covers` marker to be genuine.
5. June arm R on the same split and seed, then repeats of B and R; compare with `scripts/15 --run <B> --run <R>`.
6. Evaluation per PLAN §5 (June frame check, `05_extract_cuboids.py`, P1 T45–T47, P2 T56–T57).
