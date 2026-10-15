# CLAUDE.md — viking-splatad

Goal: render the Viking Hill forest with SplatAD (neurad-studio). Plan: `~/.claude/plans/start-a-new-claude-cosmic-cascade.md`.

## Rules (in addition to ~/CLAUDE.md, which still applies: heavy comments, descriptive names, no hardcoded values)
1. Every conversion/calibration function is decorated `@transform("T<n>", "<description>")` (see `src/viking_splatad/registry.py`).
2. Every registered transform has at least one unit test marked `@pytest.mark.covers("T<n>")`. `tests/test_traceability.py` enforces this statically.
3. Unit tests use synthetic known-answer input; expected values are computed in the test from first principles.
4. Converter output must satisfy the SplatAD input contract (`contract/splatad_contract.py`); never hand data to the GPU box without a passing contract report.
5. Radar never enters the SplatAD dataset (SplatAD asserts no radars); it is a sidecar for alignment diagnostics.
6. Ground-truth cuboids stay separate from SplatAD inputs (`data/gt/`), used only for occupancy evaluation.
7. SplatAD training/rendering only runs on the remote NVIDIA box; the Mac runs CPU tests.
8. Network fetches use an anonymous identifier (e.g. `viking-splatad/0.1`); no personal info in headers, URLs or bodies.
9. Run Python with the project venv: `uv run pytest`.
