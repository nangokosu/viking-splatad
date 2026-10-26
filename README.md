# viking-splatad

Convert the Viking Hill radar/lidar/camera dataset (ROS2 MCAP) into inputs for SplatAD (neurad-studio) and render the forest.
May (short grass) and June (tall grass) are separate sessions with separate configs, raw dirs, datasets and GT.

Research question and proposal: `docs/proposal/`.
- **Execution plan:** `docs/PLAN.md`.
  - Scene A is a region with trunks, low grass in May and high grass in June: 3 consecutive bag chunks per run (May 7–9, June 20–22), no May↔June alignment.
  - May gets a stock SplatAD run with held-out metrics.
  - June compares SplatAD (B) against SplatAD + radar seed Gaussians + a radar opacity loss (R).
  - Radar is never rendered. Runs are judged on the rendered lidar point cloud against the June trunk, bush and rock cuboids.
- **SplatAD internals:** `docs/research/splatad_extension_points.md`.
- **Experiments:** `docs/experiments.md`.
- **Phase-1 findings:** `docs/research/phase1_findings.md`.
- **Project rules:** `CLAUDE.md` (`AGENTS.md` is a symlink to it, for Codex).
- **Slurm skills:** `.claude/skills/` (`.agents/skills` links to it, for Codex). Each user's cluster login stays on their own machine; see the one-time setup in `running-splatad-render-jobs`.

## Per-session runbook (replace `may_short_grass` with `june_tall_grass`)

```bash
CFG=configs/sessions/may_short_grass.yaml

# 1. Raw data (each bag chunk is an independent ~1.6 GB block of the Zenodo 7z)
uv run python scripts/00_fetch_chunk.py --config $CFG --fetch-calibration   # calibration, labels, metadata.yaml
uv run python scripts/00_fetch_chunk.py --config $CFG --chunk 1             # add --dry-run to see byte ranges first

# 2. Unpack ROS2 bag chunks -> SplatAD-ready intermediate dataset (incremental + idempotent;
#    global parameters are derived once from the first chunk and cached in derived.json)
uv run python scripts/03_extract.py --config $CFG                           # all chunks present in bag_dir
uv run python scripts/03_extract.py --config $CFG --max-seconds 10          # quick smoke on the first seconds

# 3. Contract + alignment checks (never train on a dataset that fails these)
uv run python scripts/06_validate_contract.py --dataset data/intermediate/may_short_grass
uv run python scripts/04_align_report.py --config $CFG                      # overlays + sensitivity sweep

# 4. Ground-truth cuboids / occupancy (separate from SplatAD inputs; evaluation only)
uv run python scripts/05_extract_cuboids.py --config $CFG
```

Outputs: `data/intermediate/<session>/` (manifest.json, poses.npz, camera/, lidar/, sidecar/radar/, derived.json,
contract_report.md), `data/gt/<session>/`, `reports/<session>/`.

## Tests
```bash
uv sync --extra dev                   # once: project deps + pytest
uv run pytest -q                      # CPU suite (synthetic, known-answer)
uv run pytest -m real_data -q         # needs the real chunk(s) on disk
uv run python scripts/affected_tests.py --run    # only tests affected by current changes
```
Every conversion/calibration is a registered `@transform("Tn")` with a `@pytest.mark.covers("Tn")` test
(`tests/test_traceability.py` enforces it). Existing tests are the spec: fix the module, not the test.
The read-only `transform-test-runner` agent (`.claude/agents/`) runs and audits them.

## SplatAD side (GPU box only)
`src/viking_splatad/splatad/install_into_neurad.sh` patches a neurad-studio checkout (adds `OUSTER_OS0_32`), then
`export NERFSTUDIO_DATAPARSER_CONFIGS=viking-data=viking_splatad.splatad.viking_dataparser:VIKING_DATAPARSER_SPEC`
and `pip install -e .` this repo in the neurad environment. The dataparser tests also run on CPU in a separate
venv (see `tests/unit/test_viking_dataparser.py` docstring). Nothing has been run under CUDA yet.
