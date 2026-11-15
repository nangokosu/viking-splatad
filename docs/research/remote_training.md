# Training from a lightweight Mac submit host

This opt-in workflow keeps raw bags and converted inputs on the cluster workdisk.
The Mac submits jobs and fetches diagnostic overlays and completed run outputs.
Local Docker tests were omitted at the user's request for this session; the
workflow records `native-cluster` validation rather than claiming image validation.
The original local-data `stage`, `gates`, and `submit` commands remain available.

## Environment recovery

After a native build compiled successfully but stopped on test prerequisites:

```bash
uv run python scripts/slurm/submit_run.py verify-env
```

This stages committed sources and supplies real May calibration/metadata for the
existing tests, source fixtures from the pinned upstream checkout, instruction
files, and an identity-free Git index of the staged files. It runs pip checks,
GPU imports, and the full default suite with the strict exclusion audit active.
It does not reinstall dependencies or compile kernels. The completion markers
are published only after success. It uses the compatible render GPU configured
in `configs/slurm.yaml`, sized from live `sinfo` rows.

A missing compiled environment still requires `build-env` first. Do not run
multiple build/recovery jobs on the same prefix concurrently.

After the recipe environment has passed GPU verification, source-only changes
can run the full suite on CPU without reserving a GPU:

```bash
uv run python scripts/slurm/submit_run.py verify-env --resources validation_cpu
```

This requires an existing readiness marker and never writes environment markers.
An explicit `validation_gpu` alternative is also configured for compatible
validation hardware; it cannot override a training allocation.

## Data preparation

`scripts/slurm/prepare_data.sbatch` takes four positional arguments: stage root,
committed code directory, completed native environment prefix, and session name.
Submit it to a CPU partition with resources read from `sinfo` and enough memory
for the converter. Both June attempts completed extraction, then exceeded their
10/20 GiB limits in eager contract validation. Validation now streams camera
images and lidar arrays instead of retaining all decoded frames in memory.
Resource requests are workload budgets, not proportional CPU-node shares.

The script downloads calibration and the YAML's exact pinned chunks, retains
raw files under `source-data/`, and extracts into a unique fresh workspace under
`data-prep/`. It never passes a chunk override to extraction. Only a dataset
passing the contract and configured reference-split check is atomically published
at `data/<session>/`. Existing published datasets are refused, never overwritten.
Failed workspaces remain for diagnosis. A per-session lock prevents concurrent
cache writers; do not remove an active job's lock.

For a workspace whose extraction completed before validation failed,
`publish_prepared_data.sbatch` takes stage root, code directory, verified env
prefix, session, and the completed dataset workspace. It checks the recorded
pinned chunks and radar sidecar count, repeats the complete contract and
reference split checks, and publishes atomically. It never reruns extraction or
changes the chunk set. Job 240771 successfully recovered the June dataset this way.

## Remote preflight

Commit code before staging. From the repository on the Mac:

```bash
uv run python scripts/slurm/submit_run.py remote-preflight \
  --session june_tall_grass --profile smoke_a4500_june --resources validation_cpu
```

To queue it behind a data job, add `--afterok <data-job-id>`. The batch script
still requires the completed environment when it starts.

Preflight runs full native tests, the dataset contract, the committed reference
split, and lidar-on-camera overlay generation. Window fractions are computed
from the workdisk timestamps. The overlay check diagnoses calibration and pose
geometry; script 16 separately checks the canonical frame membership.

Outputs live under `preflights/<session>/<profile>/<git-sha>/`. Inspect the
resulting overlays before training. The validation receipt binds the code,
environment, profile, session config, manifest, poses, and diagnostic files by
hash. It is invalidated before a revalidation starts so a failed retry cannot
reuse an earlier receipt. CPU preflight uses the already verified native env;
the default without `--resources` uses the configured render GPU. Full
optimization retains the profile's training GPU, regardless of preflight resources.

## Training and fetching

After successful preflight and overlay review:

```bash
uv run python scripts/slurm/submit_run.py submit-remote \
  --session june_tall_grass --profile smoke_a4500_june --reviewed-preflight
```

Preflight must be for the current commit. Training checks the matching receipt
and unchanged overlays. The acknowledgement does not bypass provenance checks.
The resolved training variables are recorded in the run's standard `job.sbatch`;
the original wrapper and validation receipt are preserved beside it.

Run and review the full profile's preflight as well. Then queue the full baseline
behind a successful smoke job:

```bash
uv run python scripts/slurm/submit_run.py submit-remote \
  --session june_tall_grass --profile june_scene_a \
  --reviewed-preflight --afterok <smoke-job-id>
```

Use `runs` to monitor training runs; use `squeue` for environment/data/preflight
jobs. Fetch successful training outputs with `fetch <run-id>` as usual. Queue
exit alone does not establish success: require the written overfitting plot,
matching split record, and `end_utc` milestone before fetching.

## Scientific scope

`smoke_a4500_june` is a short infrastructure check, not a reconstruction result.
`june_scene_a` now selects core B: lidar-only optimization with camera-painted
initialization. Older photometric runs of that profile are B+cam and must not be
mislabelled as core B. Reference split settings have not changed.

The selected radar arm is now loss-only L: fixed anchors supervise opacity,
with no additional Gaussians, semantic labels or radar rendering. Calibrated
train-only artifact construction, checksum/split/frame gates, stock-config
preservation, automatic step-zero weight matching and support diagnostics are
implemented. See [the implementation notes](loss_only_implementation.md).
A successful baseline or falling radar loss does not establish hidden-trunk gain.
Repeat arms before interpreting differences; evaluation cuboids remain outside training.
