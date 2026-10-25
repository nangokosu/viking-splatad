# Reproducible SplatAD image

`Dockerfile.splatad` builds the full training/rendering environment: CUDA 11.8, torch 2.0.1, tiny-cuda-nn,
neurad-studio with the `OUSTER_OS0_32` patch (T28), splatad/gsplat kernels compiled ahead of time, and this repo
installed editable with the `viking-data` dataparser (T27) registered. Every upstream source is pinned: base image
by digest, git repos by full SHA, pip by `constraints.txt`. The build finishes by running this repo's CPU test suite.

## Build (Mac or any x86_64 Docker host)

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.splatad \
    --build-arg CUDA_ARCHITECTURES="<semicolon list, e.g. from cluster nvidia-smi>" \
    -t viking-splatad:cu118 .
```

`CUDA_ARCHITECTURES` should list exactly the GPUs on the cluster's gpunodes. Read them from a short interactive job:
`srun --partition=gpunodes --gres=gpu:1 -t 5 nvidia-smi --query-gpu=name,compute_cap --format=csv`. Write each
capability without its dot: 8.6 becomes 86. Leaving it out compiles for upstream's default list, which builds slower and makes a larger image.

On an Apple-silicon Mac the build runs under amd64 emulation. nvcc does not need a GPU, so the build works, but expect hours.

## Lock pip versions after the first successful build

```bash
docker run --rm viking-splatad:cu118 \
    pip freeze --exclude-editable --exclude tinycudann > docker/constraints.txt
```

Commit the result. Later builds then resolve to the same versions.

## Run on the CS Lab Slurm cluster (Apptainer)

Shared Slurm nodes run containers through Apptainer, not Docker. Convert the image to a single `.sif` file:

```bash
docker save viking-splatad:cu118 -o viking-splatad-cu118.tar          # on the Mac
rsync -P viking-splatad-cu118.tar comps0.cs.toronto.edu:/scratch/scratch-space/<latest-expiring>/
# inside an sbatch/srun job (not on comps0–3):
apptainer build viking-splatad-cu118.sif docker-archive://viking-splatad-cu118.tar
apptainer exec --nv --bind <data_dir>:/data viking-splatad-cu118.sif ns-train splatad viking-data --data /data/...
```

`--nv` passes the host NVIDIA driver into the container. Only the driver comes from the host. Everything else is in the image.

## Built images

| Tag | Image ID | Built from commit | `CUDA_ARCHITECTURES` | Notes |
|---|---|---|---|---|
| `viking-splatad:cu118` | `aab3781dea3e` | `a14bb2a` | `86` | sm_86 cubin only, no PTX: runs on the RTX A6000/A4500/A4000/A2000 nodes, **not** the RTX 4090 (sm_89) nodes. `constraints.txt` was frozen from this image afterwards. |

Layer order matters for rebuild cost. `docker/constraints.txt` and `src/` are copied **before** the neurad-studio
and splatad CUDA builds, so changing either one recompiles everything (hours under emulation). `tests/`,
`configs/` and `scripts/` are copied after, so changing them is cheap.

## The CS Lab cluster runs a native build of this recipe, not the image

The CS Lab GPU nodes have no Apptainer, Singularity, Docker or Podman, and unprivileged user namespaces are disabled
(probe jobs, 2026-10-25), so neither this image nor a rootless container can run there. `scripts/slurm/build_env.sbatch`
replays this Dockerfile in user space on a GPU node, with micromamba providing Python 3.10.12, gcc 11.4.0 and CUDA
11.8 (`scripts/slurm/cluster_env.yaml`). It reads the git SHAs and the dataparser registration from this
Dockerfile and the pip pins from `constraints.txt`, so this file stays the single recipe. The image is still the
reference build, and it is used on the Mac for CPU checks such as `scripts/slurm/preflight_dataparser.py`. The
Apptainer section above applies only to clusters that provide Apptainer.

`constraints.txt` does not pin `pytest` globally: neurad-studio's `[dev]` extra pins `pytest==7.1.2` for its own
install step, while this repo's `[dev]` extra upgrades it. Only the final install step applies the
`final-step-pin` comment line.
