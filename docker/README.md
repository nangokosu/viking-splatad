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
