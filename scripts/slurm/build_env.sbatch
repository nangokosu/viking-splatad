#!/bin/bash
# Build the native (no-container) SplatAD environment on a CS Lab GPU node.
#
# Why: the GPU nodes have no container runtime and block unprivileged user namespaces (probe
# jobs 2026-10-25), so the reference image (docker/Dockerfile.splatad) cannot run there. This
# job replays the Dockerfile's recipe in user space, step for step, with the same pins:
#   * conda layer (python, gcc, CUDA toolkit)  <- scripts/slurm/cluster_env.yaml
#   * pip versions                              <- docker/constraints.txt (PIP_CONSTRAINT)
#   * git SHAs, dataparser registration         <- parsed from docker/Dockerfile.splatad itself
# so the Dockerfile stays the single source of truth for every upstream pin.
#
# Submitted by `scripts/slurm/submit_run.py build-env`, which passes resources on the sbatch
# command line and exports:
#   STAGE_ROOT          absolute stage dir on the cluster (micromamba, envs, code, data, runs)
#   CODE_DIR            committed repo snapshot to build from (git archive of HEAD)
#   ENV_PREFIX          where the environment goes (envs/splatad-<recipe hash>)
#   MICROMAMBA_VERSION  pinned micromamba release tag
#   MICROMAMBA_SHA256   its published sha256 (verified before use)
# Idempotent: exits early if ENV_PREFIX/.build-complete exists; a partial env is rebuilt.

set -euo pipefail

# ---- 0. Inputs and logging -------------------------------------------------------------------
: "${STAGE_ROOT:?}" "${CODE_DIR:?}" "${ENV_PREFIX:?}" "${MICROMAMBA_VERSION:?}" "${MICROMAMBA_SHA256:?}"
DOCKERFILE="$CODE_DIR/docker/Dockerfile.splatad"
CONSTRAINTS_FILE="$CODE_DIR/docker/constraints.txt"
echo "== build_env on $(hostname) at $(date -u +%FT%TZ), job ${SLURM_JOB_ID:-none}"
echo "   ENV_PREFIX=$ENV_PREFIX  CODE_DIR=$CODE_DIR"
if [[ -f "$ENV_PREFIX/.build-complete" ]]; then
    echo "environment already complete:"; cat "$ENV_PREFIX/.build-complete"; exit 0
fi
# Refuse to delete anything outside STAGE_ROOT/envs (guards the rm -rf below).
case "$ENV_PREFIX" in "$STAGE_ROOT"/envs/*) ;; *) echo "ENV_PREFIX outside $STAGE_ROOT/envs" >&2; exit 1;; esac
rm -rf "$ENV_PREFIX"

# ---- 1. micromamba (static binary), checksum-verified -----------------------------------------
MICROMAMBA="$STAGE_ROOT/bin/micromamba-$MICROMAMBA_VERSION"
if [[ ! -x "$MICROMAMBA" ]]; then
    mkdir -p "$STAGE_ROOT/bin"
    # Anonymous user agent (project rule 8: no personal info in network requests).
    curl -fsSL -A "viking-splatad/0.1" -o "$MICROMAMBA.download" \
        "https://github.com/mamba-org/micromamba-releases/releases/download/$MICROMAMBA_VERSION/micromamba-linux-64"
    echo "$MICROMAMBA_SHA256  $MICROMAMBA.download" | sha256sum -c -
    chmod +x "$MICROMAMBA.download" && mv "$MICROMAMBA.download" "$MICROMAMBA"
fi
# Package cache shared by all envs under the stage root (not in $HOME/.cache).
export MAMBA_ROOT_PREFIX="$STAGE_ROOT/mamba"

# ---- 2. Conda layer (replaces the base image + apt) ---------------------------------------------
"$MICROMAMBA" create -y -p "$ENV_PREFIX" -f "$CODE_DIR/scripts/slurm/cluster_env.yaml"

# ---- 3. Activation script: the one place run jobs get their environment from -----------------
# Dataparser registration copied verbatim from the Dockerfile's ENV line (T27).
DATAPARSER_CONFIGS="$(grep -oE '^ENV NERFSTUDIO_DATAPARSER_CONFIGS=[^ ]+' "$DOCKERFILE" | cut -d= -f2-)"
cat > "$ENV_PREFIX/viking-activate.sh" <<EOF
# Sourced by splatad_run.sbatch (and by this build after creation).
export PATH="$ENV_PREFIX/bin:\$PATH"
# Conda's CUDA toolkit lives in the env prefix; torch's extension builder and tcnn read CUDA_HOME.
export CUDA_HOME="$ENV_PREFIX"
# Env libs (cudart, libstdc++ from gcc 11.4) ahead of the host's; libcuda still comes from the host driver.
export LD_LIBRARY_PATH="$ENV_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export NERFSTUDIO_DATAPARSER_CONFIGS="$DATAPARSER_CONFIGS"
EOF
# shellcheck disable=SC1091
source "$ENV_PREFIX/viking-activate.sh"
# torch.utils.cpp_extension links against \$CUDA_HOME/lib64; conda's layout uses lib/.
[[ -e "$ENV_PREFIX/lib64" ]] || ln -s lib "$ENV_PREFIX/lib64"
# nvcc must use the env's g++ 11.4, never the host's gcc 15 (nvcc 11.8 rejects gcc > 11).
export NVCC_PREPEND_FLAGS="-ccbin $ENV_PREFIX/bin/g++"
export CC="$ENV_PREFIX/bin/gcc" CXX="$ENV_PREFIX/bin/g++"

# ---- 4. pip settings, identical to the Dockerfile's ENV block ----------------------------------
export PIP_CONSTRAINT="$CONSTRAINTS_FILE"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu118"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="$STAGE_ROOT/pip-cache"
export PYTHONDONTWRITEBYTECODE=1
# Parallel compile jobs = the CPUs Slurm granted (sized from sinfo by submit_run.py).
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-1}"
# Compile for the GPU actually in this node: nvidia-smi reports e.g. "8.6".
COMPUTE_CAPABILITY="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
export TCNN_CUDA_ARCHITECTURES="${COMPUTE_CAPABILITY/./}"      # "86"   (tiny-cuda-nn style)
export TORCH_CUDA_ARCH_LIST="$COMPUTE_CAPABILITY"              # "8.6"  (torch style)
echo "   compiling for compute capability $COMPUTE_CAPABILITY with MAX_JOBS=$MAX_JOBS"

# Pinned upstream SHAs, parsed from the Dockerfile ARG lines (single source of truth).
read_dockerfile_arg() { grep -oE "^ARG $1=[0-9a-f]+" "$DOCKERFILE" | cut -d= -f2; }
NEURAD_STUDIO_COMMIT="$(read_dockerfile_arg NEURAD_STUDIO_COMMIT)"
SPLATAD_COMMIT="$(read_dockerfile_arg SPLATAD_COMMIT)"
TINY_CUDA_NN_COMMIT="$(read_dockerfile_arg TINY_CUDA_NN_COMMIT)"
echo "   neurad-studio $NEURAD_STUDIO_COMMIT  splatad $SPLATAD_COMMIT  tiny-cuda-nn $TINY_CUDA_NN_COMMIT"

# ---- 5. Python toolchain + PyTorch (Dockerfile: "Python toolchain + PyTorch") -----------------
python -m pip install --upgrade pip "setuptools<70.0" wheel ninja pathtools promise pybind11
python -m pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118

# ---- 6. tiny-cuda-nn torch bindings -------------------------------------------------------------
python -m pip install --no-build-isolation \
    "git+https://github.com/NVlabs/tiny-cuda-nn.git@${TINY_CUDA_NN_COMMIT}#subdirectory=bindings/torch"

# ---- 7. neurad-studio at the pinned commit, patched (T28), editable -----------------------------
SOURCES_DIR="$ENV_PREFIX/viking-sources"
mkdir -p "$SOURCES_DIR"
git init -q "$SOURCES_DIR/neurad-studio"
git -C "$SOURCES_DIR/neurad-studio" remote add origin https://github.com/georghess/neurad-studio.git
git -C "$SOURCES_DIR/neurad-studio" fetch -q --depth 1 origin "$NEURAD_STUDIO_COMMIT"
git -C "$SOURCES_DIR/neurad-studio" checkout -q --detach FETCH_HEAD
bash "$CODE_DIR/src/viking_splatad/splatad/install_into_neurad.sh" "$SOURCES_DIR/neurad-studio"
(cd "$SOURCES_DIR/neurad-studio" && python -m pip install -e ".[dev]" "dataclass-wizard<0.23")

# ---- 8. splatad (gsplat fork) at the pinned commit, kernels compiled ahead of time -------------
git init -q "$SOURCES_DIR/splatad"
git -C "$SOURCES_DIR/splatad" remote add origin https://github.com/carlinds/splatad.git
git -C "$SOURCES_DIR/splatad" fetch -q --depth 1 origin "$SPLATAD_COMMIT"
git -C "$SOURCES_DIR/splatad" checkout -q --detach FETCH_HEAD
git -C "$SOURCES_DIR/splatad" submodule update -q --init --recursive --depth 1
(cd "$SOURCES_DIR/splatad" && BUILD_CUDA=1 python -m pip install --no-build-isolation --no-deps --force-reinstall -e .)

# ---- 9. This repo, editable, with the final-step pytest pin -------------------------------------
# Run jobs put their own committed src/ first on PYTHONPATH, so this install only provides the
# dependencies and the package metadata; it does not fix which code version a run uses.
PYTEST_PIN="$(grep -oE 'final-step-pin: pytest==[0-9.]+' "$CONSTRAINTS_FILE" | cut -d' ' -f2)"
python -m pip install -e "$CODE_DIR[dev]" "$PYTEST_PIN"

# ---- 10. viser web client (otherwise downloaded at the first ns-train) -------------------------
# The Dockerfile also runs `ns-install-cli`; skipped here because it edits shell rc files in $HOME.
python -c "import viser; viser.ViserServer()"

# ---- 11. Verification: the Dockerfile's checks plus the GPU check it could not do ---------------
python -m pip check
ns-train splatad --help > /dev/null
ns-train splatad viking-data --help > /dev/null
python -c "
import torch, gsplat.csrc, tinycudann
assert torch.cuda.is_available(), 'CUDA not available'
print('GPU import check OK:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
"
(cd "$CODE_DIR" && python -m pytest -q -p no:cacheprovider)

# ---- 12. Record the resolved environment (lock files to commit back) ---------------------------
python -m pip freeze --exclude-editable > "$ENV_PREFIX/viking-pip-freeze.txt"
"$MICROMAMBA" env export --explicit -p "$ENV_PREFIX" > "$ENV_PREFIX/viking-conda-explicit.txt"
# Drift vs the reference image's lock (lines only in one of the two files).
echo "== pip drift vs docker/constraints.txt (< lock only, > env only):"
diff <(grep -E '^[A-Za-z0-9_.-]+==' "$CONSTRAINTS_FILE" | sort) \
     <(grep -E '^[A-Za-z0-9_.-]+==' "$ENV_PREFIX/viking-pip-freeze.txt" | sort) || true
printf 'built %s on %s job %s\ncompute capability %s\ncode %s\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "${SLURM_JOB_ID:-none}" "$COMPUTE_CAPABILITY" "$CODE_DIR" \
    > "$ENV_PREFIX/.build-complete"
echo "== environment complete: $ENV_PREFIX"
