#!/bin/bash
# Tiny probe of a GPU node's environment before staging anything (CLAUDE.md "GPU training on Slurm").
# Reports: GPU model + compute capability (the env is sm_86-only), and free space in home and on
# the MSCAC workdisk (the probe's first argument, e.g. /w/20262).
# Submit from comps0:  sbatch --partition=gpunodes --gres=gpu:rtx_a6000:1 probe.sbatch /w/20262
# Resources are deliberately minimal (1 core, 1 GB, 5 min): the probe only runs df/nvidia-smi,
# so it should not hold a node's CPUs/RAM (idle claimed resources lower our priority).
#SBATCH --job-name=viking-probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:05:00
#SBATCH --output=%x-%j.out

set -uo pipefail

# Workdisk root to report on (configs/slurm.yaml workdisk_root); required, never guessed.
WORKDISK_ROOT="${1:?usage: sbatch ... probe.sbatch <workdisk_root>}"

echo "== host"; hostname
echo "== gpu"; nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
# Home has a 10 GB MSCAC quota; the workdisk holds every project file.
echo "== free space"; df -h "$HOME" "$WORKDISK_ROOT/$USER" /tmp 2>&1
echo "== internet (for reference only; jobs must not depend on it)"
timeout 10 curl -s -o /dev/null -w "%{http_code}\n" -A "viking-splatad/0.1" https://github.com 2>&1 || echo "no internet"
