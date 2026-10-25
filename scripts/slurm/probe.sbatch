#!/bin/bash
# Tiny probe of a GPU node's environment before staging anything (CLAUDE.md "GPU training on Slurm").
# Reports: scratch dirs and their expiry, apptainer location/version, GPU model + compute
# capability (the image is sm_86-only), and free space in the candidate stage dirs.
# Submit from comps0:  sbatch --partition=gpunodes --gres=gpu:rtx_a6000:1 probe.sbatch
# Resources are deliberately minimal (1 core, 1 GB, 5 min): the probe only runs ls/nvidia-smi,
# so it should not hold a node's CPUs/RAM (idle claimed resources lower our priority).
#SBATCH --job-name=viking-probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0-00:05:00
#SBATCH --output=%x-%j.out

set -uo pipefail

echo "== host"; hostname
echo "== scratch dirs (expiry encoded in name)"; ls -1 /scratch/scratch-space/ 2>&1 | egrep -i "expires|cannot"
echo "== other scratch candidates"; ls -d /scratch* /tmp 2>&1
echo "== apptainer"; command -v apptainer singularity 2>&1; apptainer --version 2>&1
echo "== gpu"; nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
echo "== free space"; df -h "$HOME" /scratch/scratch-space /tmp 2>&1
echo "== internet (for reference only; jobs must not depend on it)"
timeout 10 curl -s -o /dev/null -w "%{http_code}\n" -A "viking-splatad/0.1" https://github.com 2>&1 || echo "no internet"
