#!/bin/bash
#SBATCH --job-name=clip_hba_mem
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ---------------------------------------------------------------------------
# Usage:
#   sbatch train_mem.slurm
#
# Submit from the CLIP-HBA directory (the one containing train_mem.py):
#   cd /path/to/CLIP-HBA-Mem/CLIP-HBA
#   sbatch train_mem.slurm
#
# Logs:
#   SLURM stdout/stderr → ./logs/slurm_<jobid>.out / .err
#   Training log        → ./logs/mem_fold1_<timestamp>.log  (written by train_mem.py)
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Environment setup ------------------------------------------------------
# Activate your conda/virtualenv if needed; adjust the path below.
# conda activate myenv
# source /path/to/venv/bin/activate

# Ensure the CLIP-HBA directory is the working directory so that relative
# paths in the config (./Data/..., ./logs/..., etc.) resolve correctly.
cd "$(dirname "$0")"

# Create output directories in case they don't exist yet.
mkdir -p ./logs ./preds ./models

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

# --- Run training -----------------------------------------------------------
python train_mem.py

echo "=========================================="
echo "End time : $(date)"
echo "=========================================="
