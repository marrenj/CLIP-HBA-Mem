#!/bin/bash
#SBATCH --job-name=clip_hba_mem
#SBATCH --account=brain_ai
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err
#SBATCH --gres=gpu:nvidia_h100_80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu
 
# ---------------------------------------------------------------------------
# Usage:
#   sbatch train_mem.slurm
#
# Trains all 5 LaMem folds sequentially on a single node.
# Data is copied to /tmp once and reused across all folds.
#
# Submit from the CLIP-HBA directory (the one containing train_mem.py):
#   cd /path/to/CLIP-HBA-Mem/CLIP-HBA
#   sbatch train_mem.slurm
#
# Logs:
#   SLURM stdout/stderr  ->  ./logs/slurm_<jobid>.out / .err
#   Training log         ->  ./logs/mem_fold<N>_<timestamp>.log  (written by train_mem.py)
# ---------------------------------------------------------------------------
 
set -euo pipefail
 
# --- Environment setup ------------------------------------------------------
# conda activate myenv
# source /path/to/venv/bin/activate
 
cd "$(dirname "$0")"
 
mkdir -p ./logs ./preds ./models
 
echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

# =============================================================================
# Copy LaMem images to local /tmp once; all folds read from the same copy.
# =============================================================================

LOCAL_DATA=/tmp/lamem_${SLURM_JOB_ID}
mkdir -p "${LOCAL_DATA}"
trap "rm -rf ${LOCAL_DATA}" EXIT

echo "Copying LaMem to local storage: ${LOCAL_DATA} ..."
rsync -a --info=progress2 "./Data/lamem/images/" "${LOCAL_DATA}/"
echo "Data copy complete."
echo "=========================================="

export LAMEM_IMG_ROOT="${LOCAL_DATA}"

# --- Train all 5 folds sequentially -----------------------------------------
for FOLD in 1 2 3 4 5; do
    echo ""
    echo "=========================================="
    echo "Starting fold ${FOLD} at $(date)"
    echo "=========================================="
    export LAMEM_FOLD="${FOLD}"
    python train_mem.py
done
 
echo "=========================================="
echo "All folds complete."
echo "End time : $(date)"
echo "=========================================="