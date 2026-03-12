#!/bin/bash
#SBATCH --job-name=clip_hba_mem
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=30:00:00
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
# conda activate clip-hba
module load python/3.11.5
source ~/envs/clip_hba/bin/activate
 
cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA
 
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

LOCAL_DATA=/tmp/lamem_cache
if [ ! -d "${LOCAL_DATA}" ]; then
    echo "Copying LaMem to local storage..."
    mkdir -p "${LOCAL_DATA}"
    rsync -a --info=progress2 "./Data/lamem/images/" "${LOCAL_DATA}/"
else
    echo "Using cached LaMem data at ${LOCAL_DATA}"
fi

echo "=========================================="

export LAMEM_IMG_ROOT="${LOCAL_DATA}"

# --- Train all 5 folds sequentially -----------------------------------------
for FOLD in 2 3; do
    echo ""
    echo "=========================================="
    echo "Starting fold ${FOLD} at $(date)"
    echo "=========================================="
    export LAMEM_FOLD="${FOLD}"
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    if [ "$N_GPUS" -eq 1 ]; then
        export CUDA_DEVICE=0
    else
        export CUDA_DEVICE=-1  # DataParallel for multi-GPU
    fi
    python train_mem.py
done
 
echo "=========================================="
echo "All folds complete."
echo "End time : $(date)"
echo "=========================================="