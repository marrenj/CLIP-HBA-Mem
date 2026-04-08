#!/bin/bash
# =============================================================================
# SLURM job array — Bayesian hyperparameter search for CLIP-HBA-MEG MLP heads
#
# Runs one Optuna TPE search per MEG timepoint for the 36 NEW timepoints at
# 25 ms resolution that are not already covered by the existing results in
# meg_hyperparam_search_results.csv (which has 50–1300 ms at coarser spacing).
#
# New timepoints (36 total, indices 0–35):
#   0, 25, 75, 125, 175, 225, 275, 325, 375, 425, 450, 475,
#   525, 550, 575, 625, 650, 675, 725, 750, 775, 825, 850, 875,
#   925, 950, 975, 1025, 1050, 1075, 1125, 1150, 1175, 1225, 1250, 1275
#
# Already covered (skipped):
#   50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800,
#   900, 1000, 1100, 1200, 1300
#
# Resource notes
# --------------
# This is a lightweight workload: tiny MLP (66-dim input), in-memory embeddings
# (num_workers=0), Optuna TPE sampler (single-threaded).
#   GPU:    1 A100 — barely stressed but keeps jobs on the same partition
#   CPUs:   4      — Optuna + Python overhead; extra cores go unused
#   Memory: 16 G   — all 66-dim embeddings for 10 folds fit in ~400 MB
#   Time:   24 h   — 50 trials × 10 folds × ~2-3 min/fold ≈ 17–25 h
#
# Usage
# -----
#   cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA
#
#   # Edit %N to match available GPU count, then:
#   sbatch slurm/hyperparam_search_meg.sh
#
#   # Or override %N at submission time:
#   sbatch --array=0-35%1 slurm/hyperparam_search_meg.sh   # 1 GPU
#   sbatch --array=0-35%2 slurm/hyperparam_search_meg.sh   # 2 GPUs
#   sbatch --array=0-35%4 slurm/hyperparam_search_meg.sh   # 4 GPUs
#
#   # Resume a failed/preempted task (e.g. index 7 = tp=425 ms):
#   sbatch --array=7 slurm/hyperparam_search_meg.sh
#
# Monitoring
# ----------
#   squeue -u $USER
#   scancel <array_job_id>
#   scancel <array_job_id>_3
#   tail -f ./logs/hyperparam_search_meg/slurm_<jobid>_<idx>.out
# =============================================================================

#SBATCH --job-name=meg_hpsearch
#SBATCH --output=./logs/hyperparam_search_meg/slurm_%A_%a.out
#SBATCH --error=./logs/hyperparam_search_meg/slurm_%A_%a.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-35%2        # 36 new timepoints (indices 0–35); %2 = max 2 concurrent
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Timepoint lookup — 25 ms resolution, skipping already-completed timepoints
# (50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300)
# ---------------------------------------------------------------------------
TIMEPOINTS=(
      0    25    75   125   175   225   275   325   375
    425   450   475   525   550   575   625   650   675
    725   750   775   825   850   875   925   950   975
   1025  1050  1075  1125  1150  1175  1225  1250  1275
)
TP=${TIMEPOINTS[$SLURM_ARRAY_TASK_ID]}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load python/3.11.5
source ~/envs/clip_hba/bin/activate

cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA

# Log dir must exist before SLURM writes to it
mkdir -p ./logs/hyperparam_search_meg

# The search script reads CUDA_DEVICE from the environment (defaults to 1).
# Each SLURM task gets an exclusive GPU so cuda:0 is always correct here.
export CUDA_DEVICE=0

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Array job ID : $SLURM_ARRAY_JOB_ID"
echo "Task index   : $SLURM_ARRAY_TASK_ID"
echo "Timepoint    : ${TP} ms"
echo "Node         : $SLURMD_NODENAME"
echo "GPU(s)       : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

# ---------------------------------------------------------------------------
# Hyperparameter search — 50 Optuna TPE trials, all 10 folds per trial
# Outputs written to ./sweep_meg_out/<timestamp>/tp<TP>/
# ---------------------------------------------------------------------------
python train_mem_meg_hyperparam_search.py \
    --timepoint_ms "${TP}" \
    --n-trials 50

# ---------------------------------------------------------------------------
echo "=========================================="
echo "Finished tp=${TP} ms"
echo "End time : $(date)"
echo "=========================================="
