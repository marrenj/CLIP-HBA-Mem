#!/bin/bash
# =============================================================================
# SLURM job array — final MLP training for CLIP-HBA-MEG memorability heads
#
# Submits one task per MEG timepoint (17 total).  Each task trains all 10
# combined_lamem_memcat folds for that timepoint sequentially.  SLURM queues
# tasks beyond the concurrency limit and launches them automatically as GPUs
# free up — no manual polling required.
#
# Usage
# -----
#   cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA
#
#   # Edit %N below to match your available GPU count, then:
#   sbatch slurm/train_mem_meg_final.sh
#
#   # Or override %N at submission time without editing the file:
#   sbatch --array=0-16%1 slurm/train_mem_meg_final.sh   # 1 GPU
#   sbatch --array=0-16%2 slurm/train_mem_meg_final.sh   # 2 GPUs
#   sbatch --array=0-16%4 slurm/train_mem_meg_final.sh   # 4 GPUs
#
#   # Run only a subset of timepoints (array indices 0–4 = 50–250 ms):
#   sbatch --array=0-4%2 slurm/train_mem_meg_final.sh
#
# Monitoring
# ----------
#   squeue -u $USER                                        # running / pending tasks
#   scancel <array_job_id>                                 # cancel all tasks
#   scancel <array_job_id>_3                               # cancel task index 3 only
#   tail -f ./logs/meg_mem_final/slurm_<jobid>_<idx>.out  # follow a task log
#
# Outputs (written by train_mem_meg_final.py)
# -------------------------------------------
#   ./logs/meg_mem_final/slurm_%A_%a.{out,err}             — SLURM stdout / stderr
#   ./logs/meg_mem_final/meg_mem_tp{tp}_fold{fold}.log     — per-fold training log
#   ./models/clip_hba_meg_mem_final/tp{tp}/                — best checkpoints
#   ./results/meg_mem_final_summary_*.csv                  — mean±std across folds
# =============================================================================

#SBATCH --job-name=meg_mem_final
#SBATCH --output=./logs/meg_mem_final/slurm_%A_%a.out
#SBATCH --error=./logs/meg_mem_final/slurm_%A_%a.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-16%2        # 17 timepoints (indices 0–16); %2 = max 2 concurrent
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# Timepoint lookup — index matches --array range (0–16)
# ---------------------------------------------------------------------------
TIMEPOINTS=(50 100 150 200 250 300 350 400 500 600 700 800 900 1000 1100 1200 1300)
TP=${TIMEPOINTS[$SLURM_ARRAY_TASK_ID]}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load python/3.11.5
source ~/envs/clip_hba/bin/activate

cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA

# SLURM log dir must exist before the job writes to it
mkdir -p ./logs/meg_mem_final

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
# Training — all 10 folds for this timepoint
# Each fold trains on 100% of the training split, validates each epoch, and
# evaluates on test using the best checkpoint after early stopping.
# ---------------------------------------------------------------------------
python train_mem_meg_final.py \
    --timepoints "${TP}" \
    --epochs 300 \
    --early_stopping_patience 20 \
    --cuda 0 \
    --seed 1

# ---------------------------------------------------------------------------
echo "=========================================="
echo "Finished tp=${TP} ms"
echo "End time : $(date)"
echo "=========================================="
