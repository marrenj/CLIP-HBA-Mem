"""Hyperparameter sweep for the frozen CLIP-ViT-L/14 + MLP head.

Usage
-----
    cd CLIP-HBA
    python train_clip_mlp_sweep.py

    # Resume an interrupted sweep (skips runs with valid results, re-runs nans)
    python train_clip_mlp_sweep.py --resume ./sweep_out_clip_mlp/20260306_214413

Edit BASE_CONFIG (data paths, device) and SWEEP_GRID (search space) below
before running.  All 48 combinations are evaluated sequentially on fold 1
using 20 % of the training data and early-stopping patience of 12, so each
run is much faster than a full training run.  After the sweep, re-train the
winning config on the full training set (train_fraction=1.0, patience=10)
across all 5 folds to get final performance numbers.

Outputs
-------
  sweep_out_clip_mlp/<timestamp>/run_NNN/log.txt  -- per-run training log
  sweep_out_clip_mlp/<timestamp>/sweep_results.csv -- all results sorted by val Spearman rho
"""

import argparse
import csv
import datetime
import itertools
import os
import sys

import torch.nn as nn

from functions.train_mem_pipeline import run_mem_training

# ---------------------------------------------------------------------------
# Fixed configuration -- edit paths and device to match your environment
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    'model_type': 'clip_frozen_mlp',

    # Fold used for the sweep; run the winning config on all 5 folds afterward
    'fold':      1,
    'train_csv': './Data/lamem/lamem_train_1.csv',
    'val_csv':   './Data/lamem/lamem_val_1.csv',
    'test_csv':  './Data/lamem/lamem_test_1.csv',
    'img_root':  './Data/lamem/images/',

    # Device: 0=cuda:0, 1=cuda:1, -1=DataParallel, 2=cpu
    'cuda': 1,

    # Fixed training settings for the sweep
    'epochs':                  300,
    'early_stopping_patience': 12,
    'train_fraction':          0.2,
    'criterion':               nn.MSELoss(),
    'random_seed':             42,
}

# ---------------------------------------------------------------------------
# Search grid   (4 x 2 x 3 x 2 = 48 total combinations)
# ---------------------------------------------------------------------------
SWEEP_GRID = {
    'hidden_dims':  [(512, 256), (256, 128), (512, 256, 128), (256,)],
    'dropout_rate': [0.3, 0.5],
    'lr':           [1e-4, 5e-5, 1e-5],
    'batch_size':   [32, 64],
}

# ---------------------------------------------------------------------------
# Sweep output directory
# ---------------------------------------------------------------------------
SWEEP_DIR = f'./sweep_out_clip_mlp/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'


def _load_completed_runs(results_csv):
    """Return a set of run_ids that already have a non-nan best_val_rho."""
    import math
    completed = set()
    if not os.path.exists(results_csv):
        return completed
    with open(results_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rho = float(row['best_val_rho'])
                if not math.isnan(rho):
                    completed.add(int(row['run_id']))
            except (ValueError, KeyError):
                pass
    return completed


def run_sweep(resume_dir=None):
    import json

    sweep_dir = resume_dir if resume_dir else SWEEP_DIR
    os.makedirs(sweep_dir, exist_ok=True)

    # Keep a reference to the real stdout so sweep progress is always visible
    # even though run_mem_training redirects stdout to a per-run log file.
    sweep_stdout = sys.stdout

    keys = list(SWEEP_GRID.keys())
    combos = list(itertools.product(*[SWEEP_GRID[k] for k in keys]))
    n_total = len(combos)

    results_csv = os.path.join(sweep_dir, 'sweep_results.csv')

    # On resume, find which runs completed successfully; otherwise write fresh header.
    if resume_dir:
        completed_ids = _load_completed_runs(results_csv)
        sweep_stdout.write(
            f'[Resume] Found {len(completed_ids)} completed run(s) in {resume_dir}\n'
        )
        sweep_stdout.flush()
    else:
        completed_ids = set()
        with open(results_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['run_id', *keys, 'best_val_rho'])

    results = []

    for run_id, values in enumerate(combos):
        params = dict(zip(keys, values))

        if run_id in completed_ids:
            sweep_stdout.write(f'\n[Sweep {run_id + 1}/{n_total}] Skipping (already done)\n')
            sweep_stdout.flush()
            continue

        sweep_stdout.write(
            f'\n[Sweep {run_id + 1}/{n_total}] '
            + '  '.join(f'{k}={v}' for k, v in params.items())
            + '\n'
        )
        sweep_stdout.flush()

        run_dir = os.path.join(sweep_dir, f'run_{run_id:03d}')
        os.makedirs(run_dir, exist_ok=True)

        config = {
            **BASE_CONFIG,
            **params,
            'log_path':        os.path.join(run_dir, 'log.txt'),
            'checkpoint_path': os.path.join(run_dir, 'checkpoint'),
        }

        with open(os.path.join(run_dir, 'config.json'), 'w') as _f:
            json.dump(
                {k: str(v) for k, v in config.items() if k != 'criterion'},
                _f, indent=2,
            )

        try:
            best_rho = run_mem_training(config)
        except Exception as exc:
            sweep_stdout.write(f'  [ERROR] run {run_id} failed: {exc}\n')
            sweep_stdout.flush()
            best_rho = float('nan')

        # sys.stdout was restored inside run_mem_training; sweep output is safe.
        results.append({'run_id': run_id, **params, 'best_val_rho': best_rho})

        with open(results_csv, 'a', newline='') as f:
            csv.writer(f).writerow(
                [run_id, *[params[k] for k in keys], f'{best_rho:.6f}']
            )

        sweep_stdout.write(f'  -> best_val_rho = {best_rho:.4f}\n')
        sweep_stdout.flush()

    # --- Summary ---
    # Re-read all results (including skipped runs) for the final leaderboard.
    all_results = []
    with open(results_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rho = float(row['best_val_rho'])
            except ValueError:
                rho = float('nan')
            all_results.append({
                'run_id':       int(row['run_id']),
                'hidden_dims':  row['hidden_dims'],
                'dropout_rate': float(row['dropout_rate']),
                'lr':           float(row['lr']),
                'batch_size':   int(row['batch_size']),
                'best_val_rho': rho,
            })
    all_results.sort(key=lambda r: r['best_val_rho'], reverse=True)

    sweep_stdout.write('\n' + '=' * 65 + '\n')
    sweep_stdout.write(f'Sweep complete. Full results: {results_csv}\n')
    sweep_stdout.write('Top 10 configurations by validation Spearman rho:\n\n')
    header = (
        f"{'Run':>4}  {'hidden_dims':>18}  {'drop':>5}"
        f"  {'lr':>8}  {'bs':>4}  {'val_rho':>8}"
    )
    sweep_stdout.write(header + '\n')
    sweep_stdout.write('-' * len(header) + '\n')
    for r in all_results[:10]:
        sweep_stdout.write(
            f"{r['run_id']:>4}  {str(r['hidden_dims']):>18}  {r['dropout_rate']:>5.1f}"
            f"  {r['lr']:>8.0e}  {r['batch_size']:>4}  {r['best_val_rho']:>8.4f}\n"
        )
    sweep_stdout.write('=' * 65 + '\n')
    sweep_stdout.write(
        '\nNext step: copy the winning config into train_clip_mlp.py and run with\n'
        '  train_fraction=1.0  and  early_stopping_patience=10  across all 5 folds.\n'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for frozen CLIP + MLP.')
    parser.add_argument(
        '--resume', metavar='DIR', default=None,
        help='Path to an existing sweep_out_clip_mlp/<timestamp> directory to resume. '
             'Runs with a valid (non-nan) best_val_rho are skipped; all others are re-run.',
    )
    args = parser.parse_args()
    run_sweep(resume_dir=args.resume)
