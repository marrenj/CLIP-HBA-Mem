"""Bayesian hyperparameter search for CLIP-HBA-MEG memorability MLP heads.

Runs one independent Optuna TPE search per sampled timepoint.  Each trial
trains a small MLP head on top of precomputed 66-dim CLIP-HBA-MEG embeddings
and evaluates it across all cross-validation folds.  The search space includes
both compression-only and expand-then-compress architectures, which are more
appropriate for a 66-dim input than the 512/256 hidden sizes used by the
768-dim CLIP-HBA-Behavior head.

Usage
-----
    cd CLIP-HBA

    # Sweep for a single timepoint:
    python train_mem_meg_sweep.py --timepoint_ms 100 [--n-trials 50]

    # Resume a previous sweep:
    python train_mem_meg_sweep.py --timepoint_ms 100 --resume sweep_meg_out/.../optuna_study.db

    # Run all 15 timepoints sequentially (consider parallelising via SLURM):
    for TP in -100 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300; do
        python train_mem_meg_sweep.py --timepoint_ms $TP --n-trials 50
    done

Prerequisites
-------------
    Pre-extract MEG embeddings once:
        python extract_embeddings_meg.py --meg_checkpoint ./models/cliphba_meg_group.pth

Outputs (per timepoint)
-----------------------
    sweep_meg_out/<timestamp>/tp<timepoint_ms>/run_NNN/config.json
    sweep_meg_out/<timestamp>/tp<timepoint_ms>/run_NNN/fold_K/log.txt
    sweep_meg_out/<timestamp>/tp<timepoint_ms>/sweep_results.csv
    sweep_meg_out/<timestamp>/tp<timepoint_ms>/optuna_study.db
    sweep_meg_out/<timestamp>/tp<timepoint_ms>/sweep_config.json
"""

import argparse
import ast
import csv
import datetime
import json
import math
import os
import sys

import torch.nn as nn

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as _e:
    raise ImportError(
        'Optuna is required for Bayesian hyperparameter search.\n'
        'Install it with:  pip install optuna'
    ) from _e

from functions.train_mem_pipeline import run_mem_training

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
TRAINING_DATA: str = os.environ.get('TRAINING_DATA', 'combined_lamem_memcat')

if TRAINING_DATA == 'lamem':
    N_FOLDS: int = 5
    _IMG_ROOT: 'str | dict' = os.environ.get('LAMEM_IMG_ROOT', './Data/lamem/images/')
    _EMBEDDINGS_DIR_BASE: str = './Data/lamem/meg_embeddings/'
    _MEMCAT_META_CSV: 'str | None' = None

    def _csv_path(split: str, fold: int) -> str:
        return f'./Data/lamem/lamem_{split}_{fold}.csv'

elif TRAINING_DATA == 'combined_lamem_memcat':
    N_FOLDS: int = 10
    _IMG_ROOT: 'str | dict' = {
        'lamem':  os.environ.get('LAMEM_IMG_ROOT',  './Data/lamem/images/'),
        'memcat': os.environ.get('MEMCAT_IMG_ROOT', './Data/memcat/images/'),
    }
    _EMBEDDINGS_DIR_BASE: str = (
        os.environ.get('MEG_EMBEDDINGS_DIR') or
        './Data/combined_lamem_memcat/meg_embeddings/'
    )
    _MEMCAT_META_CSV: 'str | None' = os.environ.get(
        'MEMCAT_META_CSV', './Data/memcat/memcat_image_data.csv'
    )

    def _csv_path(split: str, fold: int) -> str:
        return f'./Data/combined_lamem_memcat/lamem_memcat_{split}_split_{fold:02d}.csv'

else:
    raise ValueError(
        f'Unknown TRAINING_DATA: {TRAINING_DATA!r}. '
        f"Choose 'lamem' or 'combined_lamem_memcat'."
    )

# ---------------------------------------------------------------------------
# Hyperparameter search space
#
# For a 66-dim input the original 512/256 hidden sizes are over-parameterised.
# We search compression-only architectures (monotonically decreasing from 66)
# and expand-then-compress architectures (bottleneck after an initial expansion).
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict = {
    'hidden_dims': [
        # Compression-only
        '(32,)',
        '(16,)',
        '(32, 16)',
        '(16, 8)',
        # Expand-then-compress
        '(128, 32)',
        '(256, 64)',
        '(128, 64, 16)',
        '(256, 128, 32)',
    ],
    'dropout_rate': (0.1, 0.7),
    'lr':           (1e-5, 5e-4),
    'weight_decay': (1e-5, 1e-1),
    'batch_size':   [32, 64, 128, 256],
}

N_TRIALS_DEFAULT: int = 50
INPUT_DIM:        int = 66   # CLIP-HBA-MEG embedding dimensionality

_CSV_FIELDS = [
    'trial_id', 'hidden_dims', 'dropout_rate', 'lr', 'weight_decay', 'batch_size',
    'mean_val_mse', 'std_val_mse', 'mean_val_rho', 'std_val_rho',
]


def _sample_std(values: list) -> float:
    """Return the sample standard deviation of *values*, or 0.0 if len < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _objective(
    trial: 'optuna.Trial',
    timepoint_ms: int,
    sweep_dir: str,
    results_csv: str,
    sweep_stdout,
) -> float:
    """Sample hyperparameters, train on all folds, return mean validation MSE.

    Args:
        trial:          Optuna trial object.
        timepoint_ms:   Timepoint in ms being swept (determines embedding file).
        sweep_dir:      Root directory for this sweep's per-trial subdirectories.
        results_csv:    Path to the incremental CSV results file.
        sweep_stdout:   Reference to real stdout for progress reporting.

    Returns:
        Mean best validation MSE across all folds (Optuna minimises this).
    """
    hidden_dims:  tuple = ast.literal_eval(
        trial.suggest_categorical('hidden_dims', SEARCH_SPACE['hidden_dims'])
    )
    dropout_rate: float = trial.suggest_float('dropout_rate', *SEARCH_SPACE['dropout_rate'])
    lr:           float = trial.suggest_float('lr', *SEARCH_SPACE['lr'], log=True)
    weight_decay: float = trial.suggest_float('weight_decay', *SEARCH_SPACE['weight_decay'], log=True)
    batch_size:   int   = trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size'])

    trial_id = trial.number
    run_dir  = os.path.join(sweep_dir, f'run_{trial_id:03d}')
    os.makedirs(run_dir, exist_ok=True)

    sweep_stdout.write(
        f'\n[tp={timepoint_ms:+d} ms | Trial {trial_id}] '
        f'hidden_dims={hidden_dims}  dropout={dropout_rate:.3f}'
        f'  lr={lr:.2e}  wd={weight_decay:.2e}  batch={batch_size}\n'
    )
    sweep_stdout.flush()

    with open(os.path.join(run_dir, 'config.json'), 'w') as _f:
        json.dump(
            {
                'timepoint_ms': timepoint_ms,
                'hidden_dims':  str(hidden_dims),
                'dropout_rate': dropout_rate,
                'lr':           lr,
                'weight_decay': weight_decay,
                'batch_size':   batch_size,
            },
            _f, indent=2,
        )

    # model_type encodes the timepoint so embeddings_dir path lookup is correct
    model_type = f'clip_hba_meg_mem_tp{timepoint_ms}'

    fold_mses: list[float] = []
    fold_rhos: list[float] = []

    try:
        for fold in range(1, N_FOLDS + 1):
            fold_dir = os.path.join(run_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)

            fold_config = {
                'model_type':    model_type,
                'input_dim':     INPUT_DIM,
                'training_data': TRAINING_DATA,
                'fold':          fold,
                'train_csv':     _csv_path('train', fold),
                'val_csv':       _csv_path('val',   fold),
                'test_csv':      _csv_path('test',  fold),
                'img_root':      _IMG_ROOT,
                'embeddings_dir': _EMBEDDINGS_DIR_BASE,
                'memcat_meta_csv': _MEMCAT_META_CSV,
                # MLP head
                'hidden_dims':   hidden_dims,
                'dropout_rate':  dropout_rate,
                # Optimisation
                'lr':            lr,
                'weight_decay':  weight_decay,
                'batch_size':    batch_size,
                'epochs':        300,
                'early_stopping_patience': 20,
                'train_fraction': 1.0,
                'criterion':     nn.MSELoss(),
                # Logging
                'log_path':        os.path.join(fold_dir, 'log.txt'),
                'checkpoint_path': os.path.join(fold_dir, 'checkpoint'),
                'random_seed':     42,
                # Device
                'cuda': int(os.environ.get('CUDA_DEVICE', 1)),
            }
            fold_mse, fold_rho = run_mem_training(fold_config)
            fold_mses.append(fold_mse)
            fold_rhos.append(fold_rho)

            sweep_stdout.write(
                f'  fold {fold}/{N_FOLDS}: mse={fold_mse:.6f}  rho={fold_rho:.4f}\n'
            )
            sweep_stdout.flush()

    except Exception as exc:
        sweep_stdout.write(f'  [ERROR] trial {trial_id} failed: {exc}\n')
        sweep_stdout.flush()
        raise

    mean_mse = sum(fold_mses) / N_FOLDS
    std_mse  = _sample_std(fold_mses)
    mean_rho = sum(fold_rhos) / N_FOLDS
    std_rho  = _sample_std(fold_rhos)

    with open(results_csv, 'a', newline='') as f:
        csv.writer(f).writerow([
            trial_id, str(hidden_dims), f'{dropout_rate:.6f}',
            f'{lr:.2e}', f'{weight_decay:.2e}', batch_size,
            f'{mean_mse:.6f}', f'{std_mse:.6f}',
            f'{mean_rho:.6f}', f'{std_rho:.6f}',
        ])

    sweep_stdout.write(
        f'  -> mean_val_mse = {mean_mse:.6f} ± {std_mse:.6f}'
        f'  |  mean_val_rho = {mean_rho:.4f} ± {std_rho:.4f}\n'
    )
    sweep_stdout.flush()

    return mean_mse


# ---------------------------------------------------------------------------
# Sweep metadata
# ---------------------------------------------------------------------------

def _write_sweep_config(
    sweep_dir: str,
    n_trials: int,
    timepoint_ms: int,
    resumed: bool,
) -> None:
    """Write (or update) sweep_config.json with metadata for this sweep run."""
    import subprocess

    config_path = os.path.join(sweep_dir, 'sweep_config.json')

    try:
        git_hash: 'str | None' = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_hash = None

    timestamp = datetime.datetime.now().isoformat(timespec='seconds')

    if not resumed:
        config: dict = {
            'started_at':          timestamp,
            'n_trials_requested':  n_trials,
            'model_type':          f'clip_hba_meg_mem_tp{timepoint_ms}',
            'timepoint_ms':        timepoint_ms,
            'input_dim':           INPUT_DIM,
            'training_data':       TRAINING_DATA,
            'n_folds':             N_FOLDS,
            'embeddings_dir_base': _EMBEDDINGS_DIR_BASE,
            'search_space': {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in SEARCH_SPACE.items()
            },
            'optuna_sampler': 'TPESampler',
            'python_version': sys.version,
            'git_commit':     git_hash,
            'resume_events':  [],
        }
    else:
        with open(config_path, encoding='utf-8') as _f:
            config = json.load(_f)
        config.setdefault('resume_events', []).append({
            'resumed_at':          timestamp,
            'n_trials_requested':  n_trials,
            'git_commit':          git_hash,
        })

    with open(config_path, 'w', encoding='utf-8') as _f:
        json.dump(config, _f, indent=2)


# ---------------------------------------------------------------------------
# Main sweep function
# ---------------------------------------------------------------------------

def run_meg_sweep(
    timepoint_ms: int,
    n_trials: int = N_TRIALS_DEFAULT,
    resume_db: 'str | None' = None,
) -> None:
    """Run (or resume) a Bayesian hyperparameter search for one MEG timepoint.

    Args:
        timepoint_ms:  Target timepoint in ms (e.g. 100).  Selects the
                       corresponding precomputed embedding files.
        n_trials:      Number of Optuna trials to run.
        resume_db:     Path to an existing optuna_study.db to resume.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_sweep_dir = (
        os.path.dirname(resume_db)
        if resume_db else
        f'./sweep_meg_out/{timestamp}'
    )
    sweep_dir = os.path.join(base_sweep_dir, f'tp{timepoint_ms}')
    os.makedirs(sweep_dir, exist_ok=True)

    _write_sweep_config(sweep_dir, n_trials, timepoint_ms, resumed=resume_db is not None)

    sweep_stdout = sys.stdout

    results_csv = os.path.join(sweep_dir, 'sweep_results.csv')
    db_path     = resume_db if resume_db else os.path.join(sweep_dir, 'optuna_study.db')
    storage_url = f'sqlite:///{db_path}'

    if not os.path.exists(results_csv):
        with open(results_csv, 'w', newline='') as f:
            csv.writer(f).writerow(_CSV_FIELDS)

    study = optuna.create_study(
        study_name=f'clip_hba_meg_mem_tp{timepoint_ms}',
        direction='minimize',
        sampler=TPESampler(seed=42),
        storage=storage_url,
        load_if_exists=True,
    )

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    if n_done > 0:
        sweep_stdout.write(f'[Resume] {n_done} completed trial(s) loaded from {db_path}\n')

    sweep_stdout.write(
        f'\n=== CLIP-HBA-MEG Memorability Sweep | tp={timepoint_ms:+d} ms ===\n'
        f'{n_trials} trial(s) | TPE sampler | {N_FOLDS}-fold CV per trial\n'
        f'Embeddings: {_EMBEDDINGS_DIR_BASE}clip_hba_meg_mem_tp{timepoint_ms}_fold*_*.pt\n'
        f'DB: {db_path}\n'
        f'Search space:\n'
        f'  hidden_dims  (categorical): {SEARCH_SPACE["hidden_dims"]}\n'
        f'  dropout_rate (uniform):     [{SEARCH_SPACE["dropout_rate"][0]}, '
        f'{SEARCH_SPACE["dropout_rate"][1]}]\n'
        f'  lr           (log-uniform): [{SEARCH_SPACE["lr"][0]:.0e}, '
        f'{SEARCH_SPACE["lr"][1]:.0e}]\n'
        f'  weight_decay (log-uniform): [{SEARCH_SPACE["weight_decay"][0]:.0e}, '
        f'{SEARCH_SPACE["weight_decay"][1]:.0e}]\n'
        f'  batch_size   (categorical): {SEARCH_SPACE["batch_size"]}\n\n'
    )
    sweep_stdout.flush()

    study.optimize(
        lambda trial: _objective(trial, timepoint_ms, sweep_dir, results_csv, sweep_stdout),
        n_trials=n_trials,
        catch=(Exception,),
    )

    # --- Leaderboard ---
    all_results = []
    with open(results_csv, newline='') as f:
        for row in csv.DictReader(f):
            try:
                mean_mse = float(row['mean_val_mse'])
                std_mse  = float(row['std_val_mse'])
                mean_rho = float(row['mean_val_rho'])
                std_rho  = float(row['std_val_rho'])
            except ValueError:
                mean_mse = std_mse = mean_rho = std_rho = float('nan')
            all_results.append({
                'trial_id':     int(row['trial_id']),
                'hidden_dims':  row['hidden_dims'],
                'dropout_rate': float(row['dropout_rate']),
                'lr':           float(row['lr']),
                'weight_decay': float(row['weight_decay']),
                'batch_size':   int(row['batch_size']),
                'mean_val_mse': mean_mse,
                'std_val_mse':  std_mse,
                'mean_val_rho': mean_rho,
                'std_val_rho':  std_rho,
            })

    valid = [r for r in all_results if not math.isnan(r['mean_val_mse'])]
    valid.sort(key=lambda r: r['mean_val_mse'])

    sweep_stdout.write('\n' + '=' * 80 + '\n')
    sweep_stdout.write(
        f'Search complete for tp={timepoint_ms:+d} ms. '
        f'Full results: {results_csv}\n'
    )
    header = (
        f"{'Trial':>5}  {'hidden_dims':>18}  {'drop':>5}"
        f"  {'lr':>8}  {'wd':>8}  {'bs':>4}"
        f"  {'mean_mse':>10}  {'std_mse':>9}  {'mean_rho':>9}  {'std_rho':>8}"
    )
    sweep_stdout.write(header + '\n')
    sweep_stdout.write('-' * len(header) + '\n')
    for r in valid[:10]:
        sweep_stdout.write(
            f"{r['trial_id']:>5}  {str(r['hidden_dims']):>18}"
            f"  {r['dropout_rate']:>5.2f}"
            f"  {r['lr']:>8.0e}  {r['weight_decay']:>8.0e}  {r['batch_size']:>4}"
            f"  {r['mean_val_mse']:>10.6f}  {r['std_val_mse']:>9.6f}"
            f"  {r['mean_val_rho']:>9.4f}  {r['std_val_rho']:>8.4f}\n"
        )

    if valid:
        best = valid[0]
        sweep_stdout.write('\n' + '=' * 80 + '\n')
        sweep_stdout.write(
            f'Best configuration for tp={timepoint_ms:+d} ms (trial {best["trial_id"]}):\n'
            f'  hidden_dims:  {best["hidden_dims"]}\n'
            f'  dropout_rate: {best["dropout_rate"]:.4f}\n'
            f'  lr:           {best["lr"]:.2e}\n'
            f'  weight_decay: {best["weight_decay"]:.2e}\n'
            f'  batch_size:   {best["batch_size"]}\n'
            f'  mean_val_mse: {best["mean_val_mse"]:.6f} ± {best["std_val_mse"]:.6f}\n'
            f'  mean_val_rho: {best["mean_val_rho"]:.4f} ± {best["std_val_rho"]:.4f}\n'
        )

    sweep_stdout.write('=' * 80 + '\n')
    sweep_stdout.write(
        f'\nNext step: copy the winning config into train_mem_meg.py '
        f'for tp={timepoint_ms:+d} ms.\n'
        f'The sweep already ran all {N_FOLDS} folds — '
        f'no separate cross-validation run needed.\n'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bayesian hyperparameter search for CLIP-HBA-MEG memorability MLP heads.'
    )
    parser.add_argument(
        '--timepoint_ms',
        type=int,
        required=True,
        help='Target timepoint in ms (e.g. 100, 200, -100).  '
             'Selects precomputed embedding files ending in _tp{timepoint_ms}_fold*_*.pt.',
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=N_TRIALS_DEFAULT,
        metavar='N',
        help=f'Number of Optuna trials to run (default: {N_TRIALS_DEFAULT}).',
    )
    parser.add_argument(
        '--resume',
        metavar='DB',
        default=None,
        help='Path to an existing optuna_study.db to resume a previous search.',
    )
    args = parser.parse_args()
    run_meg_sweep(
        timepoint_ms=args.timepoint_ms,
        n_trials=args.n_trials,
        resume_db=args.resume,
    )
