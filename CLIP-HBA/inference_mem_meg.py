"""Run memorability inference with trained CLIP-HBA-MEG memorability head(s).

Loads precomputed 66-dim CLIP-HBA-MEG embeddings and a trained MLPOnlyHead
checkpoint for one or all sampled timepoints, then produces per-image
memorability prediction CSVs together with a summary of Spearman rho and MSE.

Usage
-----
    cd CLIP-HBA

    # Single timepoint, fold 1:
    python inference_mem_meg.py --timepoint_ms 100 --fold 1

    # All 15 timepoints, fold 1:
    python inference_mem_meg.py --timepoint_ms all --fold 1

    # Custom checkpoint directory:
    python inference_mem_meg.py \\
        --timepoint_ms 200 --fold 1 \\
        --checkpoint_dir ./models/clip_hba_meg_mem/ \\
        --checkpoint_pattern tp{tp}_fold{fold}_{ts}.pth

    # Combined LaMem + MemCat test sets:
    python inference_mem_meg.py \\
        --timepoint_ms all --fold all \\
        --training_data combined_lamem_memcat

    # THINGS dataset (uses a long-format CSV of THINGS MEG embeddings):
    python inference_mem_meg.py \\
        --dataset things --fold all \\
        --timepoint_ms 50 150 250 350 500 900 1300 \\
        --things_emb_csv path/to/things_meg_embeddings.csv

Output files
------------
    <output_dir>/<timestamp>/
        tp{tp}_fold{fold}_predictions.csv  — image_path, pred_score, true_score
        summary.csv                         — per-timepoint/fold Spearman rho & MSE
"""

import argparse
import csv
import datetime
import glob
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions.train_mem_pipeline import EmbeddingDataset, MLPOnlyHead
from functions.train_behavior_things_pipeline import seed_everything

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_TIMEPOINTS: list[int] = list(range(-100, 1301, 5))   # 281 timepoints at 5 ms resolution
INPUT_DIM:       int       = 66
MODEL_TYPE_BASE: str       = 'clip_hba_meg_mem'


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _infer_hidden_dims_from_state_dict(state_dict: dict) -> tuple[int, ...]:
    """Infer ``MLPOnlyHead`` hidden_dims from a checkpoint state_dict.

    ``MLPOnlyHead`` builds ``nn.Sequential`` of repeating ``[Linear, Act, Dropout]``
    blocks followed by a final ``Linear(_, 1)``.  Linear layers therefore live
    at module indices ``0, 3, 6, ...`` with weight shape ``[out_dim, in_dim]``.
    Hidden dims are the output sizes of every linear layer **except** the last.

    Returns an empty tuple if the head has only one linear layer (input → 1).
    Raises ``ValueError`` if no ``mlp_head.*.weight`` keys are found.
    """
    linear_out_dims: list[tuple[int, int]] = []
    for key, tensor in state_dict.items():
        if key.startswith('mlp_head.') and key.endswith('.weight') and tensor.ndim == 2:
            try:
                idx = int(key.split('.')[1])
            except ValueError:
                continue
            linear_out_dims.append((idx, tensor.shape[0]))

    if not linear_out_dims:
        raise ValueError(
            'Could not infer hidden_dims: no mlp_head.*.weight keys found in '
            'state_dict.  Was this checkpoint trained with MLPOnlyHead?'
        )

    linear_out_dims.sort(key=lambda pair: pair[0])
    out_dims = [d for _, d in linear_out_dims]
    # Drop the final Linear(_, 1) — that's the regression head, not a hidden layer.
    return tuple(out_dims[:-1])


def _find_checkpoint(
    checkpoint_dir: str,
    timepoint_ms: int,
    fold: int,
) -> str:
    """Find the best checkpoint for a given timepoint and fold.

    Tries several known layouts (flat or nested under ``tp{tp}/``) and
    matches both ``tp{tp}_fold{fold}_*.pth`` and
    ``*tp{tp}_*fold{fold}_*.pth`` filename styles.  Falls back to a
    recursive search.  Returns the most recently modified match.
    Raises ``FileNotFoundError`` if none is found.

    Args:
        checkpoint_dir:  Directory containing ``.pth`` checkpoint files.
        timepoint_ms:    Timepoint in ms.
        fold:            Cross-validation fold index.

    Returns:
        Absolute path to the checkpoint file.
    """
    candidate_patterns = [
        # Flat layout, original naming
        os.path.join(checkpoint_dir, f'tp{timepoint_ms}_fold{fold}_*.pth'),
        # Flat layout, *tp{tp}_*fold{fold}_*.pth (e.g. clip_hba_meg_mem_tp50_final_fold1_*.pth)
        os.path.join(checkpoint_dir, f'*tp{timepoint_ms}_*fold{fold}_*.pth'),
        # Nested under tp{tp}/ subdirectory
        os.path.join(checkpoint_dir, f'tp{timepoint_ms}', f'*fold{fold}_*.pth'),
        os.path.join(checkpoint_dir, f'tp{timepoint_ms}', f'*tp{timepoint_ms}_*fold{fold}_*.pth'),
        # Recursive fallback
        os.path.join(checkpoint_dir, '**', f'*tp{timepoint_ms}_*fold{fold}_*.pth'),
    ]

    matches: list[str] = []
    tried: list[str] = []
    for pattern in candidate_patterns:
        tried.append(pattern)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            break

    if not matches:
        tried_lines = '\n    '.join(tried)
        raise FileNotFoundError(
            f'No checkpoint found for tp={timepoint_ms} ms, fold={fold} '
            f'in {checkpoint_dir!r}.\n'
            f'  Patterns tried:\n    {tried_lines}\n'
            f'  Run train_mem_meg.py first, or check the layout.'
        )
    # Return the most recently modified file (handles multiple timestamps)
    return max(matches, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Single-timepoint inference
# ---------------------------------------------------------------------------

def _run_single(
    timepoint_ms: int,
    fold: int,
    embeddings_dir: str,
    checkpoint_dir: str,
    hidden_dims: tuple | None,
    dropout_rate: float,
    device: torch.device,
    batch_size: int,
    output_dir: pathlib.Path,
    training_data: str,
) -> dict:
    """Run inference for one timepoint / fold combination.

    Loads the precomputed MEG embedding file for ``split='test'`` and the
    corresponding trained MLPOnlyHead checkpoint, then computes predictions.

    Args:
        timepoint_ms:   Timepoint in ms.
        fold:           Cross-validation fold index.
        embeddings_dir: Directory with precomputed ``.pt`` embedding files.
        checkpoint_dir: Directory with trained ``.pth`` MLP head checkpoints.
        hidden_dims:    MLP hidden layer sizes (must match the checkpoint).
        dropout_rate:   MLP dropout rate (must match the checkpoint).
        device:         Torch device.
        batch_size:     Inference batch size.
        output_dir:     Directory to write prediction CSV.
        training_data:  ``'lamem'`` or ``'combined_lamem_memcat'``.

    Returns:
        Summary dict with ``timepoint_ms``, ``fold``, ``spearman_rho``,
        ``spearman_p``, ``mse``, ``pred_std``, ``n_images``, ``checkpoint``.
    """
    model_type = f'{MODEL_TYPE_BASE}_tp{timepoint_ms}'
    emb_path = pathlib.Path(embeddings_dir) / f'{model_type}_fold{fold}_test.pt'

    if not emb_path.exists():
        raise FileNotFoundError(
            f'Embedding file not found: {emb_path}\n'
            f'  Run extract_embeddings_meg.py first.'
        )

    dataset = EmbeddingDataset(str(emb_path))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint_path = _find_checkpoint(checkpoint_dir, timepoint_ms, fold)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Auto-infer hidden_dims from the checkpoint shapes when the user
    # didn't pass --hidden_dims explicitly (hidden_dims is None then).
    effective_hidden_dims = (
        hidden_dims if hidden_dims is not None
        else _infer_hidden_dims_from_state_dict(state_dict)
    )

    print(f'[tp={timepoint_ms:+d} ms | fold {fold}]  '
          f'{len(dataset)} images  |  checkpoint: {os.path.basename(checkpoint_path)}  |  '
          f'hidden_dims={effective_hidden_dims}')

    model = MLPOnlyHead(
        hidden_dims=effective_hidden_dims,
        dropout_rate=dropout_rate,
        input_dim=INPUT_DIM,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    all_paths:   list = []
    all_preds:   list = []
    all_targets: list = []

    with torch.no_grad():
        for image_paths, embeddings, scores in tqdm(loader, desc=f'  tp={timepoint_ms:+d}'):
            embeddings = embeddings.to(device)
            preds = model(embeddings).squeeze(1).cpu().numpy()
            all_paths.extend(image_paths)
            all_preds.extend(preds)
            all_targets.extend(scores.numpy())

    all_preds   = np.array(all_preds,   dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)

    rho, p_val = spearmanr(all_preds, all_targets)
    mse        = float(np.mean((all_preds - all_targets) ** 2))
    pred_std   = float(all_preds.std())

    print(f'  Spearman rho: {rho:.4f} (p={p_val:.2e})  |  MSE: {mse:.6f}  |  '
          f'Pred range: [{all_preds.min():.4f}, {all_preds.max():.4f}]  std: {pred_std:.4f}')

    # Save per-image predictions
    pred_csv = output_dir / f'tp{timepoint_ms}_fold{fold}_predictions.csv'
    pd.DataFrame({
        'image_path':   all_paths,
        'pred_score':   all_preds,
        'true_score':   all_targets,
        'timepoint_ms': timepoint_ms,
        'fold':         fold,
    }).to_csv(pred_csv, index=False)
    print(f'  -> {pred_csv}')

    model.cpu()
    del model

    return {
        'timepoint_ms': timepoint_ms,
        'fold':         fold,
        'n_images':     len(dataset),
        'spearman_rho': rho,
        'spearman_p':   p_val,
        'mse':          mse,
        'pred_std':     pred_std,
        'checkpoint':   checkpoint_path,
    }


# ---------------------------------------------------------------------------
# THINGS inference (CSV-based embeddings)
# ---------------------------------------------------------------------------

# Module-level caches so the (large) THINGS CSVs are read only once
# even when iterating over many timepoints and folds.
_THINGS_EMB_CACHE: dict[str, pd.DataFrame] = {}
_THINGS_SCORES_CACHE: dict[str, pd.DataFrame] = {}


def _load_things_emb_csv(things_emb_csv: str) -> pd.DataFrame:
    """Load and cache the THINGS MEG embeddings CSV."""
    if things_emb_csv not in _THINGS_EMB_CACHE:
        print(f'[THINGS] Loading embeddings CSV: {things_emb_csv}')
        _THINGS_EMB_CACHE[things_emb_csv] = pd.read_csv(things_emb_csv)
    return _THINGS_EMB_CACHE[things_emb_csv]


def _load_things_scores_csv(things_scores_csv: str) -> pd.DataFrame:
    """Load and cache the THINGS memorability scores CSV.

    Returns a DataFrame with columns ``image_name`` and ``cr``,
    with the image_name derived from the file_path basename.
    """
    if things_scores_csv not in _THINGS_SCORES_CACHE:
        print(f'[THINGS] Loading scores CSV:     {things_scores_csv}')
        scores_df = pd.read_csv(things_scores_csv)
        # Derive image_name from file_path (e.g. "images/knot/knot_12s.jpg" -> "knot_12s.jpg")
        scores_df['image_name'] = scores_df['file_path'].apply(
            lambda p: pathlib.PurePosixPath(p).name
        )
        scores_df = scores_df[['image_name', 'cr']].dropna(subset=['cr'])
        _THINGS_SCORES_CACHE[things_scores_csv] = scores_df
    return _THINGS_SCORES_CACHE[things_scores_csv]


def _run_things_single(
    timepoint_ms: int,
    fold: int,
    things_emb_csv: str,
    things_scores_csv: str,
    checkpoint_dir: str,
    hidden_dims: tuple | None,
    dropout_rate: float,
    device: torch.device,
    batch_size: int,
    output_dir: pathlib.Path,
) -> dict:
    """Run inference for one timepoint / fold on THINGS images.

    Loads MEG embeddings from a long-format CSV, joins with THINGS
    memorability scores, and runs the trained MLP head.

    Args:
        timepoint_ms:      Timepoint in ms.
        fold:              Cross-validation fold index.
        things_emb_csv:    Path to THINGS MEG embeddings CSV
                           (columns: image_name, timepoint_ms, dim_0..dim_65).
        things_scores_csv: Path to THINGS_Memorability_Scores.csv.
        checkpoint_dir:    Directory with trained ``.pth`` MLP head checkpoints.
        hidden_dims:       MLP hidden layer sizes (must match checkpoint).
        dropout_rate:      MLP dropout rate (must match checkpoint).
        device:            Torch device.
        batch_size:        Inference batch size.
        output_dir:        Directory to write prediction CSV.

    Returns:
        Summary dict with ``timepoint_ms``, ``fold``, ``spearman_rho``,
        ``spearman_p``, ``mse``, ``pred_std``, ``n_images``, ``checkpoint``.
    """
    # Resolve checkpoint first so we fail fast before doing any CSV work.
    checkpoint_path = _find_checkpoint(checkpoint_dir, timepoint_ms, fold)

    # Load (cached) embeddings CSV and filter to the requested timepoint
    emb_df = _load_things_emb_csv(things_emb_csv)
    if 'timepoint_ms' not in emb_df.columns:
        raise KeyError(
            f"'timepoint_ms' column not found in {things_emb_csv}.  "
            f"Available columns: {list(emb_df.columns)[:10]}{'...' if len(emb_df.columns) > 10 else ''}.  "
            f"Expected long-format CSV with image_name, timepoint_ms, dim_0..dim_65."
        )
    tp_df = emb_df[emb_df['timepoint_ms'] == timepoint_ms].copy()
    if tp_df.empty:
        raise ValueError(
            f'Timepoint {timepoint_ms} ms not found in {things_emb_csv}. '
            f'Available: {sorted(emb_df["timepoint_ms"].unique().tolist())}'
        )

    # Load (cached) THINGS memorability scores and join
    scores_df = _load_things_scores_csv(things_scores_csv)

    merged = tp_df.merge(scores_df, on='image_name', how='inner')
    n_images = len(merged)
    if n_images == 0:
        raise ValueError(
            'No images matched between MEG embeddings CSV and THINGS scores CSV. '
            'Check image_name formats.'
        )

    # Extract tensors
    dim_cols = [c for c in merged.columns if c.startswith('dim_')]
    embeddings = torch.tensor(merged[dim_cols].values, dtype=torch.float32)
    scores = torch.tensor(merged['cr'].values, dtype=torch.float32)
    image_names = merged['image_name'].tolist()

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Auto-infer hidden_dims from the checkpoint shapes when the user
    # didn't pass --hidden_dims explicitly (hidden_dims is None then).
    effective_hidden_dims = (
        hidden_dims if hidden_dims is not None
        else _infer_hidden_dims_from_state_dict(state_dict)
    )

    print(f'[THINGS | tp={timepoint_ms:+d} ms | fold {fold}]  '
          f'{n_images} images  |  checkpoint: {os.path.basename(checkpoint_path)}  |  '
          f'hidden_dims={effective_hidden_dims}')

    model = MLPOnlyHead(
        hidden_dims=effective_hidden_dims,
        dropout_rate=dropout_rate,
        input_dim=INPUT_DIM,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # Run inference in batches
    all_preds: list = []
    with torch.no_grad():
        for i in range(0, n_images, batch_size):
            batch_emb = embeddings[i:i + batch_size].to(device)
            preds = model(batch_emb).squeeze(1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds, dtype=np.float32)
    all_targets = scores.numpy()

    rho, p_val = spearmanr(all_preds, all_targets)
    mse = float(np.mean((all_preds - all_targets) ** 2))
    pred_std = float(all_preds.std())

    print(f'  Spearman rho: {rho:.4f} (p={p_val:.2e})  |  MSE: {mse:.6f}  |  '
          f'Pred range: [{all_preds.min():.4f}, {all_preds.max():.4f}]  std: {pred_std:.4f}')

    # Save per-image predictions
    pred_csv = output_dir / f'things_tp{timepoint_ms}_fold{fold}_predictions.csv'
    pd.DataFrame({
        'image_path': image_names,
        'pred_score': all_preds,
        'true_score': all_targets,
        'timepoint_ms': timepoint_ms,
        'fold': fold,
    }).to_csv(pred_csv, index=False)
    print(f'  -> {pred_csv}')

    model.cpu()
    del model

    return {
        'timepoint_ms': timepoint_ms,
        'fold': fold,
        'n_images': n_images,
        'spearman_rho': rho,
        'spearman_p': p_val,
        'mse': mse,
        'pred_std': pred_std,
        'checkpoint': checkpoint_path,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_meg_inference(config: dict) -> None:
    """Run inference for one or all CLIP-HBA-MEG memorability timepoints.

    Args:
        config: Configuration dict (see ``main()`` for keys).
    """
    seed_everything(config['random_seed'])

    timestamp  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = pathlib.Path(config['output_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config['device'])

    timepoints = ALL_TIMEPOINTS if config['timepoints'] == 'all' else config['timepoints']
    folds      = config['folds']
    dataset    = config.get('dataset', 'default')

    summary_rows = []

    # None signals "auto-infer per checkpoint"; otherwise pass the explicit tuple.
    hidden_dims_cfg = config['hidden_dims']
    hidden_dims_arg = tuple(hidden_dims_cfg) if hidden_dims_cfg is not None else None

    for tp in timepoints:
        for fold in folds:
            try:
                if dataset == 'things':
                    row = _run_things_single(
                        timepoint_ms=tp,
                        fold=fold,
                        things_emb_csv=config['things_emb_csv'],
                        things_scores_csv=config['things_scores_csv'],
                        checkpoint_dir=config['checkpoint_dir'],
                        hidden_dims=hidden_dims_arg,
                        dropout_rate=config['dropout_rate'],
                        device=device,
                        batch_size=config['batch_size'],
                        output_dir=output_dir,
                    )
                else:
                    row = _run_single(
                        timepoint_ms=tp,
                        fold=fold,
                        embeddings_dir=config['embeddings_dir'],
                        checkpoint_dir=config['checkpoint_dir'],
                        hidden_dims=hidden_dims_arg,
                        dropout_rate=config['dropout_rate'],
                        device=device,
                        batch_size=config['batch_size'],
                        output_dir=output_dir,
                        training_data=config['training_data'],
                    )
                summary_rows.append(row)
            except FileNotFoundError as exc:
                print(f'[WARNING] Skipping tp={tp} ms, fold={fold}: {exc}')

    if not summary_rows:
        print('[ERROR] No results produced — check embedding and checkpoint paths.')
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'summary.csv'
    summary_df.to_csv(summary_path, index=False)

    print(f'\n{"=" * 60}')
    print(f'Summary saved to: {summary_path}')
    print(summary_df[['timepoint_ms', 'fold', 'n_images',
                       'spearman_rho', 'mse', 'pred_std']].to_string(index=False))
    print('=' * 60)


def main() -> None:
    _data_dir_default = os.environ.get('DATA_DIR', './Data')

    parser = argparse.ArgumentParser(
        description='Inference with trained CLIP-HBA-MEG memorability MLP heads.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--data_dir',
        default=_data_dir_default,
        help='Root data directory (replaces the ./Data prefix). '
             'Also settable via DATA_DIR env var.',
    )
    parser.add_argument(
        '--dataset',
        default='default',
        choices=['default', 'things'],
        help='Dataset to run inference on. '
             '"default" uses precomputed lamem/combined .pt embeddings (per fold). '
             '"things" uses a long-format CSV of THINGS MEG embeddings '
             '(--things_emb_csv) and is fold-agnostic at the embedding step '
             '(same embeddings are reused with each fold\'s MLP head).',
    )
    parser.add_argument(
        '--timepoint_ms',
        nargs='+',
        default=['0'],
        help='Timepoint(s) in ms (e.g. 50 150 250) or "all".',
    )
    parser.add_argument(
        '--fold',
        default='1',
        help='Fold index or "all" to run all folds '
             '(1–5 for lamem; 1–10 for combined_lamem_memcat).',
    )
    parser.add_argument(
        '--training_data',
        default=os.environ.get('TRAINING_DATA', 'combined_lamem_memcat'),
        choices=['lamem', 'combined_lamem_memcat'],
    )
    parser.add_argument(
        '--embeddings_dir',
        default=None,
        help='Directory with precomputed MEG .pt embedding files.  '
             'Defaults to <data_dir>/{training_data}/meg_embeddings/. '
             '(Ignored when --dataset things — use --things_emb_csv.)',
    )
    parser.add_argument(
        '--things_emb_csv',
        default=None,
        help='Path to THINGS MEG embeddings CSV '
             '(long-format columns: image_name, timepoint_ms, dim_0..dim_65). '
             'Required when --dataset things.',
    )
    parser.add_argument(
        '--things_scores_csv',
        default=None,
        help='Path to THINGS_Memorability_Scores.csv. '
             'Defaults to <data_dir>/THINGS_Memorability_Scores.csv.',
    )
    parser.add_argument(
        '--checkpoint_dir',
        default=None,
        help='Directory with trained .pth MLP head checkpoints.  '
             'Defaults to ./models/clip_hba_meg_mem/.',
    )
    parser.add_argument(
        '--hidden_dims',
        nargs='+', type=int, default=None,
        help='MLP hidden layer sizes.  If omitted, the dims are auto-inferred '
             'from each checkpoint\'s state_dict shapes (per-timepoint), so '
             'checkpoints with different MLP sizes can be loaded in one run.',
    )
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='Dropout rate used to construct the MLP.  '
                             'IGNORED at inference because model.eval() disables '
                             'dropout — kept only for parameter compatibility.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0',
                        help='Device string (e.g. cuda:0, cuda:1, cpu).')
    parser.add_argument('--output_dir', default='./preds/meg_mem/',
                        help='Directory to save prediction CSVs and summary.')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Resolve training_data-specific defaults
    training_data = args.training_data
    data_dir = args.data_dir
    dataset = args.dataset

    if training_data == 'lamem':
        n_folds = 5
        default_emb_dir = f'{data_dir}/lamem/meg_embeddings/'
    else:
        n_folds = 10
        default_emb_dir = f'{data_dir}/combined_lamem_memcat/meg_embeddings/'

    embeddings_dir = args.embeddings_dir or default_emb_dir
    checkpoint_dir = args.checkpoint_dir or './models/clip_hba_meg_mem/'

    # Resolve timepoints
    if args.timepoint_ms == ['all']:
        timepoints = 'all'
    else:
        timepoints = []
        for tp_str in args.timepoint_ms:
            tp = int(tp_str)
            if tp not in ALL_TIMEPOINTS:
                raise ValueError(
                    f'--timepoint_ms {tp} is not in the valid range '
                    f'({ALL_TIMEPOINTS[0]}..{ALL_TIMEPOINTS[-1]} at 5 ms steps).'
                )
            timepoints.append(tp)

    # Resolve folds
    if args.fold == 'all':
        folds = list(range(1, n_folds + 1))
    else:
        folds = [int(args.fold)]

    # THINGS-specific validation
    if dataset == 'things' and args.things_emb_csv is None:
        raise ValueError('--things_emb_csv is required when --dataset things.')

    things_scores_csv = (
        args.things_scores_csv or f'{data_dir}/THINGS_Memorability_Scores.csv'
    )

    config = {
        'dataset':       dataset,
        'timepoints':    timepoints,
        'folds':         folds,
        'training_data': training_data,
        'embeddings_dir': embeddings_dir,
        'checkpoint_dir': checkpoint_dir,
        'things_emb_csv': args.things_emb_csv,
        'things_scores_csv': things_scores_csv,
        'hidden_dims':   args.hidden_dims,
        'dropout_rate':  args.dropout_rate,
        'batch_size':    args.batch_size,
        'device':        args.device,
        'output_dir':    args.output_dir,
        'random_seed':   args.seed,
    }

    run_meg_inference(config)


if __name__ == '__main__':
    main()
