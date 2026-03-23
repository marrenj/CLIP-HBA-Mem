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
ALL_TIMEPOINTS: list[int] = list(range(-100, 1301, 100))  # 15 timepoints
INPUT_DIM:       int       = 66
MODEL_TYPE_BASE: str       = 'clip_hba_meg_mem'


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _find_checkpoint(
    checkpoint_dir: str,
    timepoint_ms: int,
    fold: int,
) -> str:
    """Find the best checkpoint for a given timepoint and fold.

    Searches ``checkpoint_dir`` for files matching
    ``tp{tp}_fold{fold}_*.pth`` and returns the most recently modified one.
    Raises ``FileNotFoundError`` if none is found.

    Args:
        checkpoint_dir:  Directory containing ``.pth`` checkpoint files.
        timepoint_ms:    Timepoint in ms.
        fold:            Cross-validation fold index.

    Returns:
        Absolute path to the checkpoint file.
    """
    pattern = os.path.join(checkpoint_dir, f'tp{timepoint_ms}_fold{fold}_*.pth')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f'No checkpoint found for tp={timepoint_ms} ms, fold={fold} '
            f'in {checkpoint_dir!r}.\n'
            f'  Pattern searched: {pattern}\n'
            f'  Run train_mem_meg.py first.'
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
    hidden_dims: tuple,
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
    print(f'[tp={timepoint_ms:+d} ms | fold {fold}]  '
          f'{len(dataset)} images  |  checkpoint: {os.path.basename(checkpoint_path)}')

    model = MLPOnlyHead(
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        input_dim=INPUT_DIM,
    )
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
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

    summary_rows = []

    for tp in timepoints:
        for fold in folds:
            try:
                row = _run_single(
                    timepoint_ms=tp,
                    fold=fold,
                    embeddings_dir=config['embeddings_dir'],
                    checkpoint_dir=config['checkpoint_dir'],
                    hidden_dims=tuple(config['hidden_dims']),
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
    parser = argparse.ArgumentParser(
        description='Inference with trained CLIP-HBA-MEG memorability MLP heads.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--timepoint_ms',
        default='0',
        help='Timepoint in ms (e.g. 100) or "all" to run all 15 timepoints.',
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
             'Defaults to ./Data/{training_data}/meg_embeddings/.',
    )
    parser.add_argument(
        '--checkpoint_dir',
        default=None,
        help='Directory with trained .pth MLP head checkpoints.  '
             'Defaults to ./models/clip_hba_meg_mem/.',
    )
    parser.add_argument(
        '--hidden_dims',
        nargs='+', type=int, default=[32, 16],
        help='MLP hidden layer sizes.  Must match the trained checkpoint.',
    )
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate.  Must match the trained checkpoint.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda:0',
                        help='Device string (e.g. cuda:0, cuda:1, cpu).')
    parser.add_argument('--output_dir', default='./preds/meg_mem/',
                        help='Directory to save prediction CSVs and summary.')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # Resolve training_data-specific defaults
    training_data = args.training_data
    if training_data == 'lamem':
        n_folds = 5
        default_emb_dir = './Data/lamem/meg_embeddings/'
    else:
        n_folds = 10
        default_emb_dir = './Data/combined_lamem_memcat/meg_embeddings/'

    embeddings_dir = args.embeddings_dir or default_emb_dir
    checkpoint_dir = args.checkpoint_dir or './models/clip_hba_meg_mem/'

    # Resolve timepoints
    if args.timepoint_ms == 'all':
        timepoints = 'all'
    else:
        tp = int(args.timepoint_ms)
        if tp not in ALL_TIMEPOINTS:
            raise ValueError(
                f'--timepoint_ms {tp} is not in {ALL_TIMEPOINTS}.'
            )
        timepoints = [tp]

    # Resolve folds
    if args.fold == 'all':
        folds = list(range(1, n_folds + 1))
    else:
        folds = [int(args.fold)]

    config = {
        'timepoints':    timepoints,
        'folds':         folds,
        'training_data': training_data,
        'embeddings_dir': embeddings_dir,
        'checkpoint_dir': checkpoint_dir,
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
