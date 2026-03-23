"""Pre-extract 66-dim CLIP-HBA-MEG embeddings at each sampled timepoint.

The CLIP-HBA-MEG model produces a 66-dimensional semantic embedding at every
sampled timepoint via its alpha/beta/noise parameters and weighted ViT-layer
aggregation.  Because the MEG backbone is fully frozen during memorability
training, its outputs are deterministic (eval mode suppresses stochastic noise)
and can be pre-computed once per fold, yielding a large speedup during the
Bayesian hyperparameter sweep and per-timepoint MLP head training.

Timepoint configuration (defaults)
------------------------------------
  * Every 100 ms from -100 ms to 1300 ms  →  15 sampled timepoints
  * 300 ms averaging window  →  train_window_size = 150  (±150 ms half-width)
  * Full model resolution:  ms_start=-100, ms_end=1300, ms_step=5

Output files
------------
For each timepoint ``tp`` (in ms) and each data split:

    <out_dir>/clip_hba_meg_mem_tp{tp}_fold{fold}_{split}.pt

Each ``.pt`` file is a dict::

    embeddings  – Tensor[N, 66]   float32 MEG semantic embeddings
    scores      – Tensor[N]       float32 memorability scores in [0, 1]
    image_paths – list[N]         relative image path strings (from CSV)

Usage
-----
    # Single fold (combined LaMem + MemCat):
    python extract_embeddings_meg.py \\
        --meg_checkpoint ./models/cliphba_meg_group.pth \\
        --fold 1 --cuda 0

    # LaMem only:
    python extract_embeddings_meg.py \\
        --meg_checkpoint ./models/cliphba_meg_group.pth \\
        --training_data lamem --fold 1 --cuda 0

    # All folds:
    for FOLD in $(seq 1 10); do
        python extract_embeddings_meg.py \\
            --meg_checkpoint ./models/cliphba_meg_group.pth \\
            --fold $FOLD --cuda 0
    done
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append('../')
from functions.spose_dimensions import classnames66
from functions.inference_meg_group_pipeline import (
    CLIPHBA,
    apply_dora_to_ViT,
    seed_everything,
)
from functions.train_mem_pipeline import MemDataset

# ---------------------------------------------------------------------------
# Timepoint configuration
# ---------------------------------------------------------------------------
MEG_MS_START:        int = -100   # full model temporal range (must match checkpoint)
MEG_MS_END:          int = 1300
MEG_MS_STEP:         int = 5

EXTRACT_START:       int = -100   # sampled timepoints for extraction
EXTRACT_END:         int = 1300
EXTRACT_STEP:        int = 100    # every 100 ms  →  15 timepoints

TRAIN_WINDOW_SIZE:   int = 150    # ±150 ms half-window  →  300 ms total averaging window

MODEL_TYPE_BASE:     str = 'clip_hba_meg_mem'


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _build_meg_model(
    meg_checkpoint: str,
    vision_layers: int,
    transformer_layers: int,
    rank: int,
) -> CLIPHBA:
    """Instantiate and load a pretrained CLIP-HBA-MEG model for embedding extraction.

    The model is configured to sample timepoints every EXTRACT_STEP ms from
    EXTRACT_START to EXTRACT_END with a TRAIN_WINDOW_SIZE ms half-window.
    DoRA is applied with the same architecture as inference_meg_group_pipeline.py.

    Args:
        meg_checkpoint:      Path to the trained CLIP-HBA-MEG state dict.
        vision_layers:       Number of ViT layers with DoRA (must match checkpoint).
        transformer_layers:  Number of text-transformer layers with DoRA.
        rank:                DoRA rank (must match checkpoint).

    Returns:
        CLIPHBA model in eval mode, all parameters frozen.
    """
    classnames = classnames66  # list of 66 semantic dimension strings

    pos_embedding = True  # ViT-L/14 uses positional embeddings

    model = CLIPHBA(
        classnames=classnames,
        weighting_matrix=None,
        backbone_name='ViT-L/14',
        pos_embedding=pos_embedding,
        ms_start=MEG_MS_START,
        ms_step=MEG_MS_STEP,
        ms_end=MEG_MS_END,
        train_start=EXTRACT_START,
        train_step=EXTRACT_STEP,
        train_end=EXTRACT_END,
        train_window_size=TRAIN_WINDOW_SIZE,
    )

    # Apply DoRA layers (same ordering as inference_meg_group_pipeline.py)
    apply_dora_to_ViT(
        model,
        n_vision_layers=0,
        n_transformer_layers=transformer_layers,
        r=32,
        dora_dropout=0.1,
    )
    apply_dora_to_ViT(
        model,
        n_vision_layers=vision_layers,
        n_transformer_layers=0,
        r=rank,
        dora_dropout=0.1,
    )

    print(f'[MEG] Loading checkpoint: {meg_checkpoint}')
    state_dict = torch.load(meg_checkpoint, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print('[MEG] Checkpoint loaded successfully.')

    # Freeze all parameters — backbone is always frozen during memorability training
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-fold extraction
# ---------------------------------------------------------------------------

def extract_meg_embeddings_for_fold(
    meg_checkpoint: str,
    fold: int,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    img_root: 'str | dict',
    out_dir: str,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 8,
    vision_layers: int = 24,
    transformer_layers: int = 1,
    rank: int = 32,
    memcat_meta_csv: 'str | None' = None,
) -> None:
    """Extract and save 66-dim MEG embeddings for every timepoint and split.

    For each of the 15 sampled timepoints (EXTRACT_START to EXTRACT_END in steps
    of EXTRACT_STEP), one ``.pt`` file is saved per split.  Files that already
    exist are skipped so the script is safe to re-run after interruption.

    Args:
        meg_checkpoint:     Path to the trained CLIP-HBA-MEG state dict.
        fold:               Fold index used in output filenames.
        train_csv:          Path to train split CSV.
        val_csv:            Path to val split CSV.
        test_csv:           Path to test split CSV.
        img_root:           Image root directory (str or dict for combined datasets).
        out_dir:            Directory where ``.pt`` files are written.
        device:             Torch device for backbone inference.
        batch_size:         Images per forward pass.
        num_workers:        DataLoader worker processes.
        vision_layers:      ViT DoRA layers (must match checkpoint).
        transformer_layers: Text-transformer DoRA layers (must match checkpoint).
        rank:               DoRA rank (must match checkpoint).
        memcat_meta_csv:    Path to memcat_image_data.csv (combined dataset only).
    """
    out_dir_path = pathlib.Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Determine which timepoints will be produced by the model forward pass
    sampled_tps = list(range(EXTRACT_START, EXTRACT_END + 1, EXTRACT_STEP))
    n_timepoints = len(sampled_tps)

    # Build output path template — timepoint is filled in the inner loop
    def _out_path(tp_ms: int, split: str) -> pathlib.Path:
        return out_dir_path / f'{MODEL_TYPE_BASE}_tp{tp_ms}_fold{fold}_{split}.pt'

    # Check if all files already exist (fold-level skip)
    splits = {'train': train_csv, 'val': val_csv, 'test': test_csv}
    all_exist = all(
        _out_path(tp, split).exists()
        for tp in sampled_tps
        for split in splits
    )
    if all_exist:
        print(f'[Skip] All files for fold {fold} already exist — delete them to re-extract.')
        return

    model = _build_meg_model(meg_checkpoint, vision_layers, transformer_layers, rank)
    model.to(device)

    for split_name, csv_path in splits.items():
        # Check if all timepoints for this split already exist
        if all(_out_path(tp, split_name).exists() for tp in sampled_tps):
            print(f'[Skip] All timepoints for fold {fold} / {split_name} exist.')
            continue

        dataset = MemDataset(
            csv_file=csv_path,
            img_root=img_root,
            memcat_meta_csv=memcat_meta_csv,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        n_images = len(dataset)
        print(f'\n[MEG | fold {fold} | {split_name}]  {n_images} images  '
              f'->  {n_timepoints} timepoints  |  out: {out_dir_path}')

        # Accumulate embeddings for all timepoints simultaneously
        # Shape: [n_timepoints, N, 66]
        all_embeddings_per_tp: list[list] = [[] for _ in range(n_timepoints)]
        all_scores:      list = []
        all_image_paths: list = []

        with torch.no_grad(), tqdm(loader, desc=f'  {split_name}') as pbar:
            for image_paths, images, scores in pbar:
                images = images.to(device)
                # Forward pass: pred_emb_3d shape [n_timepoints, batch, 66]
                pred_emb_3d, _, _ = model(images)
                pred_emb_3d = pred_emb_3d.float().cpu()

                for t_idx in range(n_timepoints):
                    all_embeddings_per_tp[t_idx].append(pred_emb_3d[t_idx])  # [batch, 66]

                all_scores.append(scores.cpu())
                all_image_paths.extend(image_paths)

        all_scores_tensor = torch.cat(all_scores, dim=0)  # [N]

        for t_idx, tp_ms in enumerate(sampled_tps):
            out_path = _out_path(tp_ms, split_name)
            if out_path.exists():
                print(f'  [Skip] {out_path.name} already exists.')
                continue

            embeddings_tensor = torch.cat(all_embeddings_per_tp[t_idx], dim=0)  # [N, 66]
            payload = {
                'embeddings':  embeddings_tensor,    # Tensor[N, 66]
                'scores':      all_scores_tensor,    # Tensor[N]
                'image_paths': all_image_paths,      # list[N]
            }
            torch.save(payload, out_path)
            print(f'  tp={tp_ms:+5d} ms  ->  {out_path.name}  '
                  f'(shape {tuple(embeddings_tensor.shape)})')

    # Release GPU memory before next fold
    model.cpu()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pre-extract 66-dim CLIP-HBA-MEG embeddings at each sampled timepoint.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--meg_checkpoint',
        default=os.environ.get('MEG_CHECKPOINT', './models/cliphba_meg_group.pth'),
        help='Path to the trained CLIP-HBA-MEG state dict.',
    )
    parser.add_argument(
        '--training_data',
        default=os.environ.get('TRAINING_DATA', 'combined_lamem_memcat'),
        choices=['lamem', 'combined_lamem_memcat'],
        help='Dataset to extract embeddings for.',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=int(os.environ.get('FOLD', 1)),
        help='Fold number (1–5 for lamem; 1–10 for combined_lamem_memcat).',
    )
    parser.add_argument(
        '--img_root',
        default=os.environ.get('LAMEM_IMG_ROOT', './Data/lamem/images/'),
        help='LaMem images root directory.',
    )
    parser.add_argument(
        '--memcat_img_root',
        default=os.environ.get('MEMCAT_IMG_ROOT', './Data/memcat/images/'),
        help='MemCat images root directory (combined_lamem_memcat only).',
    )
    parser.add_argument(
        '--memcat_meta_csv',
        default=os.environ.get('MEMCAT_META_CSV', './Data/memcat/memcat_image_data.csv'),
        help='Path to memcat_image_data.csv (combined_lamem_memcat only).',
    )
    parser.add_argument(
        '--out_dir',
        default=None,
        help='Output directory for .pt embedding files.  Defaults to '
             './Data/lamem/meg_embeddings/ or '
             './Data/combined_lamem_memcat/meg_embeddings/ depending on training_data.',
    )
    parser.add_argument(
        '--vision_layers',
        type=int,
        default=int(os.environ.get('VISION_LAYERS', 24)),
        help='Number of ViT layers with DoRA in the MEG checkpoint.',
    )
    parser.add_argument(
        '--transformer_layers',
        type=int,
        default=int(os.environ.get('TRANSFORMER_LAYERS', 1)),
        help='Number of text-transformer layers with DoRA in the MEG checkpoint.',
    )
    parser.add_argument(
        '--rank',
        type=int,
        default=int(os.environ.get('DORA_RANK', 32)),
        help='DoRA rank used in the MEG checkpoint.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=int(os.environ.get('BATCH_SIZE', 128)),
        help='Batch size for MEG backbone inference.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='DataLoader worker processes for image loading.',
    )
    parser.add_argument(
        '--cuda',
        type=int,
        default=int(os.environ.get('CUDA_DEVICE', 0)),
        help='GPU index (0, 1, …).  Use 2 for CPU.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed for reproducibility.',
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    device = (
        torch.device('cpu') if args.cuda == 2
        else torch.device(f'cuda:{args.cuda}')
    )

    fold = args.fold

    if args.training_data == 'lamem':
        train_csv = f'./Data/lamem/lamem_train_{fold}.csv'
        val_csv   = f'./Data/lamem/lamem_val_{fold}.csv'
        test_csv  = f'./Data/lamem/lamem_test_{fold}.csv'
        img_root  = args.img_root
        memcat_meta_csv = None
        out_dir = args.out_dir or './Data/lamem/meg_embeddings/'
    else:  # combined_lamem_memcat
        train_csv = f'./Data/combined_lamem_memcat/lamem_memcat_train_split_{fold:02d}.csv'
        val_csv   = f'./Data/combined_lamem_memcat/lamem_memcat_val_split_{fold:02d}.csv'
        test_csv  = f'./Data/combined_lamem_memcat/lamem_memcat_test_split_{fold:02d}.csv'
        img_root  = {
            'lamem':  args.img_root,
            'memcat': args.memcat_img_root,
        }
        memcat_meta_csv = args.memcat_meta_csv
        out_dir = args.out_dir or './Data/combined_lamem_memcat/meg_embeddings/'

    print(f'\n=== CLIP-HBA-MEG Embedding Extraction ===')
    print(f'  Training data:  {args.training_data}')
    print(f'  Fold:           {fold}')
    print(f'  Timepoints:     {EXTRACT_START} to {EXTRACT_END} ms, step {EXTRACT_STEP} ms')
    print(f'  Window:         ±{TRAIN_WINDOW_SIZE} ms ({2*TRAIN_WINDOW_SIZE} ms total)')
    print(f'  Output dir:     {out_dir}')
    print(f'  Device:         {device}')
    print()

    extract_meg_embeddings_for_fold(
        meg_checkpoint=args.meg_checkpoint,
        fold=fold,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        img_root=img_root,
        out_dir=out_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        vision_layers=args.vision_layers,
        transformer_layers=args.transformer_layers,
        rank=args.rank,
        memcat_meta_csv=memcat_meta_csv,
    )


if __name__ == '__main__':
    main()
