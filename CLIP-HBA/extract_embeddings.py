"""Pre-extract frozen backbone embeddings for fast MLP head training.

For both clip_hba_mem and clip_frozen_mlp, the backbone is fully frozen and
its output embeddings are identical every epoch.  Running 48 k images through
ViT-L/14 on every training epoch (1 500 forward passes at batch_size=32) is
the primary throughput bottleneck.  This script runs all images through the
backbone *once*, saves the 768-dim CLS tokens alongside memorability scores
and image paths, and then training uses EmbeddingDataset (defined in
functions/train_mem_pipeline.py) to serve in-memory tensors instead of images.

Expected speedup: ~100x per epoch (seconds rather than ~7.5 minutes).

Usage
-----
    # Single fold, clip_hba_mem (reads LAMEM_* env vars if set):
    python extract_embeddings.py --model_type clip_hba_mem --fold 1

    # Single fold, vanilla CLIP:
    python extract_embeddings.py --model_type clip_frozen_mlp --fold 1

    # All 5 folds for both model types (via the accompanying SLURM script):
    sbatch extract_embeddings.slurm

Outputs
-------
    <out_dir>/<model_type>_fold<N>_train.pt
    <out_dir>/<model_type>_fold<N>_val.pt
    <out_dir>/<model_type>_fold<N>_test.pt

    Each .pt file is a dict:
        embeddings  - Tensor[N, 768]   float32 CLS token embeddings
        scores      - Tensor[N]        float32 memorability scores
        image_paths - list[N]          relative image path strings (from CSV)

Enabling fast training
----------------------
    Add ``'embeddings_dir': './Data/lamem/embeddings/'`` to the config dict
    in train_mem.py or train_clip_mlp.py.  When this key is present,
    run_mem_training() uses EmbeddingDataset + MLPOnlyHead instead of loading
    raw images and running the full backbone on every batch.
"""

import argparse
import os
import pathlib

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions.train_mem_pipeline import (
    MemDataset,
    PerceptCLIPDataset,
    CLIPHBAMem,
    CLIPFrozenMLP,
)
from functions.train_behavior_things_pipeline import seed_everything


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _build_backbone_and_extractor(
    model_type: str,
    backbone_checkpoint: str,
    backbone_name: str,
    vision_layers: int,
    transformer_layers: int,
    rank: int,
):
    """Instantiate the frozen backbone and return (model, DatasetClass, embed_fn).

    embed_fn(model, images) -> Tensor[B, 768] extracts the CLS-token
    embedding without running through the MLP head.
    """
    if model_type == 'clip_hba_mem':
        model = CLIPHBAMem(
            backbone_checkpoint=backbone_checkpoint,
            backbone_name=backbone_name,
            vision_layers=vision_layers,
            transformer_layers=transformer_layers,
            rank=rank,
            hidden_dims=(256, 128),  # dummy head — never called during extraction
            dropout_rate=0.0,
        )
        DatasetClass = MemDataset

        def embed_fn(m: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
            raw = m.module if isinstance(m, torch.nn.DataParallel) else m
            return raw.backbone.clip_model.encode_image(
                images, raw.backbone.pos_embedding)   # [B, 768]

    elif model_type == 'clip_frozen_mlp':
        model = CLIPFrozenMLP(
            hidden_dims=(256, 128),  # dummy head — never called during extraction
            dropout_rate=0.0,
        )
        DatasetClass = PerceptCLIPDataset

        def embed_fn(m: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
            raw = m.module if isinstance(m, torch.nn.DataParallel) else m
            vis_out = raw.vision_model(images)
            return raw.visual_projection(vis_out[1])   # [B, 768]

    else:
        raise ValueError(
            f'Unknown model_type: {model_type!r}. '
            f'Choose "clip_hba_mem" or "clip_frozen_mlp".'
        )

    return model, DatasetClass, embed_fn


def extract_embeddings_for_fold(
    model_type: str,
    fold: int,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    img_root: str,
    backbone_checkpoint: str,
    out_dir: str,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 8,
    backbone_name: str = 'ViT-L/14',
    vision_layers: int = 2,
    transformer_layers: int = 1,
    rank: int = 32,
) -> None:
    """Extract and save embeddings for all three splits of one fold.

    Skips any split whose output file already exists so the script is safe
    to re-run after an interruption.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, DatasetClass, embed_fn = _build_backbone_and_extractor(
        model_type=model_type,
        backbone_checkpoint=backbone_checkpoint,
        backbone_name=backbone_name,
        vision_layers=vision_layers,
        transformer_layers=transformer_layers,
        rank=rank,
    )
    model.eval()
    model.to(device)

    splits = {
        'train': train_csv,
        'val':   val_csv,
        'test':  test_csv,
    }

    for split_name, csv_path in splits.items():
        out_path = out_dir / f'{model_type}_fold{fold}_{split_name}.pt'

        if out_path.exists():
            print(f'[Skip] {out_path} already exists — delete it to re-extract.')
            continue

        dataset = DatasetClass(csv_file=csv_path, img_root=img_root)
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        all_embeddings:  list = []
        all_scores:      list = []
        all_image_paths: list = []

        print(f'\n[{model_type} | fold {fold} | {split_name}]  '
              f'{len(dataset)} images  ->  {out_path}')

        with torch.no_grad(), tqdm(loader, desc=f'  {split_name}') as pbar:
            for image_paths, images, scores in pbar:
                images = images.to(device)
                emb    = embed_fn(model, images)   # [B, 768]
                all_embeddings.append(emb.cpu())
                all_scores.append(scores.cpu())
                all_image_paths.extend(image_paths)

        payload = {
            'embeddings':  torch.cat(all_embeddings, dim=0),   # [N, 768]
            'scores':      torch.cat(all_scores,      dim=0),   # [N]
            'image_paths': all_image_paths,                      # list[N]
        }
        torch.save(payload, out_path)
        print(f'  Saved {payload["embeddings"].shape[0]:,} embeddings  '
              f'({payload["embeddings"].shape}) -> {out_path}')

    # Free GPU memory before the caller potentially builds another backbone.
    model.cpu()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pre-extract frozen backbone embeddings for memorability MLP training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model_type',
        default=os.environ.get('MODEL_TYPE', 'clip_frozen_mlp'),
        choices=['clip_hba_mem', 'clip_frozen_mlp'],
        help='Which frozen backbone to extract from.',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=int(os.environ.get('LAMEM_FOLD', 5)),
        help='LaMem fold number (1-5).',
    )
    parser.add_argument(
        '--img_root',
        default=os.environ.get('LAMEM_IMG_ROOT', './Data/lamem/images/'),
        help='Root directory prepended to image_path column values.',
    )
    parser.add_argument(
        '--out_dir',
        default='./Data/lamem/embeddings/',
        help='Directory where .pt embedding files are written.',
    )
    parser.add_argument(
        '--backbone_checkpoint',
        default='./Data/lamem/epoch97_dora_params.pth',
        help='CLIP-HBA checkpoint path (only used for clip_hba_mem).',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for backbone inference during extraction.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='DataLoader worker count for image loading.',
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
        help='Random seed (affects reproducibility of any stochastic ops).',
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device('cpu') if args.cuda == 2 else torch.device(f'cuda:{args.cuda}')

    fold = args.fold
    extract_embeddings_for_fold(
        model_type=args.model_type,
        fold=fold,
        train_csv=f'./Data/lamem/lamem_train_{fold}.csv',
        val_csv=f'./Data/lamem/lamem_val_{fold}.csv',
        test_csv=f'./Data/lamem/lamem_test_{fold}.csv',
        img_root=args.img_root,
        backbone_checkpoint=args.backbone_checkpoint,
        out_dir=args.out_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
