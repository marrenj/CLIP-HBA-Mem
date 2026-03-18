"""Run memorability inference on LaMem, THINGS, or MemCat datasets.

Usage examples:
    # LaMem test set (fold 1) — CLIP-HBA-Mem (default)
    python inference_mem.py --dataset lamem --fold 1

    # All 5 LaMem folds
    python inference_mem.py --dataset lamem --fold all

    # All 58,741 LaMem images (requires explicit --checkpoint)
    python inference_mem.py --dataset lamem --fold all_lamem \
        --checkpoint ./models/clip_hba_mem_fold1.pth

    # Vanilla frozen CLIP + MLP
    python inference_mem.py --dataset lamem --fold 1 \
        --model_type clip_frozen_mlp \
        --checkpoint ./models/clip_frozen_mlp_fold{fold}.pth

    # PerceptCLIP (CLIP + LoRA + MLP)
    python inference_mem.py --dataset lamem --fold 1 \
        --model_type perceptclip \
        --checkpoint ./models/perceptclip_fold{fold}.pth

    # THINGS dataset (requires --things_img_dir)
    python inference_mem.py --dataset things --things_img_dir ./Data/Things1854

    # MemCat dataset
    python inference_mem.py --dataset memcat

    # Custom checkpoint and MLP architecture
    python inference_mem.py --dataset lamem --fold 1 \
        --checkpoint ./models/clip_hba_mem_fold1.pth \
        --hidden_dims 512 256 --dropout_rate 0.3
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr

from functions.train_mem_pipeline import (
    CLIPHBAMem, CLIPFrozenMLP, clip_lora_model,
    MemDataset, PerceptCLIPDataset,
)
from functions.train_behavior_things_pipeline import seed_everything


def build_standardised_csv(dataset, output_path, **kwargs):
    """Create a temporary CSV with columns (image_path, score) for any dataset.

    Returns the path to the CSV and the img_root to use with MemDataset.
    """
    if dataset == 'lamem':
        fold = kwargs['fold']
        if fold == 'all_lamem':
            csv_path = './Data/lamem/all_lamem.csv'
        else:
            csv_path = f'./Data/lamem/lamem_test_{fold}.csv'
        return csv_path, './Data/lamem/images/'

    elif dataset == 'things':
        things_csv = './Data/THINGS_Memorability_Scores.csv'
        things_img_dir = kwargs['things_img_dir']
        df = pd.read_csv(things_csv)
        df_out = pd.DataFrame({
            'image_path': df['file_path'].str.removeprefix('images/'),
            'score': df['cr'],
        })
        df_out = df_out.dropna(subset=['score'])
        df_out.to_csv(output_path, index=False)
        return output_path, things_img_dir

    elif dataset == 'memcat':
        memcat_csv = './Data/memcat/memcat_image_data.csv'
        memcat_img_root = './Data/memcat/memcat_images'
        df = pd.read_csv(memcat_csv)
        # Images are at memcat_images/<category>/<subcategory>/<image_file>
        df_out = pd.DataFrame({
            'image_path': df.apply(
                lambda r: os.path.join(r['category'], r['subcategory'], r['image_file']),
                axis=1),
            'score': df['memorability_w_fa_correction'],
        })
        df_out.to_csv(output_path, index=False)
        return output_path, memcat_img_root

    else:
        raise ValueError(f'Unknown dataset: {dataset!r}')


def _build_dataset(model_type: str, csv_path: str, img_root: str):
    """Return the correct Dataset class for the given model type.

    clip_frozen_mlp and perceptclip were trained with standard CLIP
    normalisation (PerceptCLIPDataset).  clip_hba_mem uses the CLIP-HBA
    normalisation (MemDataset).
    """
    if model_type in ('perceptclip', 'clip_frozen_mlp'):
        return PerceptCLIPDataset(csv_file=csv_path, img_root=img_root)
    return MemDataset(csv_file=csv_path, img_root=img_root)


def _build_model(config: dict) -> nn.Module:
    """Instantiate the correct model class and return it (weights not yet loaded)."""
    model_type = config['model_type']
    hidden_dims = tuple(config['hidden_dims'])
    dropout_rate = config['dropout_rate']

    if model_type == 'clip_hba_mem':
        return CLIPHBAMem(
            backbone_checkpoint=config['backbone_checkpoint'],
            backbone_name=config['backbone'],
            vision_layers=config['vision_layers'],
            transformer_layers=config['transformer_layers'],
            rank=config['rank'],
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )
    elif model_type == 'clip_frozen_mlp':
        return CLIPFrozenMLP(
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
        )
    elif model_type == 'perceptclip':
        return clip_lora_model()
    else:
        raise ValueError(f'Unknown model_type: {model_type!r}')


def run_inference(config):
    seed_everything(config['random_seed'])

    dataset_label = config['dataset']
    model_type = config['model_type']
    folds = config['folds']
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(config['output_dir'], f'{dataset_label}_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    for fold in folds:
        if dataset_label == 'lamem':
            fold_suffix = '_all_lamem' if fold == 'all_lamem' else f'_fold{fold}'
        else:
            fold_suffix = ''
        tmp_csv = os.path.join(out_dir, f'_tmp_{dataset_label}{fold_suffix}.csv')

        csv_path, img_root = build_standardised_csv(
            dataset_label, tmp_csv, fold=fold,
            things_img_dir=config.get('things_img_dir', ''))

        ds = _build_dataset(model_type, csv_path, img_root)
        print(f'\n[{dataset_label.upper()}{fold_suffix}] {len(ds)} images')

        loader = DataLoader(ds, batch_size=config['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)

        # Resolve fold placeholder in checkpoint path
        checkpoint = config['checkpoint']
        if dataset_label == 'lamem' and fold != 'all_lamem' and '{fold}' in checkpoint:
            checkpoint = checkpoint.replace('{fold}', str(fold))
        elif fold == 'all_lamem' and '{fold}' in checkpoint:
            raise ValueError(
                '--fold all_lamem requires an explicit --checkpoint path '
                '(the {fold} placeholder cannot be resolved for all_lamem).'
            )

        model = _build_model(config)

        state_dict = torch.load(checkpoint, map_location='cpu')
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Remap legacy fc1/fc2/fc3 keys (old MLP class) to mlp_head Sequential indices.
        # Only applies to clip_hba_mem and clip_frozen_mlp checkpoints.
        if model_type in ('clip_hba_mem', 'clip_frozen_mlp'):
            _fc_to_idx = {'fc1': 'mlp_head.0', 'fc2': 'mlp_head.3', 'fc3': 'mlp_head.6'}
            state_dict = {
                (_fc_to_idx[parts[0]] + '.' + '.'.join(parts[1:])
                 if (parts := k.split('.'))[0] in _fc_to_idx else k): v
                for k, v in state_dict.items()
            }

        if model_type in ('clip_frozen_mlp', 'perceptclip'):
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        print(f'[Model] Loaded {model_type} checkpoint: {checkpoint}')

        device = torch.device(config['device'])
        model.to(device)
        model.eval()

        all_paths, all_preds, all_targets = [], [], []
        with torch.no_grad():
            for image_paths, images, targets in tqdm(loader, desc='Inference'):
                images = images.to(device)
                preds = model(images).squeeze(1).cpu().numpy()
                all_paths.extend(image_paths)
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        rho, p_val = spearmanr(all_preds, all_targets)
        mse = float(np.mean((all_preds - all_targets) ** 2))

        print(f'[Results] Spearman rho: {rho:.4f} (p={p_val:.2e})  |  MSE: {mse:.6f}')
        print(f'[Results] Pred range: [{all_preds.min():.4f}, {all_preds.max():.4f}]  '
              f'std: {all_preds.std():.4f}')

        results_df = pd.DataFrame({
            'image_path': all_paths,
            'pred_score': all_preds,
            'true_score': all_targets,
        })
        save_path = os.path.join(out_dir, f'{dataset_label}{fold_suffix}_predictions.csv')
        results_df.to_csv(save_path, index=False)
        print(f'[Saved] {save_path}')

        summary_rows.append({
            'dataset': dataset_label,
            'model_type': model_type,
            'fold': fold if fold is not None else '',
            'n_images': len(ds),
            'spearman_rho': rho,
            'spearman_p': p_val,
            'mse': mse,
            'pred_std': float(all_preds.std()),
            'checkpoint': checkpoint,
        })

        # Clean up temp CSV
        if os.path.exists(tmp_csv) and tmp_csv != csv_path:
            os.remove(tmp_csv)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, f'{dataset_label}_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'\n[Summary] Saved to {summary_path}')
    print(summary_df.to_string(index=False))
    print('\nDone.')


def main():
    parser = argparse.ArgumentParser(
        description='Run memorability inference with a trained memorability model.')

    parser.add_argument('--dataset', required=True,
                        choices=['lamem', 'things', 'memcat'],
                        help='Dataset to run inference on.')
    parser.add_argument('--fold', default='1',
                        help='LaMem fold: 1-5, "all" (all 5 test folds), or "all_lamem" '
                             '(all 58,741 images from all_lamem.csv). '
                             '"all_lamem" requires an explicit --checkpoint path. '
                             'Ignored for non-LaMem datasets.')
    parser.add_argument('--things_img_dir', default='./Data/Things1854',
                        help='Directory containing THINGS images (for --dataset things).')

    parser.add_argument('--model_type', default='clip_hba_mem',
                        choices=['clip_hba_mem', 'clip_frozen_mlp', 'perceptclip'],
                        help='Model architecture to use for inference.')
    parser.add_argument('--checkpoint', default='./models/clip_hba_mem_fold{fold}.pth',
                        help='Path to trained model checkpoint. Use {fold} as placeholder '
                             'for LaMem fold number.')

    # clip_hba_mem-only backbone args (ignored for clip_frozen_mlp / perceptclip)
    parser.add_argument('--backbone_checkpoint',
                        default='./Data/lamem/epoch97_dora_params.pth',
                        help='Path to frozen CLIP-HBA backbone weights '
                             '(clip_hba_mem only; ignored for other model types).')
    parser.add_argument('--backbone', default='ViT-L/14',
                        help='CLIP backbone name (clip_hba_mem only).')
    parser.add_argument('--vision_layers', type=int, default=2,
                        help='DoRA vision layers (clip_hba_mem only).')
    parser.add_argument('--transformer_layers', type=int, default=1,
                        help='DoRA transformer layers (clip_hba_mem only).')
    parser.add_argument('--rank', type=int, default=32,
                        help='DoRA rank (clip_hba_mem only).')

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 256],
                        help='MLP hidden layer dimensions (e.g. 512 256).')
    parser.add_argument('--dropout_rate', type=float, default=0.5)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda:0',
                        help='Device string (e.g. cuda:0, cuda:1, cpu).')
    parser.add_argument('--output_dir', default='./preds/inference/',
                        help='Directory to save prediction CSVs.')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # Resolve folds
    if args.dataset == 'lamem':
        if args.fold == 'all':
            folds = [1, 2, 3, 4, 5]
        elif args.fold == 'all_lamem':
            folds = ['all_lamem']
        else:
            folds = [int(args.fold)]
    else:
        folds = [None]

    config = {
        'dataset': args.dataset,
        'folds': folds,
        'things_img_dir': args.things_img_dir,
        'model_type': args.model_type,
        'checkpoint': args.checkpoint,
        'backbone_checkpoint': args.backbone_checkpoint,
        'backbone': args.backbone,
        'vision_layers': args.vision_layers,
        'transformer_layers': args.transformer_layers,
        'rank': args.rank,
        'hidden_dims': args.hidden_dims,
        'dropout_rate': args.dropout_rate,
        'batch_size': args.batch_size,
        'device': args.device,
        'output_dir': args.output_dir,
        'random_seed': args.seed,
    }

    run_inference(config)


if __name__ == '__main__':
    main()
