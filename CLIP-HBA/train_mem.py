from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn
import os

def main():
    fold = int(os.environ.get('LAMEM_FOLD', 1))
    config = {
        'model_type': 'clip_hba_mem',  # 'clip_hba_mem' | 'perceptclip'

        # --- Data ---
        'fold':      fold,
        'train_csv': f'./Data/lamem/lamem_train_{fold}.csv',   # columns: image_path, score
        'val_csv':   f'./Data/lamem/lamem_val_{fold}.csv',     # columns: image_path, score
        'test_csv':  f'./Data/lamem/lamem_test_{fold}.csv',    # columns: image_path, score
        'img_root':  os.environ.get('LAMEM_IMG_ROOT', './Data/lamem/images/'),  # prepended to image_path if set
        'preds_dir': './preds/',
        'log_path':  './logs/mem.log',
 
        # --- Backbone (frozen CLIP-HBA) ---
        'backbone_checkpoint': './Data/lamem/epoch97_dora_params.pth',
        'backbone':            'ViT-L/14',
        'vision_layers':       2,   # must match the checkpoint's DoRA config
        'transformer_layers':  1,
        'rank':                32,

        # --- Device ---
        'cuda': int(os.environ.get('CUDA_DEVICE', -1)),   # 0=cuda:0, 1=cuda:1, -1=all GPUs (DataParallel), 2=cpu

        # --- MLP head ---
        'hidden_dims':   (256, 128),
        'dropout_rate':  0.5,

        # --- Training ---
        'epochs':                   300,
        'batch_size':               32,
        'lr':                       1e-5,
        'early_stopping_patience':  20,
        'checkpoint_path':          './models/clip_hba_mem',
        'random_seed':              1,
        'criterion':                nn.MSELoss(),
    }
 
    run_mem_training(config)


if __name__ == '__main__':
    main()