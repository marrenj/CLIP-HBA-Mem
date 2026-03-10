from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn


def main():
    config = {
        # --- Data ---
        'train_csv': './Data/lamem_memcat_train.csv',   # columns: image_path, score
        'val_csv':   './Data/lamem_memcat_val.csv',     # columns: image_path, score
        'img_root':  './Data/lamem_images/',            # prepended to image_path if set

        # --- Backbone (frozen CLIP-HBA) ---
        'backbone_checkpoint': './models/cliphba_behavior_test.pth',
        'backbone':            'ViT-L/14',
        'vision_layers':       2,   # must match the checkpoint's DoRA config
        'transformer_layers':  1,
        'rank':                32,

        # --- Training ---
        'epochs':                   50,
        'batch_size':               64,
        'lr':                       1e-4,
        'early_stopping_patience':  10,
        'checkpoint_path':          './models/clip_hba_mem.pth',
        'preds_dir':                './preds/',
        'random_seed':              1,
        'criterion':                nn.MSELoss(),
    }

    run_mem_training(config)


if __name__ == '__main__':
    main()
