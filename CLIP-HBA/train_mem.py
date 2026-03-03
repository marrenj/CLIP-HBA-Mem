from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn

def main():
    config = {
        # --- Data ---
        'train_csv': './Data/lamem/lamem_train_1.csv',   # columns: image_path, score
        'val_csv':   './Data/lamem/lamem_val_1.csv',     # columns: image_path, score
        'test_csv':  './Data/lamem/lamem_test_1.csv',    # columns: image_path, score
        'img_root':  './Data/lamem/images/',            # prepended to image_path if set
 
        # --- Backbone (frozen CLIP-HBA) ---
        'backbone_checkpoint': './Data/lamem/epoch97_dora_params.pth',
        'backbone':            'ViT-L/14',
        'vision_layers':       2,   # must match the checkpoint's DoRA config
        'transformer_layers':  1,
        'rank':                32,

        # --- Device ---
        'cuda':                     0,   # 0=cuda:0, 1=cuda:1, -1=all GPUs (DataParallel), 2=cpu

        # --- Training ---
        'epochs':                   300,
        'batch_size':               32,
        'lr':                       5e-5,
        'early_stopping_patience':  20,
        'checkpoint_path':          './models/clip_hba_mem.pth',
        'random_seed':              1,
        'criterion':                nn.MSELoss(),
    }
 
    run_mem_training(config)


if __name__ == '__main__':
    main()