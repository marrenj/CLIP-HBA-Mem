from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn

def main():
    config = {
        'model_type': 'clip_frozen_mlp',

        # --- Data ---
        'fold':      1,
        'train_csv': './Data/lamem/lamem_train_1.csv',
        'val_csv':   './Data/lamem/lamem_val_1.csv',
        'test_csv':  './Data/lamem/lamem_test_1.csv',
        'img_root':  './Data/lamem/images/',
        'preds_dir': './preds/',
        'log_path':  './logs/clip_frozen_mlp.log',

        # --- Device ---
        'cuda': 1,

        # --- MLP head ---
        'hidden_dims':  (512, 256),
        'dropout_rate': 0.5,

        # --- Training ---
        'epochs':                  300,
        'batch_size':              32,
        'lr':                      5e-5,
        'early_stopping_patience': 10,
        'checkpoint_path':         './models/clip_frozen_mlp',
        'random_seed':             1,
        'criterion':               nn.MSELoss(),
    }

    run_mem_training(config)


if __name__ == '__main__':
    main()
