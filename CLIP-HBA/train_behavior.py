from functions.train_behavior_things_pipeline import run_behavioral_traning
import torch.nn as nn

def main():
    # Define configuration
    config = {
        'csv_file': './Data/spose_embedding66d_rescaled_1806train.csv',
        'img_dir': './Data/Things1854',
        'backbone': 'ViT-L/14',
        'epochs': 500,
        'batch_size': 64,
        'train_portion': 0.8,
        'lr': 3e-4,
        'early_stopping_patience': 20,
        'checkpoint_path': './models/cliphba_dora66_test.pth',
        'random_seed': 1,
        'vision_layers': 2,
        'transformer_layers': 1,
        'rank': 32,
        'criterion': nn.MSELoss(),
        'cuda': 0  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
    }
    
    # Run training
    run_behavioral_traning(config)

if __name__ == '__main__':
    main()