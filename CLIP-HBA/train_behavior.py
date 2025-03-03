from functions.train_behavior_things_pipeline import run_behavioral_traning
import torch.nn as nn

def main():
    # Define configuration
    config = {
        'csv_file': './Data/spose_embedding66d_rescaled_1806train.csv', # csv data annotations of the training stimuli with the corresponding target embeddings
        'img_dir': './Data/Things1854', # path to the image directory
        'backbone': 'ViT-L/14', # CLIP backbone model, ViT-L/14 is the CLIP-HBA model default
        'epochs': 500, 
        'batch_size': 64,
        'train_portion': 0.8,
        'lr': 3e-4, # learning rate
        'early_stopping_patience': 20, # early stopping patience
        'checkpoint_path': './models/cliphba_behavior_test.pth', # path to save the trained model weights
        'random_seed': 1, # random seed, default CLIP-HBA is 1
        'vision_layers': 2, # Last n ViT layers of the model to be trained, default CLIP-HBA-Behavior is 2
        'transformer_layers': 1, # Last n transformer layers to be trained, default CLIP-HBA-Behavior is 1
        'rank': 32, # Rank of the feature reweighting matrix, default CLIP-HBA-Behavior is 32
        'criterion': nn.MSELoss(), # MSE Loss
        'cuda': 0  # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
    }
    
    # Run training
    run_behavioral_traning(config)

if __name__ == '__main__':
    main()