from functions.inference_meg_group_pipeline import run_meg_group_inference

def main():
    # Define configuration
    config = {
        # Data paths
        'img_dir': './Data/test_images/', # Path to the directory containing images
        'n_dim': 66,  # options: 66
        
        # Model parameters
        'backbone': 'ViT-L/14', # backbone model for CLIP. All published CLIP-HBA models use ViT-L/14
        'model_path': './models/cliphba_meg_group.pth', # Path to the trained model
        'save_folder': '../output/cliphba_meg_things_demo/things/', # Path to save the output
        'batch_size': 128,
        'vision_layers': 24, # Number of vision layers in the CLIP model that was trained, default pretrained CLIP-HBA-MEG model has 24 layers
        'transformer_layers': 1, # Number of transformer layers in the CLIP model that was trained, default pretrained CLIP-HBA-MEG model has 1 layer
        'rank': 32, # Rank of DoRA modules, default pretrained CLIP-HBA-MEG model has rank 32
        
        # Time parameters
        'ms_start': -100, # Start time in ms, follow the parameters used in the training. Default pretrained CLIP-HBA-MEG model uses -100 ms
        'ms_end': 1300, # End time in ms, follow the parameters used in the training. Default pretrained CLIP-HBA-MEG model uses 1300 ms
        'ms_step': 5, # Time step in ms, follow the parameters used in the training. Default pretrained CLIP-HBA-MEG model uses 5 ms
        'inference_start': -100, # Start time for inference in ms, this can be changed based on the desired inference window
        'inference_end': 1300, # End time for inference in ms, this can be changed based on the desired inference window
        'inference_step': 5, # Time step for inference in ms, this can be changed based on the desired inference window
        'train_window_size': 15, # Window size for training in ms, follow the parameters used in the training. Default pretrained CLIP-HBA-MEG model uses 15 ms
        
        # Device settings
        'cuda': 'cuda:1', # Device to run the model on, 'cuda:0' for GPU 0, 'cuda:1' for GPU 1, 'cuda' for all GPUs
        'load_pretrained': True, # Must load pretrained model
        'random_seed': 1 # default pre-trained CLIP-HBA models all use random seed 1
    }

    # Run inference
    run_meg_group_inference(config)

if __name__ == '__main__':
    main()