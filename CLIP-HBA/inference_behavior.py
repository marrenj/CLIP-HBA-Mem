from functions.inference_behavior_code import run_inference

if __name__ == '__main__':
    # hyperparameters
    img_dir = './Data/demo_images'
    load_hba = True  # False will load the original CLIP-ViT weights
    backbone = 'ViT-L/14'
    model_path = f'./models/cliphba_behavior.pth'
    save_folder = f'../output/cliphba_behavior_demo/'
    batch_size = 32
    cuda = 'cuda:0'

    run_inference(img_dir, load_hba, backbone, model_path, save_folder, batch_size, cuda)