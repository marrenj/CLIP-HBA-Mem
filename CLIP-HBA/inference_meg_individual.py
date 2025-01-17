from train_individual_cichy import  *
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torch.nn import DataParallel
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ImageDataset(Dataset):
    def __init__(self, img_path):
        """
        img_path can be either a directory containing images or a path to a single image.
        """
        if os.path.isdir(img_path):
            self.img_dir = img_path
            self.image_names = [img for img in sorted(os.listdir(img_path))
                                if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.img_dir, single_image_name = os.path.split(img_path)
            self.image_names = [single_image_name]  # Single image in a list
        else:
            raise ValueError(f"Provided path '{img_path}' is neither a directory nor a file, or file type is not supported.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                 std=[0.27608301, 0.26593025, 0.28238822])
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert('RGB')  # Ensure image is RGB
        image = self.transform(image)
        
        return image_name, image


def pearson_rdm(emb):
    # emb shape: (n_images, n_dim)
    corr_matrix = 1 - np.corrcoef(emb)
    np.fill_diagonal(corr_matrix, 0)
    return corr_matrix


def run_image(model, dataset, data_loader, save_folder, inference_start, inference_step, inference_end ,device=torch.device("cuda"), n_dim=66):
    model.eval()
    model.to(device)

    n_images =  len(dataset)
    
    
    with torch.no_grad():
        n_inference_timepoints = len(range(inference_start, inference_end+1, inference_step))
        image_names = []
        output_embs = np.zeros((n_inference_timepoints, n_images, n_dim))
        visual_features = np.zeros((n_inference_timepoints, n_images, 768))
        
        # To keep track of the current position in the dataset
        current_idx = 0

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Processing images")
        for batch_idx, (batch_image_names, batch_images) in progress_bar:
            batch_images = batch_images.to(device)
            image_names.extend(batch_image_names)
            
            # Perform the forward pass
            pred_embs, _, pred_features = model(batch_images)

            # Move to CPU and convert to numpy
            pred_embs = pred_embs.cpu().numpy()  # shape: (n_inference_timepoints, batch_size, n_dim)
            pred_features = pred_features.cpu().numpy()  # shape: (n_inference_timepoints, batch_size, 768)



            # Get the size of the current batch
            batch_size = batch_images.size(0)

            # Store embeddings and RDMs in the output arrays
            output_embs[:, current_idx:current_idx + batch_size, :] = pred_embs  # Store embeddings
            visual_features[:, current_idx:current_idx + batch_size, :] = pred_features  # Store visual features

            # Update current index for the next batch
            current_idx += batch_size

        output_embs = (output_embs - np.min(output_embs)) / (np.max(output_embs) - np.min(output_embs)) * 100

        # save each timepoint embeddings into a folder
        output_rdms = np.zeros((n_inference_timepoints, n_images, n_images))
        for i in range(n_inference_timepoints):
            #rdm generation
            rdm = pearson_rdm(output_embs[i])
            output_rdms[i] = rdm

        return output_rdms, output_embs, visual_features


if __name__ == '__main__':

    overall_rdms = []
    overall_embs = []
    overall_visual_feature = []
    save_folder = f'../output/cliphba_individual_cichy_official_v2/cichy/'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    # Folder containing the models
    # model_save_folder = './models/cichy_individual_official_v3'
    model_save_folder = './models/test_models/cichy_individual_transfer_v2/'

    # Filter files that match the expected naming pattern and ignore 'trial_checkpoint.pth'
    files = [
        name for name in os.listdir(model_save_folder)
        if os.path.isfile(os.path.join(model_save_folder, name)) 
        and name.startswith('cliphba_dynamic_individual_cichy_p')
        and name.endswith('.pth')
        and name != 'trial_checkpoint.pth'
    ]

    # Extract participant IDs from file names
    participant_ids = [
        int(name.split('_p')[1].split('.pth')[0]) for name in files
    ]

    print(f"Found {len(participant_ids)} Participant Models.")

    if participant_ids == []:
        raise ValueError("No participant models found in the model save folder.")

    # Process participants in order of their IDs
    for p_id in sorted(participant_ids):
        seed_everything(1)
        print(f"Processing participant {p_id}")
        ########################################
        img_dir = './Data/Cichy/stimuli'
        # weighting_matrix_path = "./Encoder_Correspondence/weighting_matrix/weighting_matrix_cichy.npy"
        n_dim = 66 # options: 49, 66
        backbone = 'ViT-L/14'
        model_path = f'{model_save_folder}/cliphba_dynamic_individual_cichy_p{p_id}.pth'
        batch_size = 118 #1854/2
        vision_layers = 24
        transformer_layers = 1
        rank = 32
        smoothen_window = 5
        ms_start, ms_end, ms_step = -100, 1000, smoothen_window
        inference_start, inference_end, inference_step, train_window_size = -100, 1000, smoothen_window, 15
        cuda = 'cuda:0'
        load_pretrained = True
        sample_timepoints = list(range(inference_start, inference_end+1, inference_step))
        ########################################


        classnames = classnames66
        classnames = [x[0] for x in classnames]
        
        if backbone == 'RN50':
            pos_embedding = False
        if backbone == 'ViT-B/16' or backbone == 'ViT-B/32' or backbone == 'ViT-L/14': 
            pos_embedding = True

        # weighting_matrix = torch.tensor(np.load(weighting_matrix_path), dtype=torch.float32)
        # weighting_matrix = None
        model = CLIPHBA(classnames=classnames, weighting_matrix=None, backbone_name=backbone, pos_embedding=pos_embedding, ms_start=ms_start, ms_step=ms_step, ms_end=ms_end, train_start=inference_start, train_step=inference_step, train_end=inference_end, train_window_size=train_window_size)
        

        # apply_lora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transformer_layers, r=16, lora_dropout=0.1)
        # apply_lora_to_ViT(model, n_vision_layers=0, n_transformer_layers=transformer_layers, r=16, lora_dropout=0.1)
        apply_dora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transformer_layers, r=rank, dora_dropout=0.1)


        if load_pretrained:
            print(f'Loading Pretrained Model: "{model_path}"')
            model_state_dict = torch.load(model_path)
            adjusted_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
            model.load_state_dict(adjusted_state_dict)
            print('Model Loaded Successfully\n')
        else: 
            print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('Pretrained model not loaded')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')


        device = torch.device(cuda)
        
        # Load the dataset
        dataset = ImageDataset(img_dir)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Run the model and save output embeddings
        p_rdm, p_emb, p_visual_feature = run_image(model, dataset, data_loader, save_folder, inference_start, inference_step, inference_end, device=device, n_dim=n_dim)

        
        overall_rdms.append(p_rdm)
        overall_embs.append(p_emb)
        overall_visual_feature.append(p_visual_feature)

    overall_rdms = np.array(overall_rdms)
    # print(overall_rdms[-1])
    overall_embs = np.array(overall_embs)
    overall_visual_feature = np.array(overall_visual_feature)
    print("Finished processing all participants, overall rdms shape: ", overall_rdms.shape, "overall embs shape: ", overall_embs.shape, "overall visual feature shape: ", overall_visual_feature.shape)
    np.save(f'{save_folder}/overall_rdms.npy', overall_rdms)
    np.save(f'{save_folder}/overall_embs.npy', overall_embs)
    np.save(f'{save_folder}/overall_visual_feature.npy', overall_visual_feature)

