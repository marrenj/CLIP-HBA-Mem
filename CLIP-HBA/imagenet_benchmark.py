import os
import sys
import torch
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import glob
import math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Add the project root to the path so we can import the src modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.CLIPs.clip import clip

def get_accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# DoRA implementation based on the inference code
class DoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, dora_alpha=16, dora_dropout=0.1):
        super(DoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r  # Low-rank factor
        self.dora_alpha = dora_alpha  # Scaling parameter
        self.dora_dropout = nn.Dropout(p=dora_dropout)
        
        # Get the dtype of the original layer
        self.weight_dtype = original_layer.weight.dtype
        # print(f"DoRA Layer init - original weight dtype: {self.weight_dtype}")

        # Decompose original weights into magnitude and direction
        with torch.no_grad():
            W = original_layer.weight.data.clone().to(self.weight_dtype)  # [out_features, in_features]
            W = W.T  # Transpose to [in_features, out_features]
            S = torch.norm(W, dim=0)  # Magnitudes (norms of columns), shape [out_features]
            D = W / S  # Direction matrix with unit-norm columns, shape [in_features, out_features]

        # Store S as a trainable parameter with the original dtype
        self.m = nn.Parameter(S.to(self.weight_dtype))  # [out_features]
        # Store D as a buffer with the original dtype
        self.register_buffer('D', D.to(self.weight_dtype))  # [in_features, out_features]

        # LoRA adaptation of D, use the original dtype
        self.delta_D_A = nn.Parameter(torch.zeros(self.r, original_layer.out_features, dtype=self.weight_dtype))
        self.delta_D_B = nn.Parameter(torch.zeros(original_layer.in_features, self.r, dtype=self.weight_dtype))

        # Scaling
        self.scaling = self.dora_alpha / self.r

        # Initialize delta_D_A and delta_D_B
        self.reset_parameters()

        # Copy the bias from the original layer
        if self.original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone().to(self.weight_dtype))
        else:
            self.bias = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.delta_D_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.delta_D_B, a=math.sqrt(5))
        
        # Ensure parameters stay in the correct dtype
        self.delta_D_A.data = self.delta_D_A.data.to(self.weight_dtype)
        self.delta_D_B.data = self.delta_D_B.data.to(self.weight_dtype)

    @property
    def weight(self):
        # Compute adapted D
        delta_D = (self.delta_D_B @ self.delta_D_A) * self.scaling  # [in_features, out_features]

        D_new = self.D + delta_D  # [in_features, out_features]

        # Normalize columns of D_new
        D_norms = torch.norm(D_new, dim=0, keepdim=True) + 1e-8  # [1, out_features], add epsilon to avoid division by zero
        D_normalized = D_new / D_norms  # [in_features, out_features]

        # Reconstruct the adapted weight
        W = D_normalized * self.m  # [in_features, out_features], m is [out_features]

        W = W.T  # Transpose back to [out_features, in_features]

        return W

    def forward(self, x):
        # Ensure x is the same dtype as our weights
        if x.dtype != self.weight_dtype:
            x = x.to(self.weight_dtype)
            
        # Compute adapted D
        delta_D = (self.delta_D_B @ self.delta_D_A) * self.scaling  # [in_features, out_features]
        delta_D = self.dora_dropout(delta_D)

        D_new = self.D + delta_D  # [in_features, out_features]

        # Normalize columns of D_new
        D_norms = torch.norm(D_new, dim=0, keepdim=True) + 1e-8  # [1, out_features]
        D_normalized = D_new / D_norms  # [in_features, out_features]

        # Reconstruct the adapted weight
        W = D_normalized * self.m  # [in_features, out_features]
        W = W.T  # [out_features, in_features]

        # Compute output
        return F.linear(x, W, self.bias)


    

def apply_dora_to_ViT(model, n_vision_layers=1, n_transformer_layers=1, r=8, dora_dropout=0.1, seed=123):

    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Specific blocks to modify in the visual transformer
    block_indices = range(-n_vision_layers, 0)  # Adjusted for proper indexing

    for idx in block_indices:
        # Access the specific ResidualAttentionBlock
        target_block = model_module.visual.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a DoRALayer
        dora_layer = DoRALayer(target_layer, r=r, dora_dropout=dora_dropout)
        target_block.attn.out_proj = dora_layer

    # Specific blocks to modify in the main transformer
    block_indices = range(-n_transformer_layers, 0)

    for idx in block_indices:
        # Access the specific ResidualAttentionBlock
        target_block = model_module.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a DoRALayer
        dora_layer = DoRALayer(target_layer, r=r, dora_dropout=dora_dropout)
        target_block.attn.out_proj = dora_layer

def load_model(config):
    """Load the CLIP model with appropriate weights"""
    device = torch.device(config['cuda'] if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {config['backbone']} model...")
    model, preprocess = clip.load(config['backbone'])

    if config['load_hba']:
        print(f"Loading HBA weights from {config['model_path']}...")
        
        # 1. First apply DoRA to the model - using approach from inference code
        print("Applying DoRA to model...")

        apply_dora_to_ViT(model, 
                         n_vision_layers=2,  # Apply to the last 2 vision layers
                         n_transformer_layers=1,  # Apply to the last text layer
                         r=32,  # DoRA rank
                         dora_dropout=0.1)  # DoRA dropout
        
        # 2. Load the checkpoint - use the approach from inference code
        model_state_dict = torch.load(config['model_path'])
        adjusted_state_dict = {key.replace("clip_model.", ""): value 
                             for key, value in model_state_dict.items()}
                
        # Filter the model state dict to match the current model structure
        model.load_state_dict(adjusted_state_dict)
        
    else:
        print("Using original CLIP weights")
    
    # Explicitly move the entire model to the specified device
    model = model.to(device)
    
    model.eval()
    return model, preprocess, device

def find_imagenet_val_root(config):
    """Find the actual root directory for ImageNet validation images by searching the directory structure"""
    # Start from the Data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Data'))
    
    # Search patterns to try
    search_patterns = [
        # Common patterns for ImageNet validation data
        os.path.join(data_dir, 'imagenet', 'imagenet-val'),
        os.path.join(data_dir, 'imagenet', 'val'),
        os.path.join(data_dir, 'imagenet-val'),
        os.path.join(data_dir, 'val')
    ]
    
    # Try to find an existing directory
    for pattern in search_patterns:
        if os.path.isdir(pattern):
            print(f"Found ImageNet validation directory: {pattern}")
            return pattern
    
    # If not found with patterns, look for directories containing class folders
    for root, dirs, files in os.walk(data_dir):
        # Check if this directory contains ImageNet class directories (starting with 'n')
        n_dirs = [d for d in dirs if d.startswith('n') and len(d) == 9]  # ImageNet synsets start with n followed by 8 digits
        if n_dirs:
            print(f"Found ImageNet validation directory by class folders: {root}")
            return root
    
    # If we still can't find it, just return the configured directory
    return config['img_dir']

class ImageNetValidationDataset(Dataset):
    """ImageNet validation dataset"""
    
    def __init__(self, image_paths, labels, preprocess):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            processed_image = self.preprocess(image)
            label = self.labels[idx]
            return processed_image, label
        except Exception as e:
            # Return a default image/label on error
            # In a real-world scenario, you might want to handle this differently
            print(f"Error loading image {img_path}: {e}")
            # Return an empty tensor of the right shape as a placeholder
            return torch.zeros(3, 224, 224), -1  # assuming 224x224 images

def load_image_paths_and_labels(config):
    """Load image paths and corresponding class labels"""
    # Load the CSV file containing image paths and class labels
    csv_path = os.path.join(os.path.dirname(config['img_dir']), 'imagenet_val_images.csv')
    df = pd.read_csv(csv_path)
    
    # Extract relative paths and class names
    csv_image_paths = df['images'].tolist()
    class_names = df['classes'].tolist()
    
    # Find the actual root directory for validation images
    val_root = find_imagenet_val_root(config)
    
    # Create a mapping of file basenames to their full paths for fast lookups
    file_map = {}
    for root, _, files in os.walk(val_root):
        for file in files:
            if file.endswith('.JPEG'):
                file_map[file] = os.path.join(root, file)
    
    print(f"Found {len(file_map)} image files in validation directory")
    
    # Create absolute paths by matching filenames
    absolute_paths = []
    labels_filtered = []
    filtered_class_names = []
    
    for i, rel_path in enumerate(csv_image_paths):
        # Extract just the filename from the relative path
        filename = os.path.basename(rel_path)
        
        if filename in file_map:
            # Found the file
            absolute_paths.append(file_map[filename])
            labels_filtered.append(class_names[i])
            filtered_class_names.append(class_names[i])
    
    print(f"Successfully mapped {len(absolute_paths)} out of {len(csv_image_paths)} images")

    
    # Get unique class names from the filtered list
    unique_classes = sorted(list(set(filtered_class_names)))
    
    # Create class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    # Convert class names to indices
    labels = [class_to_idx[cls] for cls in labels_filtered]
    
    return absolute_paths, labels, unique_classes

def batch_encode_text(model, class_names, device):
    """Encode class names with 'a photo of a {class}' template"""
    templates = [f"a photo of {cls}" for cls in class_names]
    text_tokens = clip.tokenize(templates).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=1)
    
    return text_features

def evaluate_model_with_dataloader(model, dataloader, text_features, device):
    """Evaluate the model on the validation set using a DataLoader"""
    top1_acc = 0.0
    top5_acc = 0.0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            # Filter out any error cases (-1 labels)
            valid_indices = (labels >= 0)
            if not valid_indices.any():
                continue
                
            images = images[valid_indices].to(device)
            labels = labels[valid_indices].to(device)
            
            if images.size(0) == 0:
                continue
            
            # Extract image features - using standard encode_image call with pos_embedding=True
            pos_embedding = True if hasattr(model.visual, 'transformer') else False
            image_features = model.encode_image(images, pos_embedding=pos_embedding)
            image_features = F.normalize(image_features, dim=1)
            
            # Calculate similarities
            similarity = image_features @ text_features.T
            
            # Calculate accuracies
            acc1, acc5 = get_accuracy(similarity, labels, topk=(1, 5))
            
            # Update running stats
            batch_count = images.size(0)
            top1_acc += acc1.item() * batch_count
            top5_acc += acc5.item() * batch_count
            total += batch_count
    
    # Normalize by total count
    if total > 0:
        top1_acc /= total
        top5_acc /= total
    
    return top1_acc, top5_acc

def main(config):
    """Main function to run the benchmark"""
    # Load model
    model, preprocess, device = load_model(config)
    
    # Load data
    image_paths, labels, class_names = load_image_paths_and_labels(config)
    print(f"Loaded {len(image_paths)} images with {len(class_names)} unique classes")
    
    # Create dataset and dataloader
    dataset = ImageNetValidationDataset(image_paths, labels, preprocess)
    
    # Use multiple workers for better performance
    # Number of workers set to either 4 or number of CPU cores - 1, whichever is smaller
    num_workers = min(4, os.cpu_count() - 1)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers)
    
    # Encode class names
    text_features = batch_encode_text(model, class_names, device)
    
    # Evaluate model
    top1_acc, top5_acc = evaluate_model_with_dataloader(model, dataloader, text_features, device)

    print("\n\n********** Results **********\n")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print("\n**************************************\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ImageNet Benchmark for CLIP/CLIP-HBA')
    parser.add_argument('--img_dir', type=str, default='./Data/imagenet/imagenet-val', help='input images directory')
    parser.add_argument('--load_hba', action='store_true', help='load HBA weights (default: False - use original CLIP weights)')
    parser.add_argument('--backbone', type=str, default='ViT-L/14', help='CLIP backbone model')
    parser.add_argument('--model_path', type=str, default='./models/cliphba_behavior.pth', help='path to the trained model')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--cuda', type=str, default='cuda:1', help='cuda device (cuda:0, cuda:1, or -1 for all)')
    
    args = parser.parse_args()
    config = vars(args)
    
    main(config)