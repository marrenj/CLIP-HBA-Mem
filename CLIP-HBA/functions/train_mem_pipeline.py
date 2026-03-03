import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from scipy.stats import spearmanr

from functions.train_behavior_things_pipeline import (
    CLIPHBA,
    apply_dora_to_ViT,
    seed_everything,
    count_trainable_parameters,
)
from functions.spose_dimensions import classnames66

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class MemDataset(Dataset):
    """Dataset for image memorability prediction.
 
    Expects a CSV with columns:
        image_path  - absolute path or relative to img_root
        score       - memorability score in [0, 1]
    """
 
    def __init__(self, csv_file, img_root=''):
        self.img_root = img_root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                 std=[0.27608301, 0.26593025, 0.28238822]),
        ])
        self.annotations = pd.read_csv(csv_file)
 
    def __len__(self):
        return len(self.annotations)
 
    def __getitem__(self, index):
        row = self.annotations.iloc[index]
        img_path = (os.path.join(self.img_root, row['image_path'])
                    if self.img_root else row['image_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        score = torch.tensor(float(row['score']), dtype=torch.float32)
        return image, score


class CLIPHBAMem(nn.Module):
    """Frozen CLIP-HBA backbone + PerceptCLIP-style MLP head for memorability prediction.
 
    The backbone's CLIP image encoder produces a 768-dim projected embedding (encode_image output for ViT-L/14). 
 
    Architecture:
        FC1:  Linear(768 → 512) + ReLU + Dropout(0.5)
        FC2:  Linear(512 → 256) + ReLU + Dropout(0.5)
        FC3:  Linear(256 → 1)   + Sigmoid
    """

    def __init__(self, backbone_checkpoint, backbone_name='ViT-L/14',
                 vision_layers=2, transformer_layers=1, rank=32):
        super().__init__()
 
        # --- Frozen CLIP-HBA backbone ---
        pos_embedding = (backbone_name != 'RN50')
        self.backbone = CLIPHBA(classnames=classnames66,
                                backbone_name=backbone_name,
                                pos_embedding=pos_embedding)
        apply_dora_to_ViT(self.backbone,
                          n_vision_layers=vision_layers,
                          n_transformer_layers=transformer_layers,
                          r=rank)
 
        state_dict = torch.load(backbone_checkpoint, map_location='cpu')

        # Strip DataParallel 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.backbone.load_state_dict(state_dict, strict=False)

        for p in self.backbone.parameters():
            p.requires_grad = False

        # --- MLP head (PerceptCLIP-style + sigmoid) ---
        self.fc1     = nn.Linear(768,  512)
        self.relu1   = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, 256)
        self.relu2   = nn.ReLU()
        self.fc3     = nn.Linear(256, 1)

    def train(self, mode=True):
        super().train(mode)
        # The backbone must stay in eval mode at all times (it is frozen).
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            emb = self.backbone.clip_model.encode_image(x, self.backbone.pos_embedding)  # [B, 768]
 
        h = self.fc1(emb);  h = self.relu1(h);  h = self.dropout(h)
        h = self.fc2(h);    h = self.relu2(h);  h = self.dropout(h)
        return torch.sigmoid(self.fc3(h))  # [B, 1]

    def mlp_parameters(self):
        """Returns only the MLP head parameters (used by the optimiser)."""
        head_names = {'fc1', 'relu1', 'dropout', 'fc2', 'relu2', 'fc3'}
        return [p for n, p in self.named_parameters()
                if n.split('.')[0] in head_names]


def evaluate_mem_model(model, data_loader, device, criterion):
    """Returns avg MSE loss and Spearman rank correlation on data_loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad(), tqdm(data_loader, desc='Evaluating', leave=False) as pbar:
        for images, targets in pbar:
            images  = images.to(device)
            targets = targets.to(device)
 
            preds = model(images)
            loss  = criterion(preds, targets)
            total_loss += loss.item() * images.size(0)
 
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})
 
    avg_loss = total_loss / len(data_loader.dataset)
    rho, _ = spearmanr(all_preds, all_targets)
    return avg_loss, rho


def train_mem_model(model, train_loader, val_loader, device, optimizer, criterion,
                    epochs, early_stopping_patience=10,
                    checkpoint_path='clip_hba_mem.pth',
                    test_loader=None):
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0
 
    # Initial evaluation
    print('*' * 40)
    print('Initial evaluation')
    best_val_loss, rho = evaluate_mem_model(model, val_loader, device, criterion)
    print(f'Val MSE: {best_val_loss:.4f}  |  Spearman ρ: {rho:.4f}')
    print('*' * 40 + '\n')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for images, targets in pbar:
                images  = images.to(device)
                targets = targets.to(device)
 
                optimizer.zero_grad()
                preds = model(images)
                loss  = criterion(preds, targets)
                loss.backward()
                optimizer.step()
 
                total_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': loss.item()})
 
        avg_train_loss = total_loss / len(train_loader.dataset)
        avg_val_loss, rho = evaluate_mem_model(model, val_loader, device, criterion)

        print(f'Epoch {epoch+1}: '
              f'Train MSE: {avg_train_loss:.4f}  |  '
              f'Val MSE: {avg_val_loss:.4f}  |  '
              f'Spearman ρ: {rho:.4f}')
 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  -> Checkpoint saved (epoch {epoch+1})')
        else:
            epochs_no_improve += 1
 
        if epochs_no_improve == early_stopping_patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            test_loss, test_rho = evaluate_mem_model(model, test_loader, device, criterion)
            print(f'Final Test MSE: {test_loss:.4f}  |  Final Test Spearman ρ: {test_rho:.4f}')
            break


def run_mem_training(config):
    """Run memorability training with the given configuration dict."""
    seed_everything(config['random_seed'])
 
    train_dataset = MemDataset(csv_file=config['train_csv'],
                               img_root=config.get('img_root', ''))
    val_dataset   = MemDataset(csv_file=config['val_csv'],
                               img_root=config.get('img_root', ''))
    test_dataset   = MemDataset(csv_file=config['test_csv'],
                               img_root=config.get('img_root', ''))
 
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True) #,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              shuffle=False) #, num_workers=4, pin_memory=True)
    test_loader   = DataLoader(test_dataset,   batch_size=config['batch_size'],
                                shuffle=False) #, num_workers=4, pin_memory=True)

    model = CLIPHBAMem(
        backbone_checkpoint=config['backbone_checkpoint'],
        backbone_name=config['backbone'],
        vision_layers=config['vision_layers'],
        transformer_layers=config['transformer_layers'],
        rank=config['rank'],
    )
 
    if config['cuda'] == -1:
        device = torch.device('cuda')
    elif config['cuda'] == 0:
        device = torch.device('cuda:0')
    elif config['cuda'] == 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    # Optimizer created before DataParallel wrapping so mlp_parameters() is accessible
    optimizer = torch.optim.AdamW(model.mlp_parameters(), lr=config['lr'])

    # Use DataParallel if using all GPUs
    if config['cuda'] == -1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = DataParallel(model)

    model.to(device)
 
    print('\nModel Configuration:')
    print('--------------------')
    for key, value in config.items():
        print(f'  {key}: {value}')
    print(f'\nTrainable parameters: {count_trainable_parameters(model):,}\n')

    train_mem_model(
        model, train_loader, val_loader, device,
        optimizer, config['criterion'],
        config['epochs'],
        config['early_stopping_patience'],
        config['checkpoint_path'],
    )