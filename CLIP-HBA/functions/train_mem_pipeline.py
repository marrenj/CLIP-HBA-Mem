import csv
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import sys
import numpy as np
import datetime
from torch.nn import functional as F
from tqdm import tqdm
from scipy.stats import spearmanr

from transformers import CLIPModel
from peft import LoraConfig, get_peft_model

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
        return row['image_path'],image, score


class PerceptCLIPDataset(Dataset):
    """Dataset for image memorability prediction.
 
    Expects a CSV with columns:
        image_path  - absolute path or relative to img_root
        score       - memorability score in [0, 1]
    """
 
    def __init__(self, csv_file, img_root=''):
        self.img_root = img_root
        self.transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=(224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                             std=(0.26862954, 0.26130258, 0.27577711))
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
        return row['image_path'],image, score


class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim1=512, hidden_dim2=256, output_dim=1 ,dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class clip_lora_model(nn.Module):
    def __init__(self, input_dim=768, hidden_dim1=512, hidden_dim2=256, output_dim=1,dropout_rate=0.5,r=16,lora_alpha=8):
        super(clip_lora_model, self).__init__()
        self.output_dim=output_dim
        self.mlp = MLP(input_dim, hidden_dim1, hidden_dim2, output_dim,dropout_rate)

        model_name = 'openai/clip-vit-large-patch14'
        model = CLIPModel.from_pretrained(model_name)
        self.proj = model.visual_projection 
        for param in self.proj.parameters():
            param.requires_grad = False
        encoder = model.vision_model
        target_modules = ["k_proj", "v_proj", "q_proj"]
        config = LoraConfig(
        r=int(r),
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        )
        self.model = get_peft_model(encoder, config)
        
    def forward(self, x):
        model_outputs = self.model(x)
        image_embeds = model_outputs[1]
        model_outputs = self.proj(image_embeds)
        outputs = self.mlp(model_outputs)
        return outputs


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
        print(f'[Backbone] Loaded {len(state_dict)} keys from {backbone_checkpoint}')

        for p in self.backbone.parameters():
            p.requires_grad = False
        n_frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        print(f'[Backbone] {n_frozen} parameters frozen')


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
            emb = self.backbone.clip_model.encode_image(x, self.backbone.pos_embedding)
 
        h = self.fc1(emb);  h = self.relu1(h);  h = self.dropout(h)
        h = self.fc2(h);    h = self.relu2(h);  h = self.dropout(h)
        return torch.sigmoid(self.fc3(h))  # [B, 1]

    def mlp_parameters(self):
        """Returns only the MLP head parameters (used by the optimiser)."""
        head_names = {'fc1', 'relu1', 'dropout', 'fc2', 'relu2', 'fc3'}
        return [p for n, p in self.named_parameters()
                if n.split('.')[0] in head_names]


def evaluate_mem_model(model, data_loader, device, criterion, save_csv_path=None):
    """Returns avg MSE loss and Spearman rank correlation on data_loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_image_paths = []

    with torch.no_grad(), tqdm(data_loader, desc='Evaluating', leave=False) as pbar:
        for image_paths, images, targets in pbar:
            images  = images.to(device)
            targets = targets.to(device)
 
            preds = model(images)
            loss  = criterion(preds, targets)
            total_loss += loss.item() * images.size(0)
 
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_image_paths.extend(image_paths)
            pbar.set_postfix({'loss': loss.item()})
 
    avg_loss = total_loss / len(data_loader.dataset)
    rho, _ = spearmanr(all_preds, all_targets)

    pd.DataFrame({
        'image_path': all_image_paths,
        'pred_score': all_preds,
        'true_score': all_targets,
    }).to_csv(save_csv_path, index=False)
    return avg_loss, rho


def train_mem_model(model, train_loader, val_loader, device, optimizer, criterion,
                    epochs, early_stopping_patience=10,
                    checkpoint_path='clip_hba_mem.pth',
                    test_loader=None,
                    fold=1, preds_dir=None):
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if preds_dir is not None:
        preds_dir = os.path.join(preds_dir, run_timestamp)
        os.makedirs(preds_dir, exist_ok=True)

    # initial eval — epoch 000
    save_path = os.path.join(preds_dir, f'epoch_000_fold{fold}.csv') if preds_dir else None

    model.train()
    best_val_loss = float('inf')
    best_rho = float('-inf')
    epochs_no_improve = 0

    history_path = f'{checkpoint_path}_fold{fold}_{run_timestamp}_history.csv'
    history_fields = ['epoch', 'train_loss', 'val_loss', 'spearman_rho']
    history_file = open(history_path, 'w', newline='')
    history_writer = csv.DictWriter(history_file, fieldnames=history_fields)
    history_writer.writeheader()
    history_file.flush()
 
    # Initial evaluation
    print('*' * 40)
    print('Initial evaluation')
    best_val_loss, best_rho = evaluate_mem_model(model, val_loader, device, criterion)
    print(f'Val MSE: {best_val_loss:.4f}  |  Spearman r: {best_rho:.4f}')
    print('*' * 40 + '\n')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        save_path = os.path.join(preds_dir, f'epoch_{epoch+1:03d}_fold{fold}.csv') if preds_dir else None

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}') as pbar:
            for _, images, targets in pbar:
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
        avg_val_loss, rho = evaluate_mem_model(model, val_loader, device, criterion, save_path)

        print(f'Epoch {epoch+1}: '
              f'Train MSE: {avg_train_loss:.4f}  |  '
              f'Val MSE: {avg_val_loss:.4f}  |  '
              f'Spearman r: {rho:.4f}')

        history_writer.writerow({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'spearman_rho': rho,
        })
        history_file.flush()

        if rho > best_rho:
            best_rho = rho
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'{checkpoint_path}_fold{fold}.pth')
            print(f'  -> Checkpoint saved (epoch {epoch+1})')
        else:
            epochs_no_improve += 1
        print(f'Epochs without improvement: {epochs_no_improve}')

        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     epochs_no_improve = 0
        #     torch.save(model.state_dict(), f'{checkpoint_path}_fold{fold}.pth')
        #     print(f'  -> Checkpoint saved (epoch {epoch+1})')
        # else:
        #     epochs_no_improve += 1
        # print(f'Epochs without improvement: {epochs_no_improve}')

        if epochs_no_improve == early_stopping_patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            test_loss, test_rho = evaluate_mem_model(model, test_loader, device, criterion)
            print(f'Final Test MSE: {test_loss:.4f}  |  Final Test Spearman r: {test_rho:.4f}')
            history_file.close()
            break

    else:
        history_file.close()


def run_mem_training(config):
    """Run memorability training with the given configuration dict."""
    log_path = config.get('log_path', None)
    if log_path:
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fold = config.get('fold', 1)
        base, ext = os.path.splitext(log_path)
        log_path = f'{base}_fold{fold}_{run_timestamp}{ext}'
        log_file = open(log_path, 'w', encoding='utf-8', buffering=1)  # line-buffered
        sys.stdout = log_file

    seed_everything(config['random_seed'])

    if config['model_type'] == 'perceptclip':
        train_dataset = PerceptCLIPDataset(csv_file=config['train_csv'],
                               img_root=config.get('img_root', ''))
        val_dataset   = PerceptCLIPDataset(csv_file=config['val_csv'],
                               img_root=config.get('img_root', ''))
        test_dataset   = PerceptCLIPDataset(csv_file=config['test_csv'],
                               img_root=config.get('img_root', ''))
    else:
        train_dataset = MemDataset(csv_file=config['train_csv'],
                                   img_root=config.get('img_root', ''))
        val_dataset   = MemDataset(csv_file=config['val_csv'],
                                   img_root=config.get('img_root', ''))
        test_dataset   = MemDataset(csv_file=config['test_csv'],
                                   img_root=config.get('img_root', ''))
 

    print(f'\n[Data] Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples | Test: {len(test_dataset)} samples')

    _, img0, score0 = train_dataset[0]
    print(f'\n[Data] sample image tensor shape: {tuple(img0.shape)}')

    print(f'\n[Data] sample score: {score0.item():.4f}')

    scores = train_dataset.annotations['score']
    print(f'\n[Data] Score range: min {scores.min():.4f} to max {scores.max():.4f}')
 
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=8, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=8, pin_memory=True,
                              persistent_workers=True)
    test_loader   = DataLoader(test_dataset,   batch_size=config['batch_size'],
                                shuffle=False, num_workers=8, pin_memory=True,
                                persistent_workers=True)

    if config['model_type'] == 'clip_hba_mem':
        model = CLIPHBAMem(
        backbone_checkpoint=config['backbone_checkpoint'],
        backbone_name=config['backbone'],
        vision_layers=config['vision_layers'],
        transformer_layers=config['transformer_layers'],
        rank=config['rank'],
    )
    elif config['model_type'] == 'perceptclip':
        model = clip_lora_model()
    else:
        raise ValueError(f'Invalid model type: {config["model_type"]}')
 
    if config['cuda'] == -1:
        device = torch.device('cuda')
    elif config['cuda'] == 0:
        device = torch.device('cuda:0')
    elif config['cuda'] == 1:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    # Use DataParallel if using all GPUs
    if config['cuda'] == -1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = DataParallel(model)

    model.to(device)

    print('\n[Model] Probe forward pass...')
    model.eval()
    with torch.no_grad():
        _dummy_image = torch.randn(2, 3, 224, 224).to(device)
        if config['model_type'] == 'clip_hba_mem':
            if isinstance(model, CLIPHBAMem):
                _emb = model.backbone.clip_model.encode_image(_dummy_image, model.backbone.pos_embedding)
                print(f'[Model] encode_image output shape: {tuple(_emb.shape)}')
            elif isinstance(model, clip_lora_model):
                _emb = model.proj(_dummy_image)
                print(f'[Model] proj output shape: {tuple(_emb.shape)}')
            else:
                raise ValueError(f'Invalid model type: {config["model_type"]}')
        _out = model(_dummy_image)
        print(f'[Model] MLP output shape:          {tuple(_out.shape)}')
        print(f'[Model] Output range:              [{_out.min().item():.4f}, {_out.max().item():.4f}]')
    model.train()

    if isinstance(model, CLIPHBAMem):
        opt_params = model.mlp_parameters()
    else:
        opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=config['lr'])
 
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
        test_loader,
        config['fold'],
        preds_dir=config.get('preds_dir', None),
    )