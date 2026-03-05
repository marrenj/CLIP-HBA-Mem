import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd
from train_mem_pipeline import MemDataset

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


def evaluate_perceptclip(model, data_loader, device, criterion, save_csv_path=None):
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