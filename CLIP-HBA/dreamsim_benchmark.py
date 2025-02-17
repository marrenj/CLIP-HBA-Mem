from functions.train_behavior_things_pipeline import *

import torch
import os
from tqdm import tqdm
import logging
import numpy as np
import json
import torch.nn.functional as F

import sys
sys.path.append('../../dreamsim_benchmark/dreamsim')

from dataset.dataset import TwoAFCDataset


def score_nights_dataset(model, test_loader, device):
    logging.info("Evaluating NIGHTS dataset.")
    d0s = []
    d1s = []
    targets = []

    with torch.no_grad():
        for i, (img_ref, img_left, img_right, target, idx) in tqdm(enumerate(test_loader), total=len(test_loader)):
            img_ref, img_left, img_right, target = img_ref.to(device), img_left.to(device), \
                img_right.to(device), target.to(device)

            dist_0 = model(img_ref, img_left)
            dist_1 = model(img_ref, img_right)

            if len(dist_0.shape) < 1:
                dist_0 = dist_0.unsqueeze(0)
                dist_1 = dist_1.unsqueeze(0)
            dist_0 = dist_0.unsqueeze(1)
            dist_1 = dist_1.unsqueeze(1)
            target = target.unsqueeze(1)

            d0s.append(dist_0)
            d1s.append(dist_1)
            targets.append(target)

    d0s = torch.cat(d0s, dim=0)
    d1s = torch.cat(d1s, dim=0)
    targets = torch.cat(targets, dim=0)
    scores = (d0s < d1s) * (1.0 - targets) + (d1s < d0s) * targets + (d1s == d0s) * 0.5
    twoafc_score = torch.mean(scores, dim=0)
    print(f"2AFC score: {str(twoafc_score)}")
    return twoafc_score

def triplet_forward(img_ref, image_left):
    img_feature1 = model(img_ref)
    # print(img_feature1.shape)
    img_feature2 = model(image_left)

    dist = 1 - F.cosine_similarity(img_feature1, img_feature2)
    return dist



if __name__ == '__main__':
    ##############################
    load_hba = True
    batch_size = 256
    ##############################

    classnames=classnames66
    classnames = [x[0] for x in classnames]
    model_path = f'./models/cliphba_dora66.pth'
    root_dir = "../../dreamsim_benchmark/dreamsim/dataset/nights"
    val_dataset = TwoAFCDataset(root_dir, split = 'val', load_size = 224, preprocess = 'CLIP-HBA')
    test_dataset = TwoAFCDataset(root_dir, split = 'test', load_size = 224, preprocess = 'CLIP-HBA')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    model = CLIPHBA(classnames=classnames, backbone_name='ViT-L/14', pos_embedding=True)


    if load_hba:
        apply_dora_to_ViT(model, n_vision_layers=2, n_transformer_layers=1, r=32, dora_dropout=0.1)
        model_state_dict = torch.load(model_path)
        adjusted_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        model.load_state_dict(adjusted_state_dict)
        print("fine-tuned model loaded")
    else:
        print(f"Using Original CLIP")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Validation:")
    score_nights_dataset(triplet_forward, val_loader, device)
    print("Test:")
    score_nights_dataset(triplet_forward, test_loader, device)

