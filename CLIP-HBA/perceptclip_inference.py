import os
import csv
from datetime import datetime
from torchvision import transforms
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
import importlib.util
import numpy as np
from scipy.stats import spearmanr, pearsonr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model class definition dynamically
class_path = hf_hub_download(repo_id="PerceptCLIP/PerceptCLIP_Memorability", filename="modeling.py")
spec = importlib.util.spec_from_file_location("modeling", class_path)
modeling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modeling)

# initialize a model
ModelClass = modeling.clip_lora_model
model = ModelClass().to(device)

# Load pretrained model
model_path = hf_hub_download(repo_id="PerceptCLIP/PerceptCLIP_Memorability", filename="perceptCLIP_Memorability.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocess
def Mem_preprocess():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])
    return transform

transform = Mem_preprocess()

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Data", "lamem", "lamem_test_1.csv")
image_dir = os.path.join(script_dir, "Data", "lamem", "images")
preds_dir = os.path.join(script_dir, "preds")
os.makedirs(preds_dir, exist_ok=True)

# Read test set
with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    entries = list(reader)

print(f"Running inference on {len(entries)} images...")

predictions = []
ground_truths = []
results = []
skipped = 0

for i, entry in enumerate(entries):
    img_name = entry["image_path"]
    gt_score = float(entry["score"])
    img_path = os.path.join(image_dir, img_name)

    if not os.path.exists(img_path):
        skipped += 1
        continue

    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_score = model(image_tensor).item()

    predictions.append(pred_score)
    ground_truths.append(gt_score)
    results.append({"image_path": img_name, "ground_truth": gt_score, "prediction": pred_score})

    if (i + 1) % 500 == 0:
        print(f"  Processed {i + 1}/{len(entries)} images...")

# Save results CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = os.path.join(preds_dir, f"perceptclip_{timestamp}.csv")
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "ground_truth", "prediction"])
    writer.writeheader()
    writer.writerows(results)

# Compute metrics
predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

spearman_rho, spearman_p = spearmanr(predictions, ground_truths)
pearson_r, pearson_p = pearsonr(predictions, ground_truths)
mse = np.mean((predictions - ground_truths) ** 2)

print(f"\n--- Results ({len(predictions)} images, {skipped} skipped) ---")
print(f"Spearman rho:  {spearman_rho:.4f} (p={spearman_p:.2e})")
print(f"Pearson r:     {pearson_r:.4f} (p={pearson_p:.2e})")
print(f"MSE:           {mse:.6f}")
print(f"\nPredictions saved to: {output_csv}")
