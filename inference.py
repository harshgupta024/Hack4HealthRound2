import os
import cv2
import torch
import numpy as np
from multitask_model import MultiTaskDentalNet
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# -----------------------------
# Config
# -----------------------------
IMAGE_PATH = "data/Carries/img_001.jpg"
MODEL_PATH = "models/best_multitask_model.pth"
OUTPUT_DIR = "outputs"

os.makedirs(f"{OUTPUT_DIR}/segmentation", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/classification", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/overlays", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
model = MultiTaskDentalNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Preprocessing
# -----------------------------
transform = Compose([
    Resize(256, 256),
    Normalize(),
    ToTensorV2()
])

img = cv2.imread(IMAGE_PATH)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

data = transform(image=img_rgb)
input_tensor = data["image"].unsqueeze(0).to(device)

# -----------------------------
# Inference
# -----------------------------
with torch.no_grad():
    seg_out, cls_out, sev_out = model(input_tensor)

# -----------------------------
# Segmentation Output
# -----------------------------
seg_mask = (torch.sigmoid(seg_out)[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255

mask_path = f"{OUTPUT_DIR}/segmentation/segmentation_mask.png"
cv2.imwrite(mask_path, seg_mask)

# -----------------------------
# Classification Output
# -----------------------------
cls_prob = torch.softmax(cls_out, dim=1)[0]
pred_class = torch.argmax(cls_prob).item()

label = "Caries" if pred_class == 1 else "Normal"
confidence = cls_prob[pred_class].item()

with open(f"{OUTPUT_DIR}/classification/predictions.txt", "w") as f:
    f.write(f"Prediction: {label}\n")
    f.write(f"Confidence: {confidence:.4f}\n")
    f.write(f"Severity Score: {sev_out.item():.4f}\n")

# -----------------------------
# Overlay Visualization
# -----------------------------
overlay = img_rgb.copy()
overlay[seg_mask == 255] = [255, 0, 0]

overlay_path = f"{OUTPUT_DIR}/overlays/overlay.png"
cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print("Inference complete!")
print("Segmentation mask saved at:", mask_path)
print("Classification:", label, "Confidence:", confidence)
