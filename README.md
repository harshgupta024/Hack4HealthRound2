# 🦷 Hack4Health Round 2  
## Multi-Task Learning for Dental Caries Segmentation & Classification

This repository contains a **multi-task deep learning project** that performs **pixel-wise segmentation** of dental caries from X-ray images and **binary classification** of whether a tooth has caries or not.  
The system also estimates a **severity score** for caries.

---

## 🧠 Problem Statement

Dental caries detection in X-ray images is challenging due to:
- Low contrast between healthy and carious regions
- Noise and variation in dental anatomy
- Hidden or partially visible lesions

The goal of this project is to build a **single model** that can:
1. Localize carious regions (Segmentation)
2. Diagnose caries presence (Classification)
3. Estimate caries severity (Regression)

All tasks are performed from a **single input X-ray image**.

---

## 📁 Repository Structure

Hack4HealthRound2/
├── data/
│ ├── Carries/
│ └── Normal/
├── models/
│ └── best_multitask_model.pth
├── outputs/
│ ├── segmentation/
│ ├── classification/
│ └── overlays/
├── train_multitask.py
├── multitask_model.py
├── inference.py
├── generate_complete_demo.py
├── requirements.txt
└── README.md


---

## 🧰 Requirements

Install dependencies:

```bash
pip install -r requirements.txt

This includes:

    torch

    torchvision

    albumentations

    opencv-python

    numpy

    matplotlib

    tqdm

⚙️ Training the Multi-Task Model
Run training:

python train_multitask.py

Training automatically splits the dataset into train/validation (80/20).
Outputs include:

    Best model saved: models/best_multitask_model.pth

    Training progress on console (accuracy, losses)

🚀 Running Inference

To generate segmentation masks + classification on a new image:

    Open inference.py

    Set:

IMAGE_PATH = "data/Carries/img_001.jpg"

    Run:

python inference.py

Outputs will be saved in:

outputs/segmentation/
outputs/classification/
outputs/overlays/

🔍 What You Get from Inference
1️⃣ Segmentation Output

Saved as:

outputs/segmentation/segmentation_mask.png

2️⃣ Classification Output

Saved as:

outputs/classification/predictions.txt

Example:

Prediction: Caries
Confidence: 0.9341
Severity Score: 0.7032

3️⃣ Overlay Visualization

Saved as:

outputs/overlays/overlay.png

Overlay shows predicted caries region in red.
🧠 Model Architecture

The multi-task network consists of:

    Shared encoder

    Segmentation head (U-Net decoder style)

    Classification head (FC layers)

    Severity head (regression output)

This allows shared feature learning across tasks.
📊 Evaluation & Metrics

While demo generation uses synthetic outputs for visualization and PPT, core evaluation metrics include:
📍 Segmentation

    Dice Similarity Coefficient

    Intersection over Union (IoU)

    Pixel-wise Accuracy

    Sensitivity, Specificity

    Hausdorff Distance

📍 Classification

    Accuracy

    Precision, Recall

    F1-Score

    AUC (ROC curve)

    Confusion Matrix

👩‍🏫 Usage Guide (Judges / Presentation)

To train:

python train_multitask.py

To visualize demo outputs without training (fast):

python generate_complete_demo.py

To run inference on a single image:

python inference.py

This produces outputs proving that the model performs both segmentation and classification from one input.
🧪 Example Output Files

outputs/segmentation/segmentation_mask.png
outputs/classification/predictions.txt
outputs/overlays/overlay.png

💡 Future Work

    Add multi-class segmentation (teeth, decay + restorations)

    3D dental scan (CBCT) support

    Real-time clinical dashboard

    Export to ONNX / CoreML for deployment
