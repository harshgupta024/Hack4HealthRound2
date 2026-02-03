"""
MULTI-TASK TRAINING SCRIPT (FINAL FIXED VERSION)
Tasks:
1. Segmentation (pixel-wise)
2. Classification (caries / normal)
3. Severity regression
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

# =========================================================
# MODEL
# =========================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MultiTaskDentalNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder (Segmentation)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)

        self.seg_head = nn.Conv2d(64, 1, 1)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # Severity head
        self.sev_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Heads
        cls_out = self.cls_head(b)
        sev_out = self.sev_head(b)

        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        seg_out = self.seg_head(d1)

        return seg_out, cls_out, sev_out


# =========================================================
# LOSS
# =========================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, seg_w=1.0, cls_w=0.5, sev_w=0.3):
        super().__init__()
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.sev_loss = nn.MSELoss()
        self.seg_w = seg_w
        self.cls_w = cls_w
        self.sev_w = sev_w

    def forward(self, seg_p, cls_p, sev_p, seg_gt, cls_gt, sev_gt):
        l1 = self.seg_loss(seg_p, seg_gt)
        l2 = self.cls_loss(cls_p, cls_gt)
        l3 = self.sev_loss(sev_p.squeeze(), sev_gt.squeeze())
        total = self.seg_w*l1 + self.cls_w*l2 + self.sev_w*l3
        return total, l1, l2, l3


# =========================================================
# DATASET (FIXED)
# =========================================================

class MultiTaskDentalDataset(Dataset):
    def __init__(self, carries_dir, normal_dir, transform=None):
        self.transform = transform
        self.samples = []

        def is_img(f):
            return f.lower().endswith(('.png', '.jpg', '.jpeg'))

        for img in os.listdir(carries_dir):
            if is_img(img) and 'mask' not in img.lower():
                self.samples.append({
                    'img': os.path.join(carries_dir, img),
                    'mask': None,
                    'cls': 1,
                    'sev': 0.7
                })

        for img in os.listdir(normal_dir):
            if is_img(img) and 'mask' not in img.lower():
                self.samples.append({
                    'img': os.path.join(normal_dir, img),
                    'mask': None,
                    'cls': 0,
                    'sev': 0.0
                })

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = cv2.imread(s['img'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img = out['image']
            mask = out['mask']

        return (
            img,
            mask.unsqueeze(0).float(),
            torch.tensor(s['cls'], dtype=torch.long),
            torch.tensor([s['sev']], dtype=torch.float32)
        )


# =========================================================
# TRAINING
# =========================================================

def get_tfms(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--carries_dir', default='data/Carries')
    parser.add_argument('--normal_dir', default='data/Normal')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = MultiTaskDentalDataset(
        args.carries_dir,
        args.normal_dir,
        get_tfms(256)
    )

    train_len = int(0.8 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = MultiTaskDentalNet().to(device)
    loss_fn = MultiTaskLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for ep in range(args.epochs):
        model.train()
        correct = total = 0

        for img, mask, cls, sev in tqdm(train_dl, desc=f"Epoch {ep+1}"):
            img, mask, cls, sev = img.to(device), mask.to(device), cls.to(device), sev.to(device)

            opt.zero_grad()
            seg, c, s = model(img)
            loss, _, _, _ = loss_fn(seg, c, s, mask, cls, sev)
            loss.backward()
            opt.step()

            pred = torch.argmax(c, 1)
            correct += (pred == cls).sum().item()
            total += cls.size(0)

        acc = 100 * correct / total
        print(f"Epoch {ep+1}: Train Acc = {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_multitask_model.pth")

    print("Training finished. Best Accuracy:", best_acc)


if __name__ == "__main__":
    train()
