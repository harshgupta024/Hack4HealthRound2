"""
MULTI-TASK DENTAL X-RAY MODEL
Takes ONE image as input and performs multiple tasks:
1. Segmentation (where is the caries/tooth)
2. Classification (has caries or not)
3. Severity prediction (how bad is it)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskDentalNet(nn.Module):
    """
    Multi-task model with shared encoder and multiple task-specific heads
    
    Input: Dental X-ray image (1, 3, 256, 256)
    Outputs:
        - Segmentation mask (1, 1, 256, 256) - where is the caries/tooth
        - Classification (1, 2) - caries/no caries
        - Severity score (1, 1) - how severe (0-1)
    """
    
    def __init__(self, n_channels=3):
        super(MultiTaskDentalNet, self).__init__()
        
        # ============================================================
        # SHARED ENCODER - Extracts features from input image
        # ============================================================
        self.enc1 = self.conv_block(n_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # ============================================================
        # TASK 1: SEGMENTATION HEAD (U-Net style decoder)
        # ============================================================
        self.seg_up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.seg_dec1 = self.conv_block(1024, 512)
        
        self.seg_up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.seg_dec2 = self.conv_block(512, 256)
        
        self.seg_up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.seg_dec3 = self.conv_block(256, 128)
        
        self.seg_up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.seg_dec4 = self.conv_block(128, 64)
        
        self.seg_out = nn.Conv2d(64, 1, 1)  # Output: segmentation mask
        
        # ============================================================
        # TASK 2: CLASSIFICATION HEAD (Caries vs No Caries)
        # ============================================================
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc1 = nn.Linear(1024, 512)
        self.cls_fc2 = nn.Linear(512, 256)
        self.cls_fc3 = nn.Linear(256, 2)  # Output: 2 classes (caries/no caries)
        self.cls_dropout = nn.Dropout(0.5)
        
        # ============================================================
        # TASK 3: SEVERITY REGRESSION HEAD (How severe is the caries)
        # ============================================================
        self.sev_pool = nn.AdaptiveAvgPool2d(1)
        self.sev_fc1 = nn.Linear(1024, 256)
        self.sev_fc2 = nn.Linear(256, 128)
        self.sev_fc3 = nn.Linear(128, 1)  # Output: severity score [0-1]
        self.sev_dropout = nn.Dropout(0.3)
        
    def conv_block(self, in_c, out_c):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass - ONE image produces THREE outputs
        
        Args:
            x: Input image (B, 3, 256, 256)
            
        Returns:
            segmentation: Segmentation mask (B, 1, 256, 256)
            classification: Class probabilities (B, 2)
            severity: Severity score (B, 1)
        """
        
        # ============================================================
        # SHARED ENCODER - Extract features
        # ============================================================
        e1 = self.enc1(x)           # 64 x 256 x 256
        e2 = self.enc2(self.pool(e1))  # 128 x 128 x 128
        e3 = self.enc3(self.pool(e2))  # 256 x 64 x 64
        e4 = self.enc4(self.pool(e3))  # 512 x 32 x 32
        e5 = self.enc5(self.pool(e4))  # 1024 x 16 x 16
        
        # ============================================================
        # TASK 1: SEGMENTATION
        # ============================================================
        # Decoder with skip connections
        d1 = self.seg_up1(e5)  # 512 x 32 x 32
        d1 = torch.cat([d1, e4], dim=1)  # 1024 x 32 x 32
        d1 = self.seg_dec1(d1)  # 512 x 32 x 32
        
        d2 = self.seg_up2(d1)  # 256 x 64 x 64
        d2 = torch.cat([d2, e3], dim=1)  # 512 x 64 x 64
        d2 = self.seg_dec2(d2)  # 256 x 64 x 64
        
        d3 = self.seg_up3(d2)  # 128 x 128 x 128
        d3 = torch.cat([d3, e2], dim=1)  # 256 x 128 x 128
        d3 = self.seg_dec3(d3)  # 128 x 128 x 128
        
        d4 = self.seg_up4(d3)  # 64 x 256 x 256
        d4 = torch.cat([d4, e1], dim=1)  # 128 x 256 x 256
        d4 = self.seg_dec4(d4)  # 64 x 256 x 256
        
        segmentation = self.seg_out(d4)  # 1 x 256 x 256
        
        # ============================================================
        # TASK 2: CLASSIFICATION
        # ============================================================
        cls_feat = self.cls_pool(e5).view(e5.size(0), -1)  # 1024
        cls_feat = F.relu(self.cls_fc1(cls_feat))  # 512
        cls_feat = self.cls_dropout(cls_feat)
        cls_feat = F.relu(self.cls_fc2(cls_feat))  # 256
        cls_feat = self.cls_dropout(cls_feat)
        classification = self.cls_fc3(cls_feat)  # 2
        
        # ============================================================
        # TASK 3: SEVERITY PREDICTION
        # ============================================================
        sev_feat = self.sev_pool(e5).view(e5.size(0), -1)  # 1024
        sev_feat = F.relu(self.sev_fc1(sev_feat))  # 256
        sev_feat = self.sev_dropout(sev_feat)
        sev_feat = F.relu(self.sev_fc2(sev_feat))  # 128
        sev_feat = self.sev_dropout(sev_feat)
        severity = torch.sigmoid(self.sev_fc3(sev_feat))  # 1 (0-1 range)
        
        return segmentation, classification, severity


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    Balances all three tasks
    """
    
    def __init__(self, seg_weight=1.0, cls_weight=0.5, sev_weight=0.5):
        super(MultiTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.sev_weight = sev_weight
        
        # Loss functions for each task
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        self.sev_loss = nn.MSELoss()
    
    def forward(self, seg_pred, cls_pred, sev_pred, seg_target, cls_target, sev_target):
        """
        Calculate combined loss
        
        Args:
            seg_pred: Predicted segmentation
            cls_pred: Predicted classification
            sev_pred: Predicted severity
            seg_target: Target segmentation mask
            cls_target: Target class (0 or 1)
            sev_target: Target severity (0-1)
        """
        seg_l = self.seg_loss(seg_pred, seg_target)
        cls_l = self.cls_loss(cls_pred, cls_target)
        sev_l = self.sev_loss(sev_pred, sev_target)
        
        total_loss = (self.seg_weight * seg_l + 
                     self.cls_weight * cls_l + 
                     self.sev_weight * sev_l)
        
        return total_loss, seg_l, cls_l, sev_l


def get_multitask_model(device='cuda'):
    """Initialize multi-task model"""
    model = MultiTaskDentalNet(n_channels=3)
    model = model.to(device)
    return model


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == '__main__':
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_multitask_model(device)
    
    print("="*60)
    print("MULTI-TASK DENTAL X-RAY MODEL")
    print("="*60)
    print(f"\nModel on device: {device}")
    
    # Test with dummy input
    dummy_input = torch.randn(4, 3, 256, 256).to(device)  # Batch of 4 images
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    seg_out, cls_out, sev_out = model(dummy_input)
    
    print(f"\nOutputs:")
    print(f"  1. Segmentation: {seg_out.shape} - Where is the caries/tooth")
    print(f"  2. Classification: {cls_out.shape} - Caries or No Caries")
    print(f"  3. Severity: {sev_out.shape} - How severe (0-1)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n" + "="*60)
    print("✓ Multi-task model ready!")
    print("="*60)