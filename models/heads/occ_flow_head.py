"""
Dual-Head: 3D Semantic Occupancy (18 classes) & Occupancy Flow (vx, vy, vz)
Nâng chiều từ 2D BEV lên không gian thể tích 3D Voxel
"""
import torch
import torch.nn as nn

class DualOccFlowHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=18, z_dim=16):
        super().__init__()
        self.z_dim = z_dim
        
        # 1. Nâng chiều thể tích bằng 3D Deconvolution
        # Từ [B, C//16, 16, H, W] lên [B, 64, 16, H, W] -> [B, 32, 16, H, W]
        self.deconv3d = nn.Sequential(
            nn.ConvTranspose3d(in_channels // z_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # 2. Head 1: Phân loại ngữ nghĩa 3D Occupancy (17 classes + 1 Free Space)
        self.occ_classifier = nn.Conv3d(32, num_classes, kernel_size=1)
        
        # 3. Head 2: Dự đoán vector dịch chuyển 3D (dx, dy, dz)
        self.flow_regressor = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 3, kernel_size=1) # 3 channels: dx, dy, dz
        )

    def forward(self, bev_feature):
        """
        bev_feature: [B, C, H, W] -> ví dụ [B, 256, 200, 200]
        """
        B, C, H, W = bev_feature.shape
        # Reshape thành tensor 5D
        feat_3d = bev_feature.view(B, C // self.z_dim, self.z_dim, H, W)
        feat_3d = self.deconv3d(feat_3d) # [B, 32, 16, H, W]
        
        occ_logits = self.occ_classifier(feat_3d)   # [B, 18, 16, H, W]
        flow_vectors = self.flow_regressor(feat_3d) # [B, 3, 16, H, W]
        
        return occ_logits, flow_vectors
