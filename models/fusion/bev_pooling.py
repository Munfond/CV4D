"""
BEV Transformation: Chuyển đổi đặc trưng ảnh 2D sang mặt phẳng 3D BEV
Áp dụng cơ chế Lift-Splat-Shoot (LSS) với Depth Distribution dự đoán
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraBEVPooling(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, bev_size=(200, 200), depth_bins=40):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.depth_bins = depth_bins
        
        # Dự đoán phân bố độ sâu rời rạc (Discrete Depth Distribution)
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, depth_bins, kernel_size=1)
        )
        
        # Tầng nén và tinh chỉnh BEV
        self.bev_projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(bev_size)
        )

    def forward(self, cam_features):
        """
        cam_features: [B, N_cams, C, H, W]
        """
        B, N, C, H, W = cam_features.shape
        x = cam_features.view(B * N, C, H, W)
        
        # 1. Dự đoán phân bố xác suất chiều sâu (Depth distribution Dis)
        depth_logits = self.depth_net(x) # [B*N, D, H, W]
        depth_probs = F.softmax(depth_logits, dim=1)
        
        # 2. Outer product: Trọng số hóa đặc trưng theo độ sâu
        feat_expanded = x.unsqueeze(2) # [B*N, C, 1, H, W]
        depth_expanded = depth_probs.unsqueeze(1) # [B*N, 1, D, H, W]
        frustum_feat = (feat_expanded * depth_expanded).mean(dim=2) # [B*N, C, H, W]
        
        # 3. Gom các camera quanh xe về không gian BEV chung
        frustum_feat = frustum_feat.view(B, N, C, H, W).mean(dim=1) # [B, C, H, W]
        bev_feature = self.bev_projection(frustum_feat) # [B, out_channels, bev_h, bev_w]
        
        return bev_feature
