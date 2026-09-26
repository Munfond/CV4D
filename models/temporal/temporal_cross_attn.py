"""
Temporal Cross-Attention / Fusion Queue
Kết hợp đặc trưng hiện tại F(t) và đặc trưng quá khứ F'(t-1) đã gióng hàng
"""
import torch
import torch.nn as nn

class TemporalFusionQueue(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        # Kết hợp Channel Mixing + Residual Convolution (nhẹ hơn full 3D attention cho Kaggle)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # Nhánh Attention Gate
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, curr_feat, prev_feat_aligned=None):
        """
        curr_feat: [B, C, H, W]
        prev_feat_aligned: [B, C, H, W]
        """
        if prev_feat_aligned is None:
            return curr_feat
            
        concat = torch.cat([curr_feat, prev_feat_aligned], dim=1) # [B, 2C, H, W]
        g = self.gate(concat)
        fused = self.fusion(concat)
        
        # Residual fusion có cổng điều khiển
        out = curr_feat + g * fused
        return out
