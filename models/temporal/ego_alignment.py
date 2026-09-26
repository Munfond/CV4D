"""
Tầng Gióng hàng Tọa độ Thời gian (Ego-Spatial Alignment)
Thực hiện công thức Eq. 9 trong bài báo Survey:
F'_{t-1} = Psi_S(T_(t-1 -> t) * F_(t-1))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class EgoSpatialAlignment(nn.Module):
    def __init__(self, bev_size=(200, 200), voxel_range=[-40.0, -40.0, 40.0, 40.0]):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.x_min, self.y_min, self.x_max, self.y_max = voxel_range

    def forward(self, prev_feature, delta_transform=None):
        """
        prev_feature: Tensor [B, C, H, W]
        delta_transform: Ma trận 4x4 hoặc 3x3 chuyển đổi Ego từ (t-1) sang (t)
        """
        if delta_transform is None:
            return prev_feature

        B, C, H, W = prev_feature.shape
        device = prev_feature.device
        
        # Sinh lưới chuẩn hóa [-1, 1]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([x, y, torch.zeros_like(x), torch.ones_like(x)], dim=-1) # [H, W, 4]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1).view(B, -1, 4) # [B, H*W, 4]

        # Áp dụng ma trận biến đổi thân xe
        delta_t = delta_transform.to(device).float()
        if delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(0).repeat(B, 1, 1)

        transformed_grid = torch.bmm(grid, delta_t.transpose(1, 2))[:, :, :2]
        transformed_grid = transformed_grid.view(B, H, W, 2)

        # Lấy mẫu nội suy đặc trưng (Feature Warping)
        aligned_feature = F.grid_sample(prev_feature, transformed_grid, align_corners=True, mode='bilinear')
        return aligned_feature
