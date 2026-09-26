"""
Radar Backbone: MLP Encoder xử lý các điểm Radar kèm vận tốc Doppler hướng tâm
Trích xuất đặc trưng BEV [B, C_radar, H, W]
"""
import torch
import torch.nn as nn

class RadarPointNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=64, bev_size=(200, 200), pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.pc_range = pc_range
        self.x_min, self.y_min, self.z_min = pc_range[0], pc_range[1], pc_range[2]
        self.x_max, self.y_max, self.z_max = pc_range[3], pc_range[4], pc_range[5]

        # MLP chiếu điểm Radar [x, y, z, rcs, vx_comp]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, radar_points):
        """
        radar_points: Danh sách các Tensor [N_radar_points, 5]
        """
        device = next(self.parameters()).device
        B = len(radar_points)
        bev_maps = []

        for b in range(B):
            pts = radar_points[b]
            if pts is None or pts.shape[0] == 0:
                bev_maps.append(torch.zeros((1, 64, self.bev_h, self.bev_w), device=device))
                continue

            pts = pts.to(device)
            mask = (pts[:, 0] >= self.x_min) & (pts[:, 0] < self.x_max) & \
                   (pts[:, 1] >= self.y_min) & (pts[:, 1] < self.y_max)
            pts = pts[mask]

            if pts.shape[0] == 0:
                bev_maps.append(torch.zeros((1, 64, self.bev_h, self.bev_w), device=device))
                continue

            feat = self.mlp(pts[:, :5]) # [N, out_channels]

            x_idx = ((pts[:, 0] - self.x_min) / (self.x_max - self.x_min) * self.bev_w).long().clamp(0, self.bev_w - 1)
            y_idx = ((pts[:, 1] - self.y_min) / (self.y_max - self.y_min) * self.bev_h).long().clamp(0, self.bev_h - 1)

            linear_indices = y_idx * self.bev_w + x_idx
            bev_flat = torch.zeros((self.bev_h * self.bev_w, 64), device=device)
            bev_flat.scatter_reduce_(0, linear_indices.unsqueeze(-1).expand(-1, 64), feat, reduce="amax", include_self=False)

            bev_dense = bev_flat.view(self.bev_h, self.bev_w, 64).permute(2, 0, 1).unsqueeze(0)
            bev_maps.append(bev_dense)

        bev_tensor = torch.cat(bev_maps, dim=0)
        return self.conv(bev_tensor)
