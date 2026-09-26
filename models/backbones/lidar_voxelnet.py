"""
LiDAR Backbone: PointPillars / Pillar Feature Net
Chuyển đổi đám mây điểm LiDAR [N, 4] thành bản đồ đặc trưng BEV [B, C, H, W]
Thuần PyTorch, không lo lỗi cài đặt CUDA trên Kaggle.
"""
import torch
import torch.nn as nn

class LiDARPillarNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=256, bev_size=(200, 200), pc_range=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.pc_range = pc_range
        self.x_min, self.y_min, self.z_min = pc_range[0], pc_range[1], pc_range[2]
        self.x_max, self.y_max, self.z_max = pc_range[3], pc_range[4], pc_range[5]
        
        # Mạng chiếu điểm cục bộ (Point-wise MLP)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 3, 64),  # thêm offset [x-xc, y-yc, z-zc]
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        
        # Mạng tích chập 2D BEV
        self.bev_conv = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, point_clouds):
        """
        point_clouds: Danh sách các Tensor [N_points, 4] (mỗi phần tử là 1 sample trong batch)
        Trả về: [B, out_channels, H_bev, W_bev]
        """
        device = next(self.parameters()).device
        B = len(point_clouds)
        bev_maps = []

        for b in range(B):
            pts = point_clouds[b]
            if pts is None or pts.shape[0] == 0:
                bev_maps.append(torch.zeros((1, 128, self.bev_h, self.bev_w), device=device))
                continue

            pts = pts.to(device)
            # Lọc điểm nằm trong phạm vi ROI
            mask = (pts[:, 0] >= self.x_min) & (pts[:, 0] < self.x_max) & \
                   (pts[:, 1] >= self.y_min) & (pts[:, 1] < self.y_max) & \
                   (pts[:, 2] >= self.z_min) & (pts[:, 2] < self.z_max)
            pts = pts[mask]

            if pts.shape[0] == 0:
                bev_maps.append(torch.zeros((1, 128, self.bev_h, self.bev_w), device=device))
                continue

            # Tính chỉ số pillar trên lưới BEV
            x_indices = ((pts[:, 0] - self.x_min) / (self.x_max - self.x_min) * self.bev_w).long().clamp(0, self.bev_w - 1)
            y_indices = ((pts[:, 1] - self.y_min) / (self.y_max - self.y_min) * self.bev_h).long().clamp(0, self.bev_h - 1)

            # Tính offset điểm so với tâm pillar
            x_center = (x_indices.float() + 0.5) * ((self.x_max - self.x_min) / self.bev_w) + self.x_min
            y_center = (y_indices.float() + 0.5) * ((self.y_max - self.y_min) / self.bev_h) + self.y_min
            z_center = (self.z_max + self.z_min) / 2.0
            
            offsets = torch.stack([pts[:, 0] - x_center, pts[:, 1] - y_center, pts[:, 2] - z_center], dim=-1)
            feat_in = torch.cat([pts[:, :4], offsets], dim=-1)

            # Đưa qua MLP
            feat_pts = self.mlp(feat_in) # [N, 128]

            # Scatter Max-pooling vào lưới BEV
            linear_indices = y_indices * self.bev_w + x_indices
            bev_flat = torch.zeros((self.bev_h * self.bev_w, 128), device=device)
            bev_flat.scatter_reduce_(0, linear_indices.unsqueeze(-1).expand(-1, 128), feat_pts, reduce="amax", include_self=False)
            
            bev_dense = bev_flat.view(self.bev_h, self.bev_w, 128).permute(2, 0, 1).unsqueeze(0)
            bev_maps.append(bev_dense)

        bev_tensor = torch.cat(bev_maps, dim=0) # [B, 128, H, W]
        return self.bev_conv(bev_tensor) # [B, out_channels, H, W]
