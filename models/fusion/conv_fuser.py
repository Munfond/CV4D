"""
ConvFuser: Dung hợp đa phương thức Camera + LiDAR + Radar trên không gian BEV
Tích hợp Squeeze-and-Excitation (SE-Block) và hỗ trợ Sensor Dropout cho bài toán Benchmark
"""
import torch
import torch.nn as nn

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weight = self.fc(x).view(b, c, 1, 1)
        return x * weight

class ConvFuser(nn.Module):
    def __init__(self, cam_channels=256, lidar_channels=256, radar_channels=64, out_channels=256):
        super().__init__()
        in_channels = cam_channels + lidar_channels + radar_channels
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SqueezeExcitationBlock(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, cam_bev, lidar_bev=None, radar_bev=None, mode='tri_modal'):
        """
        mode:
          - 'cam_only': Chỉ dùng Camera (gán 0 cho LiDAR & Radar)
          - 'cam_radar': Dùng Camera + Radar (gán 0 cho LiDAR)
          - 'cam_lidar': Dùng Camera + LiDAR (gán 0 cho Radar)
          - 'tri_modal': Đủ cả 3 cảm biến
        """
        device = cam_bev.device
        B, _, H, W = cam_bev.shape

        if lidar_bev is None or mode in ['cam_only', 'cam_radar']:
            lidar_bev = torch.zeros((B, 256, H, W), device=device)

        if radar_bev is None or mode in ['cam_only', 'cam_lidar']:
            radar_bev = torch.zeros((B, 64, H, W), device=device)

        # Nối đặc trưng dọc theo trục Channel
        concat_feat = torch.cat([cam_bev, lidar_bev, radar_bev], dim=1)
        return self.fusion_conv(concat_feat)
