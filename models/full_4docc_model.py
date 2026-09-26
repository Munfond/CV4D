"""
Mô hình hoàn chỉnh: VinFast 4D-OccFusion Network
Tích hợp Đa cảm biến (Camera + LiDAR + Radar), Temporal Alignment và Dual Occ/Flow Head
"""
import torch
import torch.nn as nn

from configs.base_config import Config
from models.backbones.camera_resnet import CameraBackbone
from models.backbones.lidar_voxelnet import LiDARPillarNet
from models.backbones.radar_pointnet import RadarPointNet
from models.fusion.bev_pooling import CameraBEVPooling
from models.fusion.conv_fuser import ConvFuser
from models.temporal.ego_alignment import EgoSpatialAlignment
from models.temporal.temporal_cross_attn import TemporalFusionQueue
from models.heads.occ_flow_head import DualOccFlowHead

class VinFast4DOccModel(nn.Module):
    def __init__(self, config=Config, pretrained_cam=True):
        super().__init__()
        self.config = config
        
        # 1. Backbones
        self.camera_backbone = CameraBackbone(out_channels=config.CAM_FEAT_DIM, pretrained=pretrained_cam)
        self.cam_bev_pool = CameraBEVPooling(
            in_channels=config.CAM_FEAT_DIM,
            out_channels=config.CAM_FEAT_DIM,
            bev_size=(config.GRID_SIZE_Y, config.GRID_SIZE_X)
        )
        self.lidar_backbone = LiDARPillarNet(
            in_channels=config.LIDAR_CHANNELS,
            out_channels=config.LIDAR_FEAT_DIM,
            bev_size=(config.GRID_SIZE_Y, config.GRID_SIZE_X),
            pc_range=config.POINT_CLOUD_RANGE
        )
        self.radar_backbone = RadarPointNet(
            in_channels=config.RADAR_CHANNELS,
            out_channels=config.RADAR_FEAT_DIM,
            bev_size=(config.GRID_SIZE_Y, config.GRID_SIZE_X),
            pc_range=config.POINT_CLOUD_RANGE
        )
        
        # 2. Fusion
        self.conv_fuser = ConvFuser(
            cam_channels=config.CAM_FEAT_DIM,
            lidar_channels=config.LIDAR_FEAT_DIM,
            radar_channels=config.RADAR_FEAT_DIM,
            out_channels=config.BEV_FEAT_DIM
        )
        
        # 3. Temporal
        self.ego_align = EgoSpatialAlignment(
            bev_size=(config.GRID_SIZE_Y, config.GRID_SIZE_X),
            voxel_range=[config.POINT_CLOUD_RANGE[0], config.POINT_CLOUD_RANGE[1],
                         config.POINT_CLOUD_RANGE[3], config.POINT_CLOUD_RANGE[4]]
        )
        self.temporal_queue = TemporalFusionQueue(channels=config.BEV_FEAT_DIM)
        
        # 4. Heads
        self.head = DualOccFlowHead(
            in_channels=config.BEV_FEAT_DIM,
            num_classes=config.NUM_CLASSES,
            z_dim=config.GRID_SIZE_Z
        )

    def extract_bev(self, imgs, lidar_pts=None, radar_pts=None, mode='tri_modal'):
        """Trích xuất và hợp nhất đặc trưng BEV từ các cảm biến"""
        # Camera branch
        cam_feat = self.camera_backbone(imgs) # [B, 6, C, H/16, W/16]
        cam_bev = self.cam_bev_pool(cam_feat) # [B, C, H_bev, W_bev]

        # LiDAR branch
        lidar_bev = None
        if lidar_pts is not None and mode in ['cam_lidar', 'tri_modal']:
            lidar_bev = self.lidar_backbone(lidar_pts)

        # Radar branch
        radar_bev = None
        if radar_pts is not None and mode in ['cam_radar', 'tri_modal']:
            radar_bev = self.radar_backbone(radar_pts)

        # Hợp nhất đa cảm biến
        fused_bev = self.conv_fuser(cam_bev, lidar_bev, radar_bev, mode=mode)
        return fused_bev

    def forward(self, batch_data, prev_bev=None, delta_transform=None, mode='tri_modal'):
        """
        batch_data chứa:
          - 'imgs': [B, 6, 3, H, W]
          - 'lidar_pts': list of [N, 4]
          - 'radar_pts': list of [N, 5]
        """
        imgs = batch_data['imgs']
        lidar_pts = batch_data.get('lidar_pts', None)
        radar_pts = batch_data.get('radar_pts', None)

        # 1. Trích xuất BEV hiện tại
        curr_bev = self.extract_bev(imgs, lidar_pts, radar_pts, mode=mode)

        # 2. Hợp nhất chuỗi thời gian (Temporal Fusion)
        if prev_bev is not None and delta_transform is not None:
            prev_bev_aligned = self.ego_align(prev_bev, delta_transform)
            fused_4d_bev = self.temporal_queue(curr_bev, prev_bev_aligned)
        else:
            fused_4d_bev = curr_bev

        # 3. Dự đoán 3D Occupancy & Flow
        occ_logits, flow_vectors = self.head(fused_4d_bev)
        
        return {
            'occ_logits': occ_logits,       # [B, 18, 16, 200, 200]
            'flow_vectors': flow_vectors,   # [B, 3, 16, 200, 200]
            'bev_feat': curr_bev            # Lưu lại cho frame tiếp theo
        }
