"""
Config chung cho bài toán BEV/4D Occupancy & Flow Lab (VinFast ADAS)
Tối ưu hóa chạy trên Kaggle GPU (T4 16GB)
"""
import torch

class Config:
    # 1. Định nghĩa không gian Voxel 3D quanh xe (Ego-Vehicle)
    # X: [-40m, 40m], Y: [-40m, 40m], Z: [-1.0m, 5.4m]
    POINT_CLOUD_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    VOXEL_SIZE = [0.4, 0.4, 0.4]  # 0.4m x 0.4m x 0.4m
    
    GRID_SIZE_X = int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0])  # 200
    GRID_SIZE_Y = int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1])  # 200
    GRID_SIZE_Z = int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) / VOXEL_SIZE[2])  # 16
    GRID_SIZE = [GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z]  # [200, 200, 16]

    # 2. Cảm biến đầu vào
    NUM_CAMERAS = 6
    CAM_NAMES = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    IMG_H, IMG_W = 450, 800  # Resize từ 900x1600 để tối ưu VRAM cho Kaggle T4
    LIDAR_CHANNELS = 4  # [x, y, z, intensity]
    RADAR_CHANNELS = 5  # [x, y, z, rcs, vx_comp]

    # 3. Kênh đặc trưng mạng (Channels)
    CAM_FEAT_DIM = 256
    LIDAR_FEAT_DIM = 256
    RADAR_FEAT_DIM = 64
    BEV_FEAT_DIM = 256

    # 4. Nhãn ngữ nghĩa 3D Occupancy chuẩn Occ3D-nuScenes (17 classes nuScenes-lidarseg + Class 17 Free Space)
    NUM_CLASSES = 18
    FREE_LABEL = 17
    CLASS_NAMES = [
        'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation', 'free'
    ]

    # Bảng màu hiển thị (RGB)
    COLOR_MAP = {
        0: [105, 105, 105],    # others / void
        1: [255, 120, 50],     # barrier
        2: [255, 192, 203],    # bicycle
        3: [255, 255, 0],      # bus
        4: [0, 150, 255],      # car
        5: [160, 32, 240],     # construction_vehicle
        6: [255, 69, 0],       # motorcycle
        7: [255, 0, 0],        # pedestrian
        8: [255, 140, 0],      # traffic_cone
        9: [218, 112, 214],    # trailer
        10: [75, 0, 130],      # truck
        11: [128, 128, 128],   # driveable_surface
        12: [176, 196, 222],   # other_flat
        13: [0, 250, 154],     # sidewalk
        14: [34, 139, 34],     # terrain
        15: [139, 69, 19],     # manmade
        16: [0, 100, 0],       # vegetation
        17: [0, 0, 0]          # free space (không khí)
    }

    # 5. Tham số Huấn luyện Kaggle
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 4  # Tương đương batch size = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    USE_AMP = True        # Tự động dùng FP16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Thời gian giữa 2 keyframes nuScenes (2Hz -> dt = 0.5s)
    DELTA_T = 0.5
