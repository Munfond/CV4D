"""
Multi-Modal Dataset Loader cho nuScenes
Hỗ trợ nạp 6 camera, LiDAR, Radar và nhãn Voxel Occupancy / Flow
Có sẵn chế độ giả lập (synthetic fallback) để kiểm thử ngay lập tức không cần tải 4GB data.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

from configs.base_config import Config
from pipelines.coordinate_transforms import compute_ego_relative_transform
from pipelines.anonymizer import PrivacyAnonymizer

class NuScenesOccupancyDataset(Dataset):
    def __init__(self, data_root="data/nuscenes", info_path="data/cache/nuscenes_infos_val.pkl", 
                 occ_gt_root="data/occ3d_cam4d", is_synthetic=False, synthetic_len=20):
        self.data_root = data_root
        self.occ_gt_root = occ_gt_root
        self.is_synthetic = is_synthetic
        self.synthetic_len = synthetic_len
        self.anonymizer = PrivacyAnonymizer()
        
        if not is_synthetic and os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                self.infos = pickle.load(f)
            print(f"[Dataset] Đã nạp {len(self.infos)} frames từ {info_path}")
        else:
            self.is_synthetic = True
            self.infos = [{} for _ in range(synthetic_len)]
            print(f"[Dataset] Đang chạy chế độ Synthetic Mock Data ({synthetic_len} frames)")

    def __len__(self):
        return len(self.infos)

    def _get_synthetic_sample(self, idx):
        """Tạo dữ liệu mẫu giả lập chuẩn kích thước để test pipeline"""
        # 6 Camera: [6, 3, 450, 800]
        imgs = np.random.randint(0, 255, (Config.NUM_CAMERAS, 3, Config.IMG_H, Config.IMG_W), dtype=np.uint8)
        imgs = imgs.astype(np.float32) / 255.0

        # LiDAR: 20000 điểm [x, y, z, intensity]
        lidar_pts = np.random.uniform(-35, 35, (20000, 4)).astype(np.float32)
        lidar_pts[:, 2] = np.random.uniform(-0.5, 3.0, 20000)

        # Radar: 500 điểm [x, y, z, rcs, vx_comp]
        radar_pts = np.random.uniform(-35, 35, (500, 5)).astype(np.float32)
        radar_pts[:, 2] = np.random.uniform(-0.5, 1.5, 500)
        radar_pts[:, 4] = np.random.uniform(-10.0, 10.0, 500) # vận tốc m/s

        # Ground Truth 3D Voxel: [16, 200, 200]
        gt_occ = np.zeros((Config.GRID_SIZE_Z, Config.GRID_SIZE_Y, Config.GRID_SIZE_X), dtype=np.int64)
        # Giả lập vài xe cộ và mặt đường
        gt_occ[0:2, :, :] = 11 # mặt đường
        gt_occ[1:4, 90:110, 120:140] = 4 # xe hơi phía trước

        # Ground Truth Flow: [3, 16, 200, 200]
        gt_flow = np.zeros((3, Config.GRID_SIZE_Z, Config.GRID_SIZE_Y, Config.GRID_SIZE_X), dtype=np.float32)
        gt_flow[0, 1:4, 90:110, 120:140] = 2.5 # vx = 2.5 m/s

        # Ma trận biến đổi tương đối giả lập
        delta_transform = torch.eye(4)
        delta_transform[0, 3] = 1.0 # xe tiến 1m

        return {
            'imgs': torch.tensor(imgs, dtype=torch.float32),
            'lidar_pts': torch.tensor(lidar_pts, dtype=torch.float32),
            'radar_pts': torch.tensor(radar_pts, dtype=torch.float32),
            'gt_occ': torch.tensor(gt_occ, dtype=torch.long),
            'gt_flow': torch.tensor(gt_flow, dtype=torch.float32),
            'delta_transform': delta_transform,
            'token': f"synthetic_frame_{idx}"
        }

    def __getitem__(self, idx):
        if self.is_synthetic:
            return self._get_synthetic_sample(idx)

        info = self.infos[idx]
        
        # 1. Đọc và tiền xử lý 6 camera
        imgs = []
        for cam_name in Config.CAM_NAMES:
            cam_path = os.path.join(self.data_root, info['cams'][cam_name]['data_path'])
            if os.path.exists(cam_path):
                img = cv2.imread(cam_path)
                # Tự động làm mờ mặt và biển số
                img = self.anonymizer.anonymize_image(img)
                img = cv2.resize(img, (Config.IMG_W, Config.IMG_H))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1)) # [3, H, W]
            else:
                img = np.zeros((3, Config.IMG_H, Config.IMG_W), dtype=np.float32)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0) # [6, 3, H, W]

        # 2. Đọc LiDAR
        lidar_path = os.path.join(self.data_root, info['lidar']['data_path'])
        if os.path.exists(lidar_path):
            lidar_pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
        else:
            lidar_pts = np.zeros((100, 4), dtype=np.float32)

        # 3. Đọc Radar
        radar_path = os.path.join(self.data_root, info['radar']['data_path'])
        if os.path.exists(radar_path):
            # Với định dạng pcd đơn giản hoặc bin
            try:
                from nuscenes.utils.data_classes import RadarPointCloud
                rpc = RadarPointCloud.from_file(radar_path)
                radar_pts = rpc.points[:5, :].T # [N, 5]
            except:
                radar_pts = np.zeros((50, 5), dtype=np.float32)
        else:
            radar_pts = np.zeros((50, 5), dtype=np.float32)

        # 4. Đọc Ground Truth Occ3D / Cam4D
        token = info['token']
        scene_token = info.get('scene_token', 'scene_0')
        gt_path = os.path.join(self.occ_gt_root, 'mini_gt', scene_token, token, 'labels.npz')
        if os.path.exists(gt_path):
            data = np.load(gt_path)
            gt_occ = data['semantics'] # [16, 200, 200] hoặc [200, 200, 16]
            if gt_occ.shape[0] == 200:
                gt_occ = np.transpose(gt_occ, (2, 1, 0)) # về [Z, Y, X]
            gt_flow = data.get('flow', np.zeros((3, Config.GRID_SIZE_Z, Config.GRID_SIZE_Y, Config.GRID_SIZE_X), dtype=np.float32))
        else:
            gt_occ = np.zeros((Config.GRID_SIZE_Z, Config.GRID_SIZE_Y, Config.GRID_SIZE_X), dtype=np.int64)
            gt_flow = np.zeros((3, Config.GRID_SIZE_Z, Config.GRID_SIZE_Y, Config.GRID_SIZE_X), dtype=np.float32)

        # 5. Tính delta transform nếu có frame trước
        if idx > 0 and self.infos[idx-1].get('scene_token') == info.get('scene_token'):
            delta_transform = compute_ego_relative_transform(self.infos[idx-1], info)
        else:
            delta_transform = torch.eye(4)

        return {
            'imgs': torch.tensor(imgs, dtype=torch.float32),
            'lidar_pts': torch.tensor(lidar_pts, dtype=torch.float32),
            'radar_pts': torch.tensor(radar_pts, dtype=torch.float32),
            'gt_occ': torch.tensor(gt_occ, dtype=torch.long),
            'gt_flow': torch.tensor(gt_flow, dtype=torch.float32),
            'delta_transform': delta_transform,
            'token': token
        }

def collate_fn_4docc(batch):
    """Custom Collate ghép danh sách mảng LiDAR/Radar không cố định số điểm"""
    imgs = torch.stack([item['imgs'] for item in batch], dim=0)
    lidar_pts = [item['lidar_pts'] for item in batch]
    radar_pts = [item['radar_pts'] for item in batch]
    gt_occ = torch.stack([item['gt_occ'] for item in batch], dim=0)
    gt_flow = torch.stack([item['gt_flow'] for item in batch], dim=0)
    delta_transform = torch.stack([item['delta_transform'] for item in batch], dim=0)
    tokens = [item['token'] for item in batch]

    return {
        'imgs': imgs,
        'lidar_pts': lidar_pts,
        'radar_pts': radar_pts,
        'gt_occ': gt_occ,
        'gt_flow': gt_flow,
        'delta_transform': delta_transform,
        'tokens': tokens
    }
