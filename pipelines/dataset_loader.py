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
# Tối ưu hóa CPU: Không cho OpenCV tranh chấp luồng với PyTorch DataLoader
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from configs.base_config import Config
from pipelines.coordinate_transforms import compute_ego_relative_transform
from pipelines.anonymizer import PrivacyAnonymizer

class NuScenesOccupancyDataset(Dataset):
    def __init__(self, data_root="data/nuscenes", info_path="data/cache/nuscenes_infos_val.pkl", 
                 occ_gt_root="data/occ3d_cam4d", is_synthetic=False, synthetic_len=20,
                 enable_anonymize=False):
        self.data_root = data_root
        self.occ_gt_root = occ_gt_root
        self.is_synthetic = is_synthetic
        self.synthetic_len = synthetic_len
        self.enable_anonymize = enable_anonymize
        self.anonymizer = PrivacyAnonymizer() if enable_anonymize else None
        
        if not is_synthetic and os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                self.infos = pickle.load(f)
            print(f"[Dataset] Đã nạp {len(self.infos)} frames từ {info_path}")
        else:
            self.is_synthetic = True
            self.infos = [{} for _ in range(synthetic_len)]
            print(f"[Dataset] Đang chạy chế độ Synthetic Mock Data ({synthetic_len} frames)")

        # Lập chỉ mục tự động cho toàn bộ nhãn Occ3D Ground Truth
        self.occ_path_map = {}
        search_dirs = [
            self.occ_gt_root,
            os.path.join(self.occ_gt_root, 'gts'),
            os.path.join(self.data_root, 'gts'),
            self.data_root,
            '/kaggle/input'
        ]
        for sdir in search_dirs:
            if os.path.exists(sdir):
                for root, _, files in os.walk(sdir):
                    if 'labels.npz' in files:
                        frame_tok = os.path.basename(root)
                        self.occ_path_map[frame_tok] = os.path.join(root, 'labels.npz')
        if len(self.occ_path_map) > 0:
            print(f"[Dataset] Đã lập chỉ mục {len(self.occ_path_map)} file nhãn Occ3D Ground Truth chuẩn.")
        elif not self.is_synthetic:
            print("[Dataset] Chú ý: Chưa thấy file labels.npz nào trong các thư mục tìm kiếm.")

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
        gt_occ = np.full((Config.GRID_SIZE_Z, Config.GRID_SIZE_Y, Config.GRID_SIZE_X), Config.FREE_LABEL, dtype=np.int64)
        # Giả lập vài xe cộ và mặt đường
        gt_occ[0:2, :, :] = 11 # mặt đường (driveable_surface)
        gt_occ[1:4, 90:110, 120:140] = 4 # xe hơi phía trước (car)

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
            'token': f"synthetic_frame_{idx}",
            'scene_token': "scene_synthetic_0"
        }

    def __getitem__(self, idx):
        if self.is_synthetic:
            return self._get_synthetic_sample(idx)

        info = self.infos[idx]
        
        # 1. Đọc và tiền xử lý 6 camera
        imgs = []
        for cam_name in Config.CAM_NAMES:
            rel_path = info.get('cams', {}).get(cam_name, {}).get('data_path', '')
            img = None
            if rel_path:
                candidate_paths = [
                    os.path.join(self.data_root, rel_path),
                    os.path.join(self.data_root, 'v1.0-mini', rel_path),
                    os.path.join(self.data_root, 'mini-nuscenes', rel_path),
                    os.path.join(os.path.dirname(self.data_root), rel_path),
                    rel_path
                ]
                cam_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
                if cam_path is not None:
                    raw_img = cv2.imread(cam_path)
                    if raw_img is not None and raw_img.size > 0:
                        raw_img = cv2.resize(raw_img, (Config.IMG_W, Config.IMG_H))
                        if self.enable_anonymize and self.anonymizer is not None:
                            raw_img = self.anonymizer.anonymize_image(raw_img)
                        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        img = np.transpose(raw_img, (2, 0, 1)) # [3, H, W]

            if img is None:
                img = np.zeros((3, Config.IMG_H, Config.IMG_W), dtype=np.float32)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0) # [6, 3, H, W]

        # 2. Đọc LiDAR
        lidar_rel = info.get('lidar', {}).get('data_path', '')
        lidar_pts = None
        if lidar_rel:
            candidate_lidar = [
                os.path.join(self.data_root, lidar_rel),
                os.path.join(self.data_root, 'v1.0-mini', lidar_rel),
                os.path.join(self.data_root, 'mini-nuscenes', lidar_rel),
                os.path.join(os.path.dirname(self.data_root), lidar_rel),
                lidar_rel
            ]
            lidar_path = next((p for p in candidate_lidar if os.path.isfile(p)), None)
            if lidar_path is not None:
                try:
                    pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :4]
                    if len(pts) > 0:
                        lidar_pts = pts
                except Exception:
                    pass
        if lidar_pts is None:
            lidar_pts = np.zeros((100, 4), dtype=np.float32)

        # 3. Đọc Radar
        radar_rel = info.get('radar', {}).get('data_path', '')
        radar_pts = None
        if radar_rel:
            candidate_radar = [
                os.path.join(self.data_root, radar_rel),
                os.path.join(self.data_root, 'v1.0-mini', radar_rel),
                os.path.join(self.data_root, 'mini-nuscenes', radar_rel),
                os.path.join(os.path.dirname(self.data_root), radar_rel),
                radar_rel
            ]
            radar_path = next((p for p in candidate_radar if os.path.isfile(p)), None)
            if radar_path is not None:
                try:
                    from nuscenes.utils.data_classes import RadarPointCloud
                    rpc = RadarPointCloud.from_file(radar_path)
                    radar_pts = rpc.points[:5, :].T # [N, 5]
                except Exception:
                    pass
        if radar_pts is None:
            radar_pts = np.zeros((50, 5), dtype=np.float32)

        # 4. Đọc Ground Truth Occ3D / Cam4D
        token = info['token']
        scene_token = info.get('scene_token', 'scene_0')
        scene_name = info.get('scene_name', '')

        # Tìm từ chỉ mục đã quét trước
        gt_path = self.occ_path_map.get(token, None)
        if gt_path is None:
            gt_candidates = [
                os.path.join(self.occ_gt_root, 'gts', scene_name, token, 'labels.npz'),
                os.path.join(self.occ_gt_root, 'gts', scene_token, token, 'labels.npz'),
                os.path.join(self.occ_gt_root, scene_name, token, 'labels.npz'),
                os.path.join(self.occ_gt_root, scene_token, token, 'labels.npz'),
                os.path.join(self.data_root, 'gts', scene_name, token, 'labels.npz'),
                os.path.join(self.data_root, 'gts', scene_token, token, 'labels.npz'),
                os.path.join(self.occ_gt_root, 'mini_gt', scene_name, token, 'labels.npz'),
                os.path.join(self.occ_gt_root, 'mini_gt', scene_token, token, 'labels.npz')
            ]
            gt_path = next((p for p in gt_candidates if os.path.exists(p)), None)

        if gt_path is not None:
            data = np.load(gt_path)
            gt_occ = data['semantics'] # [200, 200, 16] trong Occ3D chuẩn
            if gt_occ.shape[0] == 200:
                gt_occ = np.transpose(gt_occ, (2, 1, 0)) # Chuyển [X, Y, Z] về [Z, Y, X] -> [16, 200, 200]
            if 'flow' in data.files:
                gt_flow = data['flow']
            else:
                gt_flow = np.zeros((3, Config.GRID_SIZE_Z, Config.GRID_SIZE_Y, Config.GRID_SIZE_X), dtype=np.float32)
        else:
            # Fallback nếu không có file nhãn: toàn bộ không gian là free space (17)
            gt_occ = np.full((Config.GRID_SIZE_Z, Config.GRID_SIZE_Y, Config.GRID_SIZE_X), Config.FREE_LABEL, dtype=np.int64)
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
            'token': token,
            'scene_token': scene_token
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
    scene_tokens = [item.get('scene_token', '') for item in batch]

    return {
        'imgs': imgs,
        'lidar_pts': lidar_pts,
        'radar_pts': radar_pts,
        'gt_occ': gt_occ,
        'gt_flow': gt_flow,
        'delta_transform': delta_transform,
        'tokens': tokens,
        'scene_tokens': scene_tokens
    }
