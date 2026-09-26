"""
Thuật toán Cảnh báo Nguy cơ Va chạm Hình học (Collision Risk Assessment)
Tính toán giao cắt hình học giữa Hành lang an toàn xe tự hành (Ego Safety Corridor)
và Vector dòng chảy (Occupancy Flow) của các voxel di động.
"""
import numpy as np
import torch

from configs.base_config import Config

class CollisionRiskAssessment:
    def __init__(self, horizon_sec=1.5, ego_width=2.2, ego_length=4.8, safety_margin=1.0):
        self.horizon_sec = horizon_sec
        self.ego_width = ego_width + safety_margin
        self.ego_length = ego_length + safety_margin

    def evaluate_risk(self, occ_pred, flow_pred, ego_speed_mps=10.0):
        """
        occ_pred: [16, 200, 200] hoặc [Z, Y, X] nhãn ngữ nghĩa
        flow_pred: [3, 16, 200, 200] chứa vector vận tốc [vx, vy, vz]
        ego_speed_mps: Vận tốc hiện tại của xe tự hành (m/s)

        Trả về:
          risk_mask: [16, 200, 200] với:
            0: An toàn (Green)
            1: Chú ý (Yellow - áp sát hành lang)
            2: Nguy cơ va chạm cao (Red - giao cắt trực tiếp trong horizon_sec)
        """
        if torch.is_tensor(occ_pred):
            occ_pred = occ_pred.detach().cpu().numpy()
        if torch.is_tensor(flow_pred):
            flow_pred = flow_pred.detach().cpu().numpy()

        Z, Y, X = occ_pred.shape
        risk_mask = np.zeros((Z, Y, X), dtype=np.uint8)

        # Lọc các voxel có vật cản di động (classes 2->10: ô tô, xe máy, xe buýt, người đi bộ)
        dynamic_voxels = (occ_pred >= 2) & (occ_pred <= 10)
        z_idx, y_idx, x_idx = np.where(dynamic_voxels)

        if len(x_idx) == 0:
            return risk_mask

        # Tọa độ mét hiện tại của các voxel
        x_m = x_idx * Config.VOXEL_SIZE[0] + Config.POINT_CLOUD_RANGE[0] # trục trước/sau (X)
        y_m = y_idx * Config.VOXEL_SIZE[1] + Config.POINT_CLOUD_RANGE[1] # trục trái/phải (Y)
        z_m = z_idx * Config.VOXEL_SIZE[2] + Config.POINT_CLOUD_RANGE[2] # độ cao (Z)

        # Lấy vector vận tốc [vx, vy] của từng voxel
        vx = flow_pred[0, z_idx, y_idx, x_idx]
        vy = flow_pred[1, z_idx, y_idx, x_idx]

        # Vận tốc tương đối so với xe Ego (Ego tiến về phía trước theo trục X với ego_speed_mps)
        v_rel_x = vx - ego_speed_mps
        v_rel_y = vy

        # Dự phóng vị trí tương đối sau horizon_sec
        x_future = x_m + v_rel_x * self.horizon_sec
        y_future = y_m + v_rel_y * self.horizon_sec

        # 1. Kiểm tra nguy cơ va chạm cao (Red): Vị trí tương lai rơi vào thân xe
        in_ego_x = (x_future >= -self.ego_length / 2.0) & (x_future <= self.ego_length / 2.0 + 5.0)
        in_ego_y = np.abs(y_future) <= (self.ego_width / 2.0)
        high_risk = in_ego_x & in_ego_y

        # 2. Kiểm tra chú ý (Yellow): Áp sát hành lang trong phạm vi 2m
        caution_x = (x_future >= -self.ego_length) & (x_future <= self.ego_length + 10.0)
        caution_y = np.abs(y_future) <= (self.ego_width / 2.0 + 2.0)
        caution_risk = caution_x & caution_y & (~high_risk)

        # Gán nhãn cờ rủi ro
        risk_mask[z_idx[caution_risk], y_idx[caution_risk], x_idx[caution_risk]] = 1
        risk_mask[z_idx[high_risk], y_idx[high_risk], x_idx[high_risk]] = 2

        return risk_mask
