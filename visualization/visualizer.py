"""
Trực quan hóa Bản đồ 3D BEV & 4D Occupancy Flow
Hỗ trợ 2 chế độ:
1. Matplotlib BEV Snapshot (Lưu file .png giống hệt màn hình thực tế xe tự hành)
2. Rerun.io 3D Interactive (Tương tác 3D xoay 360 độ)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from configs.base_config import Config

class OccupancyVisualizer:
    def __init__(self, use_rerun=False, session_name="VinFast_4DOcc_Lab"):
        self.use_rerun = use_rerun
        if use_rerun:
            try:
                import rerun as rr
                rr.init(session_name, spawn=True)
                self.rr = rr
                print("[Visualizer] Đã kích hoạt Rerun.io 3D Viewer")
            except ImportError:
                print("[Visualizer] Chưa cài rerun-sdk, chuyển sang chế độ Matplotlib.")
                self.use_rerun = False

    def plot_bev_snapshot(self, occ_pred, occ_gt=None, flow_pred=None, lidar_pts=None, radar_pts=None, risk_mask=None, save_path="bev_output.png"):
        """
        Vẽ bản đồ BEV góc nhìn từ trên cao tương tự giao diện kiểm thử chuyên nghiệp:
        - Tâm xe: Tam giác màu xanh lơ
        - Điểm LiDAR: Màu vàng/xám
        - Điểm Radar: Màu hồng/tím kèm vector vận tốc
        - Dự đoán Occupancy: Cam (an toàn) / Vàng (chú ý) / Đỏ (nguy cơ va chạm cao)
        - Ground Truth: Viền màu Xanh lá
        """
        if torch.is_tensor(occ_pred):
            occ_pred = occ_pred.detach().cpu().numpy()
        if torch.is_tensor(occ_gt):
            occ_gt = occ_gt.detach().cpu().numpy()
        if torch.is_tensor(flow_pred):
            flow_pred = flow_pred.detach().cpu().numpy()
        if torch.is_tensor(risk_mask):
            risk_mask = risk_mask.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
        ax.set_facecolor('black')

        # 1. Vẽ các vòng tròn khoảng cách quanh xe (10m, 20m, 30m, 40m)
        for r in [10, 20, 30, 40]:
            circle = plt.Circle((0, 0), r, color='#333333', fill=False, linestyle='--', linewidth=0.8)
            ax.add_patch(circle)
            ax.text(0, r + 0.5, f"{r}m", color='#666666', fontsize=8, ha='center')

        # 2. Vẽ điểm LiDAR
        if lidar_pts is not None:
            if torch.is_tensor(lidar_pts):
                lidar_pts = lidar_pts.detach().cpu().numpy()
            if len(lidar_pts) > 0:
                ax.scatter(lidar_pts[:, 1], lidar_pts[:, 0], s=0.5, c='#e6c619', alpha=0.6, label='LiDAR Points')

        # 3. Vẽ điểm Radar kèm vector Doppler
        if radar_pts is not None:
            if torch.is_tensor(radar_pts):
                radar_pts = radar_pts.detach().cpu().numpy()
            if len(radar_pts) > 0:
                ax.scatter(radar_pts[:, 1], radar_pts[:, 0], s=12, c='#ff00ff', marker='D', label='Radar Detections')

        # 4. Chiếu Voxel Occupancy lên mặt phẳng BEV (Gộp trục Z)
        # occ_pred shape: [16, 200, 200] -> gộp thành 2D [200, 200]
        bev_occ_pred = np.max(occ_pred, axis=0) # lấy nhãn lớn nhất theo chiều cao
        occupied_mask = (bev_occ_pred > 0) & (bev_occ_pred != 11) # bỏ qua mặt đường để nhìn rõ vật cản

        y_idx, x_idx = np.where(occupied_mask)
        if len(x_idx) > 0:
            # Đổi chỉ số lưới sang tọa độ mét
            x_m = x_idx * Config.VOXEL_SIZE[0] + Config.POINT_CLOUD_RANGE[0]
            y_m = y_idx * Config.VOXEL_SIZE[1] + Config.POINT_CLOUD_RANGE[1]
            labels = bev_occ_pred[y_idx, x_idx]
            
            # Vẽ các voxel dự đoán (Màu theo rủi ro nếu có)
            if risk_mask is not None:
                bev_risk = np.max(risk_mask, axis=0)
                risk_vals = bev_risk[y_idx, x_idx]
                colors = np.array(['#ff7700'] * len(x_idx), dtype=object)
                colors[risk_vals == 1] = '#ffcc00' # Vàng: Chú ý
                colors[risk_vals == 2] = '#ff0033' # Đỏ: Nguy cơ va chạm cao
                ax.scatter(x_m, y_m, s=7, c=colors, alpha=0.85, label='Predicted Occupancy (Red=Risk)')
            else:
                ax.scatter(x_m, y_m, s=6, c='#ff7700', alpha=0.8, label='Predicted Occupancy')

            # 5. Vẽ vector dòng chảy (Occupancy Flow)
            if flow_pred is not None:
                # Lấy vector [vx, vy] trung bình tại mỗi điểm
                vx = flow_pred[0, :, y_idx, x_idx].mean(axis=0)
                vy = flow_pred[1, :, y_idx, x_idx].mean(axis=0)
                # Chỉ vẽ những vector có vận tốc đáng kể
                speed = np.sqrt(vx**2 + vy**2)
                dynamic = speed > 0.5
                if np.any(dynamic):
                    ax.quiver(x_m[dynamic], y_m[dynamic], vx[dynamic], vy[dynamic], 
                              color='#00ffcc', scale=25, width=0.004, label='Occupancy Flow')

        # 6. Vẽ Ground Truth (Viền xanh lá - Green = GT)
        if occ_gt is not None:
            bev_occ_gt = np.max(occ_gt, axis=0)
            gt_mask = (bev_occ_gt > 0) & (bev_occ_gt != 11)
            y_gt, x_gt = np.where(gt_mask)
            if len(x_gt) > 0:
                x_gt_m = x_gt * Config.VOXEL_SIZE[0] + Config.POINT_CLOUD_RANGE[0]
                y_gt_m = y_gt * Config.VOXEL_SIZE[1] + Config.POINT_CLOUD_RANGE[1]
                ax.scatter(x_gt_m, y_gt_m, s=10, facecolors='none', edgecolors='#00ff00', 
                           linewidths=0.5, alpha=0.7, label='Ground Truth (GT)')

        # 7. Vẽ Xe tự hành Ego tại tâm (Tam giác màu xanh lơ)
        ego_triangle = patches.Polygon([[0, 2.0], [-1.0, -2.0], [1.0, -2.0]], 
                                       closed=True, facecolor='#00ffff', edgecolor='white', label='Ego Vehicle')
        ax.add_patch(ego_triangle)

        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_xlabel('Y (Mét - Trái/Phải)', color='white')
        ax.set_ylabel('X (Mét - Trước/Sau)', color='white')
        ax.tick_params(colors='white')
        ax.set_title('VinFast ADAS: 4D Occupancy & Flow BEV Verification\n(Green = Ground Truth | Orange = Predicted)', color='white', fontsize=12)
        ax.legend(loc='upper right', facecolor='#111111', edgecolor='#444444', labelcolor='white', fontsize=8)
        ax.grid(True, color='#222222', linestyle=':')

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, facecolor='black')
        plt.close()
        print(f"[Visualizer] Đã lưu ảnh BEV Map Snapshot vào: {save_path}")

    def log_rerun(self, frame_idx, occ_pred, flow_pred=None, lidar_pts=None):
        """Bắn dữ liệu sang Rerun.io 3D Viewer nếu bật chế độ tương tác"""
        if not self.use_rerun:
            return
        self.rr.set_time_sequence("frame", frame_idx)
        if lidar_pts is not None:
            self.rr.log("world/lidar", self.rr.Points3D(lidar_pts[:, :3], colors=[220, 200, 30]))
            
        occupied = np.where(occ_pred > 0)
        z, y, x = occupied[0], occupied[1], occupied[2]
        centers = np.stack([
            x * Config.VOXEL_SIZE[0] + Config.POINT_CLOUD_RANGE[0],
            y * Config.VOXEL_SIZE[1] + Config.POINT_CLOUD_RANGE[1],
            z * Config.VOXEL_SIZE[2] + Config.POINT_CLOUD_RANGE[2]
        ], axis=-1)

        self.rr.log("world/voxels", self.rr.Boxes3D(centers=centers, half_sizes=[0.2, 0.2, 0.2], colors=[255, 120, 0]))
