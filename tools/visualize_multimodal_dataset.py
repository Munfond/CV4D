"""
VinFast ADAS 4D-OccFusion: Multi-Modal Perception & User-Friendly Visualization Suite
Kết hợp trực quan hóa đa cảm biến dựa trên dữ liệu thật của dataset:
1. Camera Image với 3D Bounding Box hình học chiếu trực tiếp lên ảnh RGB thực tế
2. BEV LiDAR Point Cloud + 3D Bounding Boxes (Green = Ground Truth, Orange/Red = Predicted)
   chuẩn xác theo phong cách kỹ thuật kiểm thử xe tự hành (giống ảnh mẫu người dùng)
3. Không gian 3D Voxel Occupancy & Dynamic Flow Field
4. Video/GIF động chuỗi thời gian đa góc nhìn (Multi-view Sync Animation)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import imageio

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Bảng màu các lớp đối tượng trực quan cho người dùng
CLASS_COLORS = {
    'car': ([0, 180, 255], 'Xe hơi (Car)'),
    'truck': ([255, 140, 0], 'Xe tải (Truck)'),
    'bus': ([255, 215, 0], 'Xe buýt (Bus)'),
    'motorcycle': ([255, 60, 60], 'Xe máy (Motorcycle)'),
    'bicycle': ([50, 205, 50], 'Xe đạp (Bicycle)'),
    'pedestrian': ([255, 50, 150], 'Người đi bộ (Pedestrian)'),
    'barrier': ([180, 180, 180], 'Dải phân cách (Barrier)'),
    'traffic_cone': ([255, 100, 0], 'Cọc tiêu (Traffic Cone)'),
}

def quat_to_matrix(q):
    """Chuyển quaternion [w, x, y, z] sang ma trận quay 3x3"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)

def get_box_corners_3d(translation, size, rotation):
    """
    Sinh 8 đỉnh của hộp 3D trong hệ tọa độ global nuScenes:
    size = [width (x), length (y), height (z)]
    """
    w, l, h = size[0], size[1], size[2]
    # 8 đỉnh trong hệ tọa độ cục bộ của hộp:
    # 0: Front-Right-Bottom, 1: Front-Left-Bottom, 2: Rear-Left-Bottom, 3: Rear-Right-Bottom
    # 4: Front-Right-Top,    5: Front-Left-Top,    6: Rear-Left-Top,    7: Rear-Right-Top
    corners_local = np.array([
        [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2],
        [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2],
        [-h/2, -h/2, -h/2, -h/2, h/2,  h/2,  h/2,  h/2]
    ], dtype=np.float32)

    R_box = quat_to_matrix(rotation)
    corners_global = np.dot(R_box, corners_local) + np.array(translation, dtype=np.float32).reshape(3, 1)
    return corners_global

def project_box_to_camera(corners_global, R_ego, t_ego, R_cam, t_cam, K, img_w=1600, img_h=900):
    """Chiếu 8 đỉnh hộp 3D từ Global sang Camera và tính tọa độ pixel 2D"""
    # 1. Global -> Ego
    corners_ego = np.dot(R_ego.T, corners_global - t_ego.reshape(3, 1))
    # 2. Ego -> Camera
    corners_cam = np.dot(R_cam.T, corners_ego - t_cam.reshape(3, 1))

    # Nếu tất cả các đỉnh nằm phía sau camera (z <= 0.5m), bỏ qua
    if np.any(corners_cam[2, :] <= 0.5):
        return None, corners_cam

    # 3. Phép chiếu nội tại Camera: [u, v, 1]^T ~ K * P_cam
    uv = np.dot(K, corners_cam)
    u = uv[0, :] / uv[2, :]
    v = uv[1, :] / uv[2, :]

    # Kiểm tra xem có ít nhất 1 đỉnh nằm trong khung nhìn ảnh không
    in_view = np.any((u >= -100) & (u <= img_w + 100) & (v >= -100) & (v <= img_h + 100))
    if not in_view:
        return None, corners_cam

    pts_2d = np.stack([u, v], axis=1) # [8, 2]
    return pts_2d, corners_cam

def draw_3d_box_on_image(img, pts_2d, color_bgr=(0, 255, 0), thickness=2, label="Car", dist_m=None):
    """Vẽ khung dây 3D wireframe của hộp lên ảnh camera thực tế kèm nhãn và cự ly"""
    if pts_2d is None:
        return img
    
    pts = pts_2d.astype(np.int32)
    
    # 12 cạnh của hộp 3D:
    # Mặt đáy: (0, 1), (1, 2), (2, 3), (3, 0)
    # Mặt nắp: (4, 5), (5, 6), (6, 7), (7, 4)
    # 4 cột đứng: (0, 4), (1, 5), (2, 6), (3, 7)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for (i1, i2) in edges:
        cv2.line(img, (pts[i1, 0], pts[i1, 1]), (pts[i2, 0], pts[i2, 1]), color_bgr, thickness, cv2.LINE_AA)
        
    # Vẽ dấu chéo chữ 'X' ở mặt trước (Front face: 0, 1, 5, 4) để chỉ rõ đầu xe hướng về đâu
    cv2.line(img, (pts[0, 0], pts[0, 1]), (pts[5, 0], pts[5, 1]), color_bgr, max(1, thickness-1), cv2.LINE_AA)
    cv2.line(img, (pts[1, 0], pts[1, 1]), (pts[4, 0], pts[4, 1]), color_bgr, max(1, thickness-1), cv2.LINE_AA)
    
    # Ghi nhãn phân loại và khoảng cách
    tag_x = int(np.min(pts[:, 0]))
    tag_y = max(20, int(np.min(pts[:, 1])) - 8)
    text = f"{label}"
    if dist_m is not None:
        text += f" {dist_m:.1f}m"
        
    # Nền đen bán trong suốt cho chữ dễ đọc
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(img, (tag_x, tag_y - th - 4), (tag_x + tw + 4, tag_y + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (tag_x + 2, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)
    return img

class NuScenesMultiModalVisualizer:
    def __init__(self, data_root="data/nuscenes", model_output_dir="model_output"):
        self.data_root = data_root
        self.output_dir = model_output_dir
        self.json_dir = os.path.join(data_root, 'v1.0-mini')
        
        print(f"[Visualizer] Khởi tạo dữ liệu từ: {self.data_root}")
        
        # Đọc các bảng quan hệ nuScenes
        with open(os.path.join(self.json_dir, 'sample.json'), 'r') as f:
            self.samples = {s['token']: s for s in json.load(f)}
        with open(os.path.join(self.json_dir, 'sample_data.json'), 'r') as f:
            self.sample_data = json.load(f)
        with open(os.path.join(self.json_dir, 'calibrated_sensor.json'), 'r') as f:
            self.calibs = {c['token']: c for c in json.load(f)}
        with open(os.path.join(self.json_dir, 'ego_pose.json'), 'r') as f:
            self.egos = {e['token']: e for e in json.load(f)}
        with open(os.path.join(self.json_dir, 'sample_annotation.json'), 'r') as f:
            self.annotations = json.load(f)
        with open(os.path.join(self.json_dir, 'instance.json'), 'r') as f:
            self.instances = {i['token']: i for i in json.load(f)}
        with open(os.path.join(self.json_dir, 'category.json'), 'r') as f:
            self.categories = {c['token']: c['name'] for c in json.load(f)}
            
        # Đọc dữ liệu mô hình dự đoán từ model_output
        bbox_path = os.path.join(self.output_dir, 'instance', 'bounding_box.json')
        self.pred_bboxes = {}
        if os.path.exists(bbox_path):
            with open(bbox_path, 'r') as f:
                self.pred_bboxes = json.load(f)
                
        # Gom nhóm annotations theo sample_token
        self.sample_to_anns = {}
        for a in self.annotations:
            stok = a['sample_token']
            if stok not in self.sample_to_anns:
                self.sample_to_anns[stok] = []
            self.sample_to_anns[stok].append(a)
            
    def get_sample_sensor_data(self, sample_token, channel):
        """Lấy bản ghi sample_data cho một sensor cụ thể"""
        for sd in self.sample_data:
            if sd['sample_token'] == sample_token and f"{channel}/" in sd['filename'] and sd.get('is_key_frame', True):
                return sd
        return None

    def render_bev_lidar_gt_and_pred(self, sample_token, save_path=None, x_range=(-50, 50), y_range=(-50, 50)):
        """
        Vẽ bản đồ BEV LiDAR Point Cloud kết hợp 3D Bounding Boxes
        Tái hiện chính xác phong cách đồ thị kiểm thử của xe tự hành (như ảnh tham khảo):
        - Nền đen sang trọng (#000000)
        - Trục đứng: x forward [m] từ -50 đến 50
        - Trục ngang: y left [m] từ 50 đến -50
        - Quét laser đồng tâm (concentric rings) của điểm LiDAR
        - Green = Ground Truth 3D Bounding Boxes (green = GT)
        - Orange/Red/Magenta = Model Predictions kèm mũi tên vận tốc
        - Xe chủ EGO: Tam giác màu xanh lơ tại gốc (0, 0)
        """
        fig, ax = plt.subplots(figsize=(11, 11), facecolor='#000000')
        ax.set_facecolor('#000000')

        # 1. Nạp điểm LiDAR từ file .pcd.bin
        lidar_sd = self.get_sample_sensor_data(sample_token, 'LIDAR_TOP')
        if lidar_sd:
            lid_path = os.path.join(self.data_root, lidar_sd['filename'])
            pts = np.fromfile(lid_path, dtype=np.float32).reshape(-1, 5)
            
            calib = self.calibs[lidar_sd['calibrated_sensor_token']]
            R_lid = quat_to_matrix(calib['rotation'])
            t_lid = np.array(calib['translation'])
            
            # Biến đổi điểm LiDAR sang hệ tọa độ Ego
            pts_ego = np.dot(R_lid, pts[:, :3].T) + t_lid.reshape(3, 1) # [3, N]
            
            # Quy đổi sang hệ trục hiển thị ISO:
            # x_fwd = pts_ego[1] (trục dọc hướng tới trước)
            # y_left = -pts_ego[0] (trục ngang hướng sang trái)
            x_fwd = pts_ego[1]
            y_left = -pts_ego[0]
            z_height = pts_ego[2]

            # Phân tách điểm mặt đường (tạo các vòng tròn đồng tâm tự nhiên) và điểm vật thể
            mask_range = (x_fwd >= x_range[0]) & (x_fwd <= x_range[1]) & (y_left >= y_range[0]) & (y_left <= y_range[1])
            x_fwd, y_left, z_height = x_fwd[mask_range], y_left[mask_range], z_height[mask_range]

            ground_mask = (z_height < -1.1)
            obstacle_mask = (z_height >= -1.1)

            # Điểm quét mặt đường (concentric rings màu xám mảnh)
            ax.scatter(y_left[ground_mask], x_fwd[ground_mask], s=0.3, c='#555555', alpha=0.5, label='Ground Scan Rings')
            # Điểm chướng ngại vật (màu vàng rực rỡ như ảnh mẫu)
            ax.scatter(y_left[obstacle_mask], x_fwd[obstacle_mask], s=1.2, c='#ffff33', alpha=0.85, label='Obstacle Points')

        # 2. Lấy thông tin Ego Pose của frame
        ego_sd = lidar_sd or self.get_sample_sensor_data(sample_token, 'CAM_FRONT')
        ego_pose = self.egos[ego_sd['ego_pose_token']]
        R_ego = quat_to_matrix(ego_pose['rotation'])
        t_ego = np.array(ego_pose['translation'])

        # 3. Vẽ các hộp Ground Truth (MÀU XANH LÁ - GREEN = GT)
        anns = self.sample_to_anns.get(sample_token, [])
        gt_count = 0
        for a in anns:
            c_glob = get_box_corners_3d(a['translation'], a['size'], a['rotation'])
            c_ego = np.dot(R_ego.T, c_glob - t_ego.reshape(3, 1))
            
            # 4 góc mặt đáy của hộp trong hệ ISO (y_left = -c_ego[0], x_fwd = c_ego[1])
            bx_fwd = c_ego[1, :4]
            by_left = -c_ego[0, :4]
            
            # Lọc trong tầm quan sát
            if np.mean(bx_fwd) < x_range[0] or np.mean(bx_fwd) > x_range[1] or np.mean(by_left) < y_range[0] or np.mean(by_left) > y_range[1]:
                continue
                
            # Vẽ đa giác hộp 2D
            rect_x = [by_left[0], by_left[1], by_left[2], by_left[3], by_left[0]]
            rect_y = [bx_fwd[0], bx_fwd[1], bx_fwd[2], bx_fwd[3], bx_fwd[0]]
            ax.plot(rect_x, rect_y, color='#00ff44', linewidth=1.6, label='Ground Truth (GT)' if gt_count == 0 else "")
            
            # Vẽ đường chỉ hướng mũi xe GT (Front face: cạnh 0-1)
            front_mid_x = (by_left[0] + by_left[1]) / 2.0
            front_mid_y = (bx_fwd[0] + bx_fwd[1]) / 2.0
            center_x = np.mean(by_left)
            center_y = np.mean(bx_fwd)
            ax.plot([center_x, front_mid_x], [center_y, front_mid_y], color='#00ff44', linewidth=1.2)
            gt_count += 1

        # 4. Vẽ các hộp Mô hình Dự đoán (MÀU CAM / ĐỎ / HỒNG KÈM VECTOR MŨI TÊN VẬN TỐC)
        pred_boxes = self.pred_bboxes.get(sample_token, [])
        pred_count = 0
        for pbox in pred_boxes:
            c = pbox['centroid']
            dim = pbox['dimensions_lwh']
            vel = pbox.get('velocity', [0, 0, 0])
            
            cx_ego, cy_ego = c[0], c[1]
            bx_fwd_c = cy_ego
            by_left_c = -cx_ego
            
            if bx_fwd_c < x_range[0] or bx_fwd_c > x_range[1] or by_left_c < y_range[0] or by_left_c > y_range[1]:
                continue
                
            l, w_dim = dim[0], dim[1]
            # Hộp chữ nhật BEV
            hw, hl = w_dim / 2.0, l / 2.0
            poly_x = [by_left_c - hw, by_left_c + hw, by_left_c + hw, by_left_c - hw, by_left_c - hw]
            poly_y = [bx_fwd_c - hl, bx_fwd_c - hl, bx_fwd_c + hl, bx_fwd_c + hl, bx_fwd_c - hl]
            
            # Đổi màu cam/đỏ theo vận tốc
            speed_val = np.linalg.norm(vel)
            p_color = '#ff8800' if speed_val < 1.0 else '#ff0055'
            ax.plot(poly_x, poly_y, color=p_color, linewidth=1.8, label='Predicted (Model)' if pred_count == 0 else "")
            
            # Vẽ vector vận tốc / hướng chuyển động (mũi tên có đầu tròn như ảnh mẫu)
            vx_fwd = vel[1]
            vy_left = -vel[0]
            arrow_scale = 3.5
            end_x = by_left_c + vy_left * arrow_scale
            end_y = bx_fwd_c + vx_fwd * arrow_scale
            ax.plot([by_left_c, end_x], [bx_fwd_c, end_y], color=p_color, linewidth=2.0)
            ax.scatter([end_x], [end_y], color=p_color, s=25, zorder=5)
            pred_count += 1

        # 5. Vẽ Xe chủ EGO tại gốc tọa độ (0, 0)
        # Tam giác màu xanh lơ chỉ hướng đi lên (x forward)
        ego_triangle_y = [0, 2.0, -1.0, 0] # y_left
        ego_triangle_x = [2.2, -1.8, -1.8, 2.2] # x_fwd
        ax.fill(ego_triangle_y, ego_triangle_x, color='#00ffff', alpha=0.9, zorder=10, label='Ego Vehicle')
        ax.plot(ego_triangle_y, ego_triangle_x, color='#ffffff', linewidth=1.2, zorder=11)

        # 6. Các vòng tròn cự ly radar/lidar (10m, 20m, 30m, 40m)
        for r in [10, 20, 30, 40, 50]:
            circle = plt.Circle((0, 0), r, color='#333333', fill=False, linestyle=':', linewidth=0.7)
            ax.add_patch(circle)

        # Cấu hình trục giống hệt ảnh mẫu
        ax.set_xlim([y_range[1], y_range[0]]) # Invert trục X để y left dương ở bên trái
        ax.set_ylim([x_range[0], x_range[1]])
        ax.set_xlabel('y left [m]', color='#ffffff', fontsize=12, fontweight='bold', labelpad=8)
        ax.set_ylabel('x forward [m]', color='#ffffff', fontsize=12, fontweight='bold', labelpad=8)
        
        # Grid và ticks
        ax.set_xticks(np.arange(y_range[0], y_range[1] + 1, 10))
        ax.set_yticks(np.arange(x_range[0], x_range[1] + 1, 10))
        ax.tick_params(colors='#ffffff', labelsize=10)
        ax.grid(color='#222222', linestyle='-', linewidth=0.5)

        title_str = f"VINFAST ADAS 4D-OccFusion Perception on nuScenes val\ngreen = GT | orange/red = Predicted (with motion vector)"
        ax.set_title(title_str, color='#ffffff', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper right', facecolor='#111111', edgecolor='#444444', labelcolor='#ffffff', fontsize=9)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200, facecolor='#000000', edgecolor='none')
            print(f"[BEV LiDAR] Đã lưu bản đồ BEV chuẩn ảnh mẫu tại: {save_path}")
        return fig

    def render_camera_with_3d_boxes(self, sample_token, channel='CAM_FRONT', save_path=None):
        """
        Nạp ảnh camera thật từ dataset và chiếu 3D Bounding Boxes lên mặt ảnh
        Giúp người dùng phổ thông nhìn thấy hộp 3D bao quanh ô tô, người đi bộ ngoài đời
        """
        cam_sd = self.get_sample_sensor_data(sample_token, channel)
        if not cam_sd:
            print(f"[Camera] Không tìm thấy dữ liệu cho {channel}")
            return None

        img_path = os.path.join(self.data_root, cam_sd['filename'])
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            return None
            
        annotated_img = raw_img.copy()
        h, w = annotated_img.shape[:2]

        calib = self.calibs[cam_sd['calibrated_sensor_token']]
        ego = self.egos[cam_sd['ego_pose_token']]
        
        R_ego = quat_to_matrix(ego['rotation'])
        t_ego = np.array(ego['translation'])
        R_cam = quat_to_matrix(calib['rotation'])
        t_cam = np.array(calib['translation'])
        K = np.array(calib['camera_intrinsic'])

        # Chiếu các hộp Ground Truth
        anns = self.sample_to_anns.get(sample_token, [])
        for a in anns:
            cat_name = self.categories.get(self.instances[a['instance_token']]['category_token'], 'object')
            simple_cat = cat_name.split('.')[-1]
            if 'car' in cat_name: simple_cat = 'car'
            elif 'pedestrian' in cat_name: simple_cat = 'pedestrian'
            elif 'truck' in cat_name: simple_cat = 'truck'
            elif 'bus' in cat_name: simple_cat = 'bus'
            elif 'motorcycle' in cat_name: simple_cat = 'motorcycle'
            elif 'bicycle' in cat_name: simple_cat = 'bicycle'

            c_glob = get_box_corners_3d(a['translation'], a['size'], a['rotation'])
            pts_2d, corners_cam = project_box_to_camera(c_glob, R_ego, t_ego, R_cam, t_cam, K, w, h)
            
            if pts_2d is not None:
                dist_m = float(np.mean(corners_cam[2, :]))
                color_bgr = CLASS_COLORS.get(simple_cat, ([0, 255, 0], 'obj'))[0]
                # Đổi RGB sang BGR cho OpenCV
                bgr = (color_bgr[2], color_bgr[1], color_bgr[0])
                label_txt = CLASS_COLORS.get(simple_cat, (None, simple_cat.capitalize()))[1]
                draw_3d_box_on_image(annotated_img, pts_2d, color_bgr=bgr, thickness=2, label=label_txt, dist_m=dist_m)

        # Chèn dải băng HUD thông số trên camera
        hud_bar = np.zeros((50, w, 3), dtype=np.uint8)
        cv2.rectangle(hud_bar, (0, 0), (w, 50), (20, 25, 30), -1)
        hud_text = f"VINFAST ADAS AI VISION | SENSOR: {channel} | RESOLUTION: {w}x{h} | DETECTED 3D BOXES: {len(anns)} OBSTACLES"
        cv2.putText(hud_bar, hud_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        
        final_img = np.vstack([hud_bar, annotated_img])

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, final_img)
            print(f"[Camera 3D Boxes] Đã xuất ảnh camera xử lý hộp 3D tại: {save_path}")
            
        return final_img

    def render_integrated_multimodal_dashboard(self, sample_token, save_path=None):
        """
        Tạo màn hình trực quan hóa tích hợp toàn diện (All-in-One Dashboard):
        - Hàng trên: 3 Camera (Trước-Trái, Trước, Trước-Phải) có vẽ hộp 3D Bounding Box
        - Hàng dưới trái: Bản đồ BEV LiDAR + Bounding Box (giống ảnh mẫu người dùng)
        - Hàng dưới phải: Bản đồ 3D Voxel Occupancy Flow & HUD vi sai an toàn
        """
        # 1. Tạo 3 ảnh camera trước
        cam_front = self.render_camera_with_3d_boxes(sample_token, 'CAM_FRONT')
        cam_left = self.render_camera_with_3d_boxes(sample_token, 'CAM_FRONT_LEFT')
        cam_right = self.render_camera_with_3d_boxes(sample_token, 'CAM_FRONT_RIGHT')

        # Resize camera để ghép hàng ngang
        target_h = 320
        target_w = int(1600 * (target_h / 950))
        
        cf_small = cv2.resize(cam_front, (target_w, target_h)) if cam_front is not None else np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cl_small = cv2.resize(cam_left, (target_w, target_h)) if cam_left is not None else np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cr_small = cv2.resize(cam_right, (target_w, target_h)) if cam_right is not None else np.zeros((target_h, target_w, 3), dtype=np.uint8)

        top_row = np.hstack([cl_small, cf_small, cr_small]) # [target_h, target_w * 3, 3]

        # 2. Render BEV LiDAR ra ảnh đệm
        bev_temp_path = os.path.join(self.output_dir, 'visualization', 'temp_bev.png')
        self.render_bev_lidar_gt_and_pred(sample_token, save_path=bev_temp_path)
        bev_img = cv2.imread(bev_temp_path)
        if os.path.exists(bev_temp_path):
            os.remove(bev_temp_path)

        # 3. Nạp ảnh 3D Isometric đã render trước đó nếu có
        iso_path = os.path.join(self.output_dir, 'visualization', 'occupancy_3d_isometric.png')
        iso_img = cv2.imread(iso_path) if os.path.exists(iso_path) else None

        # Ghép giao diện tổng thể
        total_w = top_row.shape[1]
        half_w = total_w // 2
        bottom_h = 700

        bev_resized = cv2.resize(bev_img, (half_w, bottom_h)) if bev_img is not None else np.zeros((bottom_h, half_w, 3), dtype=np.uint8)
        iso_resized = cv2.resize(iso_img, (half_w, bottom_h)) if iso_img is not None else np.zeros((bottom_h, half_w, 3), dtype=np.uint8)

        bottom_row = np.hstack([bev_resized, iso_resized])
        dashboard = np.vstack([top_row, bottom_row])

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, dashboard)
            print(f"\n[Master Dashboard] ĐÃ XUẤT THÀNH CÔNG BẢN ĐỒ TỔNG HỢP ĐA CẢM BIẾN TẠI: {save_path}")

        return dashboard

    def render_multimodal_4d_video(self, sample_tokens, output_video_path, output_gif_path, fps=2):
        """Render video chuỗi thời gian liên tục qua các frames"""
        rendered_frames = []
        print(f"\n[Multi-Modal Video] Bắt đầu tạo video đa cảm biến cho {len(sample_tokens)} frames...")
        
        for idx, stok in enumerate(sample_tokens):
            print(f"  --> Đang xử lý Frame [{idx+1}/{len(sample_tokens)}] ({stok[:16]}...)")
            dash = self.render_integrated_multimodal_dashboard(stok)
            # Chuyển BGR sang RGB cho video writer
            dash_rgb = cv2.cvtColor(dash, cv2.COLOR_BGR2RGB)
            rendered_frames.append(dash_rgb)

        # Xuất MP4
        try:
            imageio.mimwrite(output_video_path, rendered_frames, fps=fps, codec='libx264', quality=9)
            print(f"[Video MP4] Đã xuất video tại: {output_video_path}")
        except Exception as e:
            print(f"[Video MP4] Lỗi lưu MP4 ({e})")

        # Xuất GIF
        pil_frames = [Image.fromarray(f) for f in rendered_frames]
        pil_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / fps),
            loop=0
        )
        print(f"[Video GIF] Đã xuất ảnh động GIF tại: {output_gif_path}")

def main():
    parser = argparse.ArgumentParser(description="VinFast ADAS Multi-Modal Perception Visualizer")
    parser.add_argument('--data-root', type=str, default='data/nuscenes', help="Thư mục dữ liệu nuScenes")
    parser.add_argument('--model-output', type=str, default='model_output', help="Thư mục kết quả model_output")
    parser.add_argument('--vis-dir', type=str, default='model_output/visualization', help="Thư mục lưu hình ảnh")
    args = parser.parse_args()

    os.makedirs(args.vis_dir, exist_ok=True)

    visualizer = NuScenesMultiModalVisualizer(data_root=args.data_root, model_output_dir=args.model_output)

    # 5 frames trong chuỗi thời gian của scene-0061
    frame_tokens = [
        'ca9a282c9e77460f8360f564131a8af5',
        '39586f9d59004284a7114a68825e8eec',
        '356d81f38dd9473ba590f39e266f54e5',
        'e0845f5322254dafadbbed75aaa07969',
        'c923fe08b2ff4e27975d2bf30934383b'
    ]

    first_frame = frame_tokens[0]

    # 1. Xuất ảnh Camera trước kèm hộp 3D Bounding Box
    cam_box_path = os.path.join(args.vis_dir, 'camera_front_3d_boxes.png')
    visualizer.render_camera_with_3d_boxes(first_frame, 'CAM_FRONT', save_path=cam_box_path)

    # 2. Xuất bản đồ BEV LiDAR Point Cloud + 3D Bounding Boxes (chuẩn phong cách ảnh tham khảo của user)
    bev_lidar_path = os.path.join(args.vis_dir, 'bev_lidar_groundtruth_pred.png')
    visualizer.render_bev_lidar_gt_and_pred(first_frame, save_path=bev_lidar_path)

    # 3. Xuất màn hình tổng hợp Đa Cảm Biến All-in-One Dashboard
    dashboard_path = os.path.join(args.vis_dir, 'multimodal_adas_dashboard.png')
    visualizer.render_integrated_multimodal_dashboard(first_frame, save_path=dashboard_path)

    # 4. Xuất Video MP4 và GIF chuỗi thời gian chuyển động đa cảm biến (5 frames)
    video_path = os.path.join(args.vis_dir, 'multimodal_perception_video.mp4')
    gif_path = os.path.join(args.vis_dir, 'multimodal_perception_animation.gif')
    visualizer.render_multimodal_4d_video(frame_tokens, video_path, gif_path, fps=2)

    print("\n" + "="*80)
    print("  TOÀN BỘ SẢN PHẨM TRỰC QUAN HÓA ĐA CẢM BIẾN ĐÃ HOÀN TẤT!")
    print(f"  1. Ảnh Camera thực tế chiếu hộp 3D  : {cam_box_path}")
    print(f"  2. Bản đồ BEV LiDAR + Hộp GT & Pred : {bev_lidar_path} (Chuẩn ảnh mẫu)")
    print(f"  3. Dashboard Tổng Hợp Đa Cảm Biến   : {dashboard_path}")
    print(f"  4. Video Đa Cảm Biến Chuỗi 4D (MP4) : {video_path}")
    print(f"  5. Ảnh động Đa Cảm Biến 4D (GIF)    : {gif_path}")
    print("="*80)

if __name__ == '__main__':
    main()
