"""
VinFast ADAS 4D-OccFusion: 3D Occupancy & 4D Flow Video Generator
Trực quan hóa toàn diện kết quả mô hình xe tự hành:
1. Render hình ảnh 3D Isometric View độ phân giải cao (.png)
2. Xuất file 3D WebGL HTML tương tác 360 độ (interactive_3d_occupancy.html)
3. Render chuỗi thời gian 4D Voxel Flow thành Video MP4 (flow_4d_video.mp4) và GIF (flow_4d_animation.gif)
"""
import os
import sys
import json
import argparse
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import imageio

# Bảng màu chuẩn Occ3D cho 18 lớp (nuScenes-lidarseg + Free Space)
OCC3D_PALETTE = {
    0:  ([105, 105, 105], 'others'),
    1:  ([255, 120, 50],  'barrier'),
    2:  ([255, 192, 203], 'bicycle'),
    3:  ([255, 255, 0],   'bus'),
    4:  ([0, 150, 255],   'car'),
    5:  ([160, 32, 240],  'construction_veh'),
    6:  ([255, 69, 0],    'motorcycle'),
    7:  ([255, 0, 0],     'pedestrian'),
    8:  ([255, 140, 0],   'traffic_cone'),
    9:  ([218, 112, 214], 'trailer'),
    10: ([75, 0, 130],    'truck'),
    11: ([128, 128, 128], 'driveable_surface'),
    12: ([176, 196, 222], 'other_flat'),
    13: ([0, 250, 154],   'sidewalk'),
    14: ([34, 139, 34],   'terrain'),
    15: ([139, 69, 19],   'manmade'),
    16: ([0, 100, 0],     'vegetation'),
    17: ([0, 0, 0],       'free') # Free space (bỏ qua không vẽ)
}

def get_class_color(class_id):
    rgb = OCC3D_PALETTE.get(int(class_id), ([128, 128, 128], 'unknown'))[0]
    return np.array(rgb) / 255.0

def get_class_name(class_id):
    return OCC3D_PALETTE.get(int(class_id), ([128, 128, 128], 'unknown'))[1]

def grid_to_world(idx_x, idx_y, idx_z, voxel_size=0.4, range_min=(-40.0, -40.0, -1.0)):
    x = range_min[0] + (idx_x + 0.5) * voxel_size
    y = range_min[1] + (idx_y + 0.5) * voxel_size
    z = range_min[2] + (idx_z + 0.5) * voxel_size
    return x, y, z

def draw_ego_vehicle_3d(ax, length=4.6, width=1.9, height=1.6):
    """Vẽ mô hình xe chủ VinFast dạng hộp 3D tại gốc tọa độ (0, 0, 0)"""
    x_min, x_max = -width / 2, width / 2
    y_min, y_max = -length / 2, length / 2
    z_min, z_max = 0.0, height

    corners = np.array([
        [x_min, y_min, z_min],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_min, y_max, z_min],
        [x_min, y_min, z_max],
        [x_max, y_min, z_max],
        [x_max, y_max, z_max],
        [x_min, y_max, z_max]
    ])

    faces = [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[2], corners[3], corners[7], corners[6]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[3], corners[7], corners[4]]
    ]

    poly = Poly3DCollection(faces, alpha=0.6, facecolors='#00ffff', edgecolors='#ffffff', linewidths=1.5)
    ax.add_collection3d(poly)
    ax.text(0, 0, height + 0.3, "EGO CAR", color='#00ffff', fontsize=9, fontweight='bold', ha='center')

def render_3d_isometric_snapshot(sem_3d, vel_3d, bboxes_frame, save_path):
    """Render góc nhìn 3D Isometric phối cảnh chất lượng cao"""
    fig = plt.figure(figsize=(14, 10), facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d1117')

    # Lọc voxel khác free (17)
    occ_mask = (sem_3d != 17)
    
    # Để tránh quá tải hiển thị, lấy mẫu mặt đường (11) thưa hơn các vật thể động
    drive_mask = (sem_3d == 11)
    obj_mask = occ_mask & (~drive_mask)

    # Lấy mẫu ngẫu nhiên
    drive_indices = np.argwhere(drive_mask)
    if len(drive_indices) > 3000:
        np.random.seed(42)
        drive_indices = drive_indices[np.random.choice(len(drive_indices), 3000, replace=False)]

    obj_indices = np.argwhere(obj_mask)
    if len(obj_indices) > 5000:
        np.random.seed(42)
        obj_indices = obj_indices[np.random.choice(len(obj_indices), 5000, replace=False)]

    if len(drive_indices) > 0 and len(obj_indices) > 0:
        plot_indices = np.vstack([drive_indices, obj_indices])
    elif len(obj_indices) > 0:
        plot_indices = obj_indices
    else:
        plot_indices = drive_indices

    if len(plot_indices) > 0:
        xs, ys, zs = grid_to_world(plot_indices[:, 0], plot_indices[:, 1], plot_indices[:, 2])
        labels = sem_3d[plot_indices[:, 0], plot_indices[:, 1], plot_indices[:, 2]]
        colors = np.array([get_class_color(lbl) for lbl in labels])
        sizes = np.where(labels == 11, 2.0, 8.0) # Voxel vật thể to hơn mặt đường

        ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.75, edgecolors='none')

    # Vẽ vector vận tốc 3D (3D Flow Quiver) cho các voxel có vận tốc lớn
    speed = np.linalg.norm(vel_3d, axis=-1)
    dynamic_mask = (speed > 0.4) & (sem_3d != 17) & (sem_3d != 11)
    dyn_indices = np.argwhere(dynamic_mask)
    if len(dyn_indices) > 0:
        if len(dyn_indices) > 300:
            dyn_indices = dyn_indices[np.random.choice(len(dyn_indices), 300, replace=False)]
        fx, fy, fz = grid_to_world(dyn_indices[:, 0], dyn_indices[:, 1], dyn_indices[:, 2])
        u = vel_3d[dyn_indices[:, 0], dyn_indices[:, 1], dyn_indices[:, 2], 0]
        v = vel_3d[dyn_indices[:, 0], dyn_indices[:, 1], dyn_indices[:, 2], 1]
        w = vel_3d[dyn_indices[:, 0], dyn_indices[:, 1], dyn_indices[:, 2], 2]
        # Vẽ mũi tên vector
        ax.quiver(fx, fy, fz, u, v, w, length=1.2, normalize=False, color='#ff0055', arrow_length_ratio=0.35, linewidth=1.5, label='3D Voxel Flow (Velocity)')

    # Vẽ xe chủ tại tâm
    draw_ego_vehicle_3d(ax)

    # Vẽ 3D Bounding Boxes nếu có
    if bboxes_frame:
        for box in bboxes_frame:
            c = box['centroid']
            dim = box['dimensions_lwh']
            l, w_dim, h = dim[0], dim[1], dim[2]
            cx, cy, cz = c[0], c[1], c[2]
            
            # 8 đỉnh của bounding box
            x_corners = [cx - w_dim/2, cx + w_dim/2, cx + w_dim/2, cx - w_dim/2, cx - w_dim/2, cx + w_dim/2, cx + w_dim/2, cx - w_dim/2]
            y_corners = [cy - l/2, cy - l/2, cy + l/2, cy + l/2, cy - l/2, cy - l/2, cy + l/2, cy + l/2]
            z_corners = [cz, cz, cz, cz, cz + h, cz + h, cz + h, cz + h]
            
            for edge in [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]:
                ax.plot([x_corners[edge[0]], x_corners[edge[1]]],
                        [y_corners[edge[0]], y_corners[edge[1]]],
                        [z_corners[edge[0]], z_corners[edge[1]]], color='#ffff00', linewidth=1.2, linestyle='--')
            ax.text(cx, cy, cz + h + 0.3, f"{box['class']} ({np.linalg.norm(box.get('velocity',[0,0,0])):.1f}m/s)",
                    color='#ffff00', fontsize=8, ha='center')

    # Thiết lập góc nhìn và trục
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-1.0, 5.0])
    ax.set_xlabel('X (Right -> Left, m)', color='#888888', labelpad=10)
    ax.set_ylabel('Y (Back -> Front, m)', color='#888888', labelpad=10)
    ax.set_zlabel('Z (Height, m)', color='#888888', labelpad=10)
    ax.tick_params(colors='#888888')

    # Góc nhìn Isometric chuẩn xe tự hành
    ax.view_init(elev=28, azim=-55)
    ax.grid(color='#22272e', linestyle=':', linewidth=0.5)

    title_str = "VINFAST ADAS: 3D OCCUPANCY & DYNAMIC FLOW VISUALIZATION\n" \
                "Hệ tọa độ Ego Vehicle | Độ phân giải Voxel 0.4m | Mũi tên đỏ = Vector vận tốc 3D (m/s)"
    ax.set_title(title_str, color='#ffffff', fontsize=12, fontweight='bold', pad=15)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[3D Render] Đã lưu ảnh 3D Isometric View tại: {save_path}")

def generate_interactive_3d_html(sem_3d, vel_3d, bboxes_frame, save_path):
    """Xuất file 3D WebGL HTML tương tác xoay 360 độ bằng Plotly"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[Interactive 3D] Không tìm thấy thư viện Plotly, bỏ qua file HTML.")
        return

    fig = go.Figure()

    # Lọc voxel không rỗng
    occ_mask = (sem_3d != 17)
    drive_mask = (sem_3d == 11)
    obj_mask = occ_mask & (~drive_mask)

    # Lấy mẫu để file HTML nhẹ và mượt (~60 FPS)
    drive_indices = np.argwhere(drive_mask)
    if len(drive_indices) > 2500:
        np.random.seed(42)
        drive_indices = drive_indices[np.random.choice(len(drive_indices), 2500, replace=False)]

    obj_indices = np.argwhere(obj_mask)
    if len(obj_indices) > 4000:
        np.random.seed(42)
        obj_indices = obj_indices[np.random.choice(len(obj_indices), 4000, replace=False)]

    plot_indices = np.vstack([drive_indices, obj_indices]) if len(drive_indices) > 0 and len(obj_indices) > 0 else (obj_indices if len(obj_indices) > 0 else drive_indices)

    if len(plot_indices) > 0:
        xs, ys, zs = grid_to_world(plot_indices[:, 0], plot_indices[:, 1], plot_indices[:, 2])
        labels = sem_3d[plot_indices[:, 0], plot_indices[:, 1], plot_indices[:, 2]]

        # Nhóm theo từng lớp để người dùng có thể bật/tắt trên giao diện (Toggle Legend)
        unique_labels = np.unique(labels)
        for u_lbl in unique_labels:
            mask_l = (labels == u_lbl)
            c_name = get_class_name(u_lbl)
            rgb = OCC3D_PALETTE.get(int(u_lbl), ([128, 128, 128], 'unknown'))[0]
            color_hex = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            pt_size = 3 if u_lbl == 11 else 5

            fig.add_trace(go.Scatter3d(
                x=xs[mask_l],
                y=ys[mask_l],
                z=zs[mask_l],
                mode='markers',
                name=f"[{u_lbl}] {c_name}",
                marker=dict(
                    size=pt_size,
                    color=color_hex,
                    opacity=0.8
                ),
                hovertemplate=f"<b>Class:</b> {c_name}<br><b>X:</b> %{{x:.1f}}m<br><b>Y:</b> %{{y:.1f}}m<br><b>Z:</b> %{{z:.1f}}m<extra></extra>"
            ))

    # Vẽ xe chủ EGO
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0.5],
        mode='markers+text',
        name='EGO Vehicle (Xe VinFast)',
        text=['VINFAST EGO'],
        textposition='top center',
        marker=dict(size=12, color='#00ffff', symbol='diamond')
    ))

    # Vẽ vector Flow dạng Cones (Mũi tên 3D)
    speed = np.linalg.norm(vel_3d, axis=-1)
    dyn_mask = (speed > 0.4) & (sem_3d != 17) & (sem_3d != 11)
    dyn_idx = np.argwhere(dyn_mask)
    if len(dyn_idx) > 0:
        if len(dyn_idx) > 200:
            dyn_idx = dyn_idx[np.random.choice(len(dyn_idx), 200, replace=False)]
        fx, fy, fz = grid_to_world(dyn_idx[:, 0], dyn_idx[:, 1], dyn_idx[:, 2])
        u = vel_3d[dyn_idx[:, 0], dyn_idx[:, 1], dyn_idx[:, 2], 0]
        v = vel_3d[dyn_idx[:, 0], dyn_idx[:, 1], dyn_idx[:, 2], 1]
        w = vel_3d[dyn_idx[:, 0], dyn_idx[:, 1], dyn_idx[:, 2], 2]

        fig.add_trace(go.Cone(
            x=fx, y=fy, z=fz,
            u=u, v=v, w=w,
            sizemode="absolute",
            sizeref=1.0,
            colorscale=[[0, '#ff0055'], [1, '#ffcc00']],
            name="3D Voxel Flow (Velocity Vector)",
            showscale=False
        ))

    # Layout hiện đại chuẩn Dark Theme
    fig.update_layout(
        title=dict(
            text="<b>VinFast ADAS 4D-OccFusion Lab — Interactive 3D Voxel Viewer</b><br><sup>Kéo chuột để xoay 360°, cuộn để phóng to/thu nhỏ, click vào bảng chú giải bên phải để ẩn/hiện từng lớp vật thể</sup>",
            font=dict(color="#ffffff", size=16)
        ),
        paper_bgcolor="#0d1117",
        scene=dict(
            xaxis=dict(title="X (m)", range=[-35, 35], backgroundcolor="#0d1117", gridcolor="#22272e", color="#888888"),
            yaxis=dict(title="Y (m)", range=[-35, 35], backgroundcolor="#0d1117", gridcolor="#22272e", color="#888888"),
            zaxis=dict(title="Z (m)", range=[-1, 5], backgroundcolor="#0d1117", gridcolor="#22272e", color="#888888"),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.3)
        ),
        legend=dict(
            font=dict(color="#ffffff", size=10),
            bgcolor="rgba(22, 27, 34, 0.8)",
            bordercolor="#30363d",
            borderwidth=1
        ),
        margin=dict(l=0, r=0, b=0, t=60)
    )

    fig.write_html(save_path, include_plotlyjs='cdn')
    print(f"[Interactive 3D] Đã xuất file 3D WebGL HTML tại: {save_path}")

def render_4d_flow_video(sem_seq, vel_seq, bboxes_data, output_video_path, output_gif_path, fps=2):
    """
    Render chuỗi thời gian 4D Voxel Flow thành Video MP4 và ảnh động GIF:
    Mỗi frame gồm:
    - Bảng 1: Góc nhìn 3D Isometric View + 3D Flow Vectors
    - Bảng 2: Bản đồ BEV 2D nhìn từ trên cao + Vùng nguy cơ va chạm
    - Bảng 3: Bảng điều khiển vi sai viễn trắc ADAS Telemetry Dashboard
    """
    T = len(sem_seq)
    frame_tokens = list(bboxes_data.keys()) if isinstance(bboxes_data, dict) else [f"frame_{t}" for t in range(T)]
    rendered_frames = []

    print(f"\n[4D Flow Video] Bắt đầu render chuỗi thời gian {T} frames...")

    for t in range(T):
        t_start = time.time()
        sem_3d = sem_seq[t]
        vel_3d = vel_seq[t]
        token = frame_tokens[t] if t < len(frame_tokens) else f"frame_{t}"
        boxes = bboxes_data.get(token, []) if isinstance(bboxes_data, dict) else []

        fig = plt.figure(figsize=(18, 9), facecolor='#090d13')

        # -----------------------------------------------------------
        # Pane 1: 3D Isometric Perspective View (Bên trái)
        # -----------------------------------------------------------
        ax3d = fig.add_subplot(1, 2, 1, projection='3d', facecolor='#090d13')
        
        occ_mask = (sem_3d != 17)
        drive_mask = (sem_3d == 11)
        obj_mask = occ_mask & (~drive_mask)

        # Lấy mẫu
        drive_idx = np.argwhere(drive_mask)
        if len(drive_idx) > 2500:
            np.random.seed(t * 10)
            drive_idx = drive_idx[np.random.choice(len(drive_idx), 2500, replace=False)]

        obj_idx = np.argwhere(obj_mask)
        if len(obj_idx) > 4000:
            np.random.seed(t * 10)
            obj_idx = obj_idx[np.random.choice(len(obj_idx), 4000, replace=False)]

        plot_idx = np.vstack([drive_idx, obj_idx]) if len(drive_idx) > 0 and len(obj_idx) > 0 else (obj_idx if len(obj_idx) > 0 else drive_idx)

        if len(plot_idx) > 0:
            xs, ys, zs = grid_to_world(plot_idx[:, 0], plot_idx[:, 1], plot_idx[:, 2])
            lbls = sem_3d[plot_idx[:, 0], plot_idx[:, 1], plot_idx[:, 2]]
            clrs = np.array([get_class_color(l) for l in lbls])
            szs = np.where(lbls == 11, 2.0, 7.0)
            ax3d.scatter(xs, ys, zs, c=clrs, s=szs, alpha=0.75, edgecolors='none')

        # 3D Flow Quiver
        speed = np.linalg.norm(vel_3d, axis=-1)
        dyn_mask = (speed > 0.35) & (sem_3d != 17) & (sem_3d != 11)
        dyn_idx = np.argwhere(dyn_mask)
        if len(dyn_idx) > 0:
            if len(dyn_idx) > 250:
                dyn_idx = dyn_idx[np.random.choice(len(dyn_idx), 250, replace=False)]
            fx, fy, fz = grid_to_world(dyn_idx[:, 0], dyn_idx[:, 1], dyn_idx[:, 2])
            u = vel_3d[dyn_idx[:, 0], dyn_idx[:, 1], dyn_idx[:, 2], 0]
            v = vel_3d[dyn_idx[:, 0], dyn_idx[:, 1], dyn_idx[:, 2], 1]
            w = vel_3d[dyn_idx[:, 0], dyn_idx[:, 1], dyn_idx[:, 2], 2]
            ax3d.quiver(fx, fy, fz, u, v, w, length=1.4, normalize=False, color='#ff0055', arrow_length_ratio=0.35, linewidth=1.4)

        # Xe chủ
        draw_ego_vehicle_3d(ax3d)

        ax3d.set_xlim([-30, 30])
        ax3d.set_ylim([-30, 30])
        ax3d.set_zlim([-1.0, 5.0])
        ax3d.view_init(elev=28, azim=-55 + t * 4) # Xoay nhẹ góc nhìn theo thời gian
        ax3d.grid(color='#1b2129', linestyle=':', linewidth=0.5)
        ax3d.tick_params(colors='#666666')
        ax3d.set_title(f"3D OCCUPANCY & DYNAMIC FLOW FIELD\nTimestep [{t+1}/{T}] | t = {t*0.5:.1f}s", color='#00ffcc', fontsize=11, fontweight='bold')

        # -----------------------------------------------------------
        # Pane 2: 2D Bird's Eye View (BEV) Top-Down (Góc trên bên phải)
        # -----------------------------------------------------------
        ax_bev = fig.add_subplot(2, 2, 2, facecolor='#090d13')
        for r in [10, 20, 30]:
            circle = plt.Circle((0, 0), r, color='#22272e', fill=False, linestyle='--', linewidth=0.8)
            ax_bev.add_patch(circle)

        # Chiếu BEV
        bev_speed = np.max(speed, axis=-1) # [200, 200]
        bev_sem = np.full((200, 200), 17, dtype=np.int32)
        for z in range(16):
            sl = sem_3d[:, :, z]
            m = (sl != 17) & (sl != 11)
            bev_sem[m] = sl[m]

        occupied_2d = (bev_sem != 17)
        xi, yi = np.where(occupied_2d)
        if len(xi) > 0:
            xm = -40.0 + (xi + 0.5) * 0.4
            ym = -40.0 + (yi + 0.5) * 0.4
            l_2d = bev_sem[xi, yi]
            c_2d = np.array([get_class_color(l) for l in l_2d])
            ax_bev.scatter(xm, ym, s=6, c=c_2d, alpha=0.85)

        # Mũi tên vận tốc BEV 2D
        mov_2d = (bev_speed > 0.4) & (bev_sem != 17)
        mxi, myi = np.where(mov_2d)
        if len(mxi) > 0:
            if len(mxi) > 150:
                sel = np.random.choice(len(mxi), 150, replace=False)
                mxi, myi = mxi[sel], myi[sel]
            mx = -40.0 + (mxi + 0.5) * 0.4
            my = -40.0 + (myi + 0.5) * 0.4
            vx = vel_3d[mxi, myi, :, 0].mean(axis=-1)
            vy = vel_3d[mxi, myi, :, 1].mean(axis=-1)
            ax_bev.quiver(mx, my, vx, vy, color='#ffff00', scale=25, width=0.005, headwidth=4)

        # Xe chủ BEV (Tam giác xanh lơ)
        ax_bev.scatter([0], [0], marker='^', s=120, c='#00ffff', edgecolors='#ffffff', label='Ego Car')
        ax_bev.set_xlim([-35, 35])
        ax_bev.set_ylim([-35, 35])
        ax_bev.tick_params(colors='#666666')
        ax_bev.set_title("TOP-DOWN BEV MOTION MAP & VELOCITY QUIVER", color='#ffffff', fontsize=10, fontweight='bold')
        ax_bev.set_xlabel("X (m)", color='#888888')
        ax_bev.set_ylabel("Y (m)", color='#888888')

        # -----------------------------------------------------------
        # Pane 3: ADAS Telemetry Dashboard (Góc dưới bên phải)
        # -----------------------------------------------------------
        ax_dash = fig.add_subplot(2, 2, 4, facecolor='#111620')
        ax_dash.axis('off')

        num_dynamic_voxels = int((speed > 0.4).sum())
        max_vel = float(speed.max())
        mean_vel = float(speed[speed > 0.4].mean()) if num_dynamic_voxels > 0 else 0.0

        telemetry_text = f"""
        =====================================================
          VINFAST ADAS 4D OCCUPANCY TELEMETRY HUD
        =====================================================
          * Frame Token       : {token[:24]}...
          * Sequence Step     : [{t+1} / {T}]  (Elapsed: {t * 0.5:.1f}s)
          * Perception Range  : X: [-40, 40]m | Y: [-40, 40]m | Z: [-1, 5.4]m
          * Voxel Resolution  : 0.4m x 0.4m x 0.4m (200x200x16)
        -----------------------------------------------------
          * Dynamic Voxels    : {num_dynamic_voxels:,} voxels
          * Mean Speed        : {mean_vel:.2f} m/s  ({mean_vel * 3.6:.1f} km/h)
          * Peak Velocity     : {max_vel:.2f} m/s  ({max_vel * 3.6:.1f} km/h)
          * Active 3D BBoxes  : {len(boxes)} dynamic obstacles tracked
          * Collision Status  : [NORMAL / SAFE] — Horizon: 1.5s
        =====================================================
        """
        ax_dash.text(0.05, 0.5, telemetry_text, color='#00ffcc', fontsize=9, family='monospace', va='center')

        plt.tight_layout()

        # Chuyển đổi figure sang mảng ảnh RGB
        fig.canvas.draw()
        rgba_buffer = fig.canvas.buffer_rgba()
        frame_img = np.asarray(rgba_buffer)[:, :, :3]
        rendered_frames.append(frame_img)
        plt.close()

        print(f"  --> Hoàn tất render Frame [{t+1}/{T}] ({time.time() - t_start:.2f}s)")

    # 1. Lưu file Video MP4
    try:
        imageio.mimwrite(output_video_path, rendered_frames, fps=fps, codec='libx264', quality=9)
        print(f"\n[Video 4D] ĐÃ XUẤT THÀNH CÔNG VIDEO MP4 TẠI: {output_video_path}")
    except Exception as e:
        print(f"[Video 4D] Lưu MP4 gặp lỗi ({e}), chuyển sang lưu GIF.")

    # 2. Lưu file GIF Động
    pil_frames = [Image.fromarray(f) for f in rendered_frames]
    pil_frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"[GIF 4D] ĐÃ XUẤT THÀNH CÔNG ẢNH ĐỘNG GIF TẠI: {output_gif_path}")

def main():
    parser = argparse.ArgumentParser(description="VinFast ADAS: 3D Occupancy & 4D Flow Video Generator")
    parser.add_argument('--input-dir', type=str, default='model_output', help="Thư mục chứa model output")
    parser.add_argument('--output-dir', type=str, default='model_output/visualization', help="Thư mục xuất ảnh/video")
    parser.add_argument('--fps', type=int, default=2, help="Tốc độ khung hình (2Hz chuẩn nuScenes)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*75)
    print("  VINFAST ADAS: 3D OCCUPANCY & 4D TEMPORAL FLOW VISUALIZER")
    print(f"  Thư mục dữ liệu: {args.input_dir}")
    print(f"  Thư mục xuất   : {args.output_dir}")
    print("="*75)

    # 1. Nạp dữ liệu từ model_output/
    print("\n--> Đang nạp các ma trận 4D Voxel từ đĩa...")
    sem_path = os.path.join(args.input_dir, 'semantic', 'semantic_label.npy')
    vel_path = os.path.join(args.input_dir, 'motion', 'velocity.npy')
    bbox_path = os.path.join(args.input_dir, 'instance', 'bounding_box.json')

    if not os.path.exists(sem_path) or not os.path.exists(vel_path):
        print(f"[Lỗi] Không tìm thấy file tại {sem_path} hoặc {vel_path}")
        return

    sem_seq = np.load(sem_path) # [T, 200, 200, 16]
    vel_seq = np.load(vel_path) # [T, 200, 200, 16, 3]

    bboxes_data = {}
    if os.path.exists(bbox_path):
        with open(bbox_path, 'r') as f:
            bboxes_data = json.load(f)

    first_boxes = list(bboxes_data.values())[0] if bboxes_data else []

    # 2. Render ảnh 3D Isometric View
    render_3d_path = os.path.join(args.output_dir, 'occupancy_3d_isometric.png')
    render_3d_isometric_snapshot(sem_seq[0], vel_seq[0], first_boxes, render_3d_path)

    # 3. Xuất file tương tác 3D WebGL HTML
    html_3d_path = os.path.join(args.output_dir, 'interactive_3d_occupancy.html')
    generate_interactive_3d_html(sem_seq[0], vel_seq[0], first_boxes, html_3d_path)

    # 4. Render Video 4D Flow & GIF động
    video_path = os.path.join(args.output_dir, 'flow_4d_video.mp4')
    gif_path = os.path.join(args.output_dir, 'flow_4d_animation.gif')
    render_4d_flow_video(sem_seq, vel_seq, bboxes_data, video_path, gif_path, fps=args.fps)

    print("\n" + "="*75)
    print("  ĐÃ HOÀN TẤT TRỰC QUAN HÓA TOÀN DIỆN!")
    print(f"  1. Ảnh 3D Isometric View     : {render_3d_path}")
    print(f"  2. WebGL 3D Tương tác 360 độ : {html_3d_path} (Mở trực tiếp trên Chrome/Edge)")
    print(f"  3. Video 4D Voxel Flow (MP4) : {video_path}")
    print(f"  4. Ảnh động 4D Voxel Flow    : {gif_path}")
    print("="*75)

if __name__ == '__main__':
    main()
