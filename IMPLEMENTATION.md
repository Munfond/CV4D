# HƯỚNG DẪN TRIỂN KHAI KỸ THUẬT (TECHNICAL IMPLEMENTATION GUIDE)
## DỰ ÁN: VINFAST ADAS BEV/4D OCCUPANCY BENCHMARKING LAB (RAV-25)
**Thời gian thực hiện:** 6 tuần  
**Mục tiêu:** Xây dựng hệ thống suy luận, đánh giá đối đầu đa cảm biến và trực quan hóa 3D Voxel Occupancy Grid & Motion Flow.  
**Phiên bản:** 1.0 (Ready for Execution)

---

## 1. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT DIRECTORY TREE)

```text
vinfast-bev-occupancy-lab/
├── configs/                          # File cấu hình mô hình, sensor và tham số train
│   ├── base_config.py                # Cấu hình chung về không gian voxel, kích thước grid
│   ├── bevfusion_4docc_mini.py       # Cấu hình kiến trúc mô hình chính
│   └── sensor_benchmark/             # Cấu hình 4 chế độ cảm biến
│       ├── cam_only.py
│       ├── cam_radar.py
│       ├── cam_lidar.py
│       └── tri_modal.py
├── data/                             # Thư mục chứa dữ liệu
│   ├── nuscenes/                     # Dữ liệu nuScenes-mini (samples, sweeps, json)
│   ├── occ3d_cam4d/                  # Nhãn ground truth 3D Voxel & Flow (.npz)
│   └── cache/                        # Cache metadata tiền xử lý (.pkl)
├── docker/                           # Môi trường đóng gói
│   ├── Dockerfile                    # CUDA 11.8 + PyTorch 2.1 + SpConv + MMDetection3D
│   └── docker-compose.yml            # Cấu hình mount volume và GPU passthrough
├── models/                           # Mã nguồn kiến trúc mạng nơ-ron
│   ├── __init__.py
│   ├── backbones/                    # Tầng trích xuất đặc trưng
│   │   ├── camera_resnet.py          # ResNet-50 + FPN
│   │   ├── lidar_voxelnet.py         # SpConv Sparse VoxelNet
│   │   └── radar_pointnet.py         # Radar MLP Encoder (Doppler + RCS)
│   ├── fusion/                       # Tầng chuyển đổi & hợp nhất BEV
│   │   ├── bev_pooling.py            # LSS Depth Distribution Projection
│   │   └── conv_fuser.py             # ConvFuser + SE-Block
│   ├── temporal/                     # Tầng chuỗi thời gian 4D
│   │   ├── ego_alignment.py          # Gióng hàng tọa độ theo Ego-motion
│   │   └── temporal_cross_attn.py    # Temporal Attention Queue
│   └── heads/                        # Các đầu dự đoán
│       ├── occ_head.py               # 3D Semantic Occupancy (17 classes)
│       └── flow_head.py              # 3D Occupancy Flow (vx, vy, vz)
├── pipelines/                        # Tiền xử lý dữ liệu và bảo mật
│   ├── dataset_loader.py             # PyTorch Multi-Modal Dataset & Collate
│   ├── anonymizer.py                 # Module tự động làm mờ mặt và biển số xe
│   └── coordinate_transforms.py      # Ma trận chuyển đổi Extrinsics/Intrinsics
├── benchmarking/                     # Module đánh giá & đo đạc
│   ├── metrics.py                    # Voxel IoU, mIoU, Ray-IoU, mAVE, OccScore
│   ├── latency_profiler.py           # Đo thời gian suy luận chi tiết từng tầng (ms)
│   └── run_benchmark.py              # Script chạy tự động ma trận đối đầu cảm biến
├── visualization/                    # Trực quan hóa bản đồ 3D
│   ├── rerun_visualizer.py           # Tích hợp Rerun.io SDK hiển thị 3D & Playback
│   └── risk_assessment.py            # Thuật toán tính vùng nguy cơ va chạm hình học
├── tools/                            # Scripts tiện ích
│   ├── create_data.py                # Pre-compile metadata ra file .pkl
│   ├── train_finetune.py             # Script chạy fine-tune nhẹ
│   └── demo_inference.py             # Script chạy demo 1-click
├── PRD.md                            # Bản đặc tả yêu cầu sản phẩm
├── PROJECT_BRIEF.md                  # Bản tóm lược điều hành dự án
├── requirements.txt                  # Danh mục thư viện Python
└── README.md                         # Hướng dẫn khởi chạy nhanh
```

---

## 2. THIẾT LẬP MÔI TRƯỜNG DOCKER GPU (ENVIRONMENT SETUP)

### 2.1. File `docker/Dockerfile`
```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Cài đặt thư viện hệ thống
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget curl ffmpeg libgl1 libglib2.0-0 \
    libopenblas-dev libsparsehash-dev && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Nâng cấp pip và cài đặt PyTorch chuẩn CUDA 11.8
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Cài đặt SpConv (Sparse Convolution cho LiDAR)
RUN pip3 install --no-cache-dir spconv-cu118

# Cài đặt các thư viện xe tự hành và trực quan hóa
RUN pip3 install --no-cache-dir \
    nuscenes-devkit \
    open3d \
    rerun-sdk>=0.15.0 \
    ultralytics \
    wandb \
    einops \
    scipy \
    pandas \
    matplotlib

WORKDIR /workspace
CMD ["/bin/bash"]
```

### 2.2. File `docker/docker-compose.yml`
```yaml
version: '3.8'

services:
  vinfast-lab:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: vinfast_bev_occupancy_lab
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    volumes:
      - ..:/workspace
    shm_size: '16gb'
    network_mode: "host"
    stdin_open: true
    tty: true
```

Khởi chạy môi trường:
```bash
docker compose -f docker/docker-compose.yml up -d --build
docker exec -it vinfast_bev_occupancy_lab /bin/bash
```

---

## 3. CHUẨN BỊ DỮ LIỆU & TIỀN XỬ LÝ (DATA PREPARATION)

### 3.1. Tải Dataset mẫu
Nhóm tải dữ liệu đặt vào đúng vị trí sau:
- `data/nuscenes/`: Giải nén `v1.0-mini.tgz` (gồm thư mục `samples/`, `sweeps/`, `v1.0-mini/`).
- `data/occ3d_cam4d/`: Tải bộ nhãn mini của `Occ3D-nuScenes` hoặc `Cam4DOcc` đặt vào `data/occ3d_cam4d/mini_gt/`.

### 3.2. Script `tools/create_data.py` (Tạo Metadata Cache tối ưu RAM)
Thay vì đọc 13 file JSON liên tục, script này chuyển đổi toàn bộ metadata sang file nhị phân `.pkl`:
```python
import os
import pickle
from nuscenes.nuscenes import NuScenes

def generate_infos(data_path, version='v1.0-mini', out_path='data/cache/nuscenes_infos_val.pkl'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    
    val_scenes = [s['name'] for s in nusc.scene if 'mini-val' in s.get('description', '') or True]
    infos = []

    for sample in nusc.sample:
        # Lấy thông tin 6 Camera
        cam_types = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        cams = {}
        for cam in cam_types:
            cam_data = nusc.get('sample_data', sample['data'][cam])
            calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cams[cam] = {
                'data_path': cam_data['filename'],
                'sensor2ego_translation': calib['translation'],
                'sensor2ego_rotation': calib['rotation'],
                'cam_intrinsic': calib['camera_intrinsic']
            }
        
        # Lấy thông tin LiDAR & Radar
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        radar_data = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
        radar_calib = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
        
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        info = {
            'token': sample['token'],
            'timestamp': sample['timestamp'],
            'scene_token': sample['scene_token'],
            'ego2global_translation': ego_pose['translation'],
            'ego2global_rotation': ego_pose['rotation'],
            'cams': cams,
            'lidar': {
                'data_path': lidar_data['filename'],
                'sensor2ego_translation': lidar_calib['translation'],
                'sensor2ego_rotation': lidar_calib['rotation'],
            },
            'radar': {
                'data_path': radar_data['filename'],
                'sensor2ego_translation': radar_calib['translation'],
                'sensor2ego_rotation': radar_calib['rotation'],
            }
        }
        infos.append(info)

    with open(out_path, 'wb') as f:
        pickle.dump(infos, f)
    print(f"Đã lưu thành công {len(infos)} frames metadata vào {out_path}")

if __name__ == '__main__':
    generate_infos('data/nuscenes')
```

### 3.3. Module Ẩn danh hóa Bảo mật (`pipelines/anonymizer.py`)
Tự động làm mờ khuôn mặt và biển số xe trước khi lưu frame hoặc hiển thị:
```python
import cv2
import numpy as np

class PrivacyAnonymizer:
    def __init__(self, blur_kernel=(31, 31)):
        self.blur_kernel = blur_kernel
        # Sử dụng Cascade Classifier có sẵn của OpenCV (nhẹ, chạy cực nhanh)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def anonymize_image(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        out_img = img_bgr.copy()
        for (x, y, w, h) in faces:
            roi = out_img[y:y+h, x:x+w]
            out_img[y:y+h, x:x+w] = cv2.GaussianBlur(roi, self.blur_kernel, 0)
        return out_img
```

---

## 4. MÃ NGUỒN CỐT LÕI: KIẾN TRÚC MÔ HÌNH 4D-OCCFUSION

### 4.1. Tầng Temporal Spatial Alignment (`models/temporal/ego_alignment.py`)
Thực hiện công thức gióng hàng tọa độ thời gian (Equation 9 trong bài báo Survey):
$$F'_{t-1} = \Psi_S(T_{t-1 \to t} \cdot F_{t-1})$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EgoSpatialAlignment(nn.Module):
    def __init__(self, bev_size=(200, 200), voxel_range=(-40.0, 40.0, -40.0, 40.0)):
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.x_min, self.x_max, self.y_min, self.y_max = voxel_range

    def forward(self, prev_feature, delta_transform):
        """
        prev_feature: Tensor [B, C, H, W]
        delta_transform: Ma trận biến đổi 4x4 từ frame (t-1) sang frame (t)
        """
        B, C, H, W = prev_feature.shape
        device = prev_feature.device
        
        # Sinh lưới tọa độ chuẩn hóa
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([x, y, torch.zeros_like(x), torch.ones_like(x)], dim=-1) # [H, W, 4]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1).view(B, -1, 4) # [B, H*W, 4]

        # Áp dụng ma trận biến đổi thân xe
        transformed_grid = torch.bmm(grid, delta_transform.transpose(1, 2))[:, :, :2]
        transformed_grid = transformed_grid.view(B, H, W, 2)

        # Lấy mẫu đặc trưng (Feature Sampling / Warping)
        aligned_feature = F.grid_sample(prev_feature, transformed_grid, align_corners=True, mode='bilinear')
        return aligned_feature
```

### 4.2. Dual-Head: 3D Semantic Occupancy & Occupancy Flow (`models/heads/occ_flow_head.py`)
```python
import torch
import torch.nn as nn

class DualOccFlowHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=18, z_dim=16):
        super().__init__()
        self.z_dim = z_dim
        
        # Nâng chiều từ 2D BEV lên 3D Voxel
        self.deconv3d = nn.Sequential(
            nn.ConvTranspose3d(in_channels // z_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Head 1: Phân loại ngữ nghĩa 3D Occupancy (17 classes + 1 Free)
        self.occ_classifier = nn.Conv3d(32, num_classes, kernel_size=1)
        
        # Head 2: Dự đoán vector dịch chuyển 3D (vx, vy, vz)
        self.flow_regressor = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 3, kernel_size=1) # 3 channels: dx, dy, dz
        )

    def forward(self, bev_feature):
        # bev_feature: [B, C, H, W] -> reshape thành 3D [B, C//16, 16, H, W]
        B, C, H, W = bev_feature.shape
        feat_3d = bev_feature.view(B, C // self.z_dim, self.z_dim, H, W)
        
        feat_3d = self.deconv3d(feat_3d)
        
        occ_logits = self.occ_classifier(feat_3d)  # [B, 18, 16, H, W]
        flow_vectors = self.flow_regressor(feat_3d) # [B, 3, 16, H, W]
        
        return occ_logits, flow_vectors
```

---

## 5. HƯỚNG DẪN FINE-TUNING & CHIẾN LƯỢC ĐÓNG BĂNG (FINE-TUNING PIPELINE)

### 5.1. Script `tools/train_finetune.py`
Khóa 80% trọng số của Camera/LiDAR backbone, chỉ fine-tune tầng Temporal Queue và 2 Heads:
```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from models.fusion.conv_fuser import ConvFuser
from models.heads.occ_flow_head import DualOccFlowHead

def freeze_backbones(model):
    """Bí quyết huấn luyện nhẹ RAM: Đóng băng Backbones"""
    for name, param in model.named_parameters():
        if 'camera_backbone' in name or 'lidar_backbone' in name:
            param.requires_grad = False
            
def train_one_epoch(model, dataloader, optimizer, scaler, device):
    model.train()
    criterion_occ = nn.CrossEntropyLoss(ignore_index=255)
    criterion_flow = nn.SmoothL1Loss(reduction='none')

    for batch in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            occ_pred, flow_pred = model(batch)
            loss_occ = criterion_occ(occ_pred, batch['gt_occ'])
            
            # Chỉ tính flow loss trên các voxel có vật cản di động
            dynamic_mask = batch['dynamic_mask'].unsqueeze(1) # [B, 1, 16, H, W]
            loss_flow = (criterion_flow(flow_pred, batch['gt_flow']) * dynamic_mask).sum() / (dynamic_mask.sum() + 1e-4)
            
            total_loss = loss_occ + 0.2 * loss_flow

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    print(f"Loss Epoch: {total_loss.item():.4f}")
```

Lệnh thực thi fine-tune:
```bash
python tools/train_finetune.py --epochs 8 --batch-size 2 --lr 1e-4 --amp
```

---

## 6. TRỰC QUAN HÓA BẢN ĐỒ 3D OUTPUT VỚI RERUN.IO

### 6.1. Script `visualization/rerun_visualizer.py`
Tự động sinh ra Digital Twin 3D gồm Voxel Mesh, Point cloud, Flow Arrows và Vùng rủi ro va chạm:
```python
import rerun as rr
import numpy as np

class Visualizer4D:
    def __init__(self, session_name="VinFast_ADAS_4D_Lab"):
        rr.init(session_name, spawn=True)

    def log_frame(self, frame_idx, camera_dict, lidar_pts, occ_pred, flow_pred, risk_mask):
        rr.set_time_sequence("frame", frame_idx)

        # 1. Hiển thị 6 Camera quanh xe
        for cam_name, img in camera_dict.items():
            rr.log(f"cameras/{cam_name}", rr.Image(img))

        # 2. Hiển thị Point Cloud LiDAR
        if lidar_pts is not None:
            rr.log("world/lidar", rr.Points3D(lidar_pts[:, :3], colors=[200, 200, 200]))

        # 3. Lọc voxel có vật cản
        occupied = np.where(occ_pred > 0)
        z, y, x = occupied[0], occupied[1], occupied[2]
        
        # Đổi index sang tọa độ mét
        centers = np.stack([x * 0.4 - 40.0, y * 0.4 - 40.0, z * 0.4 - 1.0], axis=-1)
        
        # 4. Gán màu: ĐỎ nếu rủi ro va chạm, XANH nếu an toàn
        colors = np.zeros((len(centers), 3), dtype=np.uint8)
        for i in range(len(centers)):
            if risk_mask[z[i], y[i], x[i]]:
                colors[i] = [255, 0, 0]    # Cảnh báo va chạm cao
            else:
                colors[i] = [0, 180, 255]  # Vật cản thông thường

        rr.log("world/3d_occupancy", rr.Boxes3D(centers=centers, half_sizes=[0.2, 0.2, 0.2], colors=colors))

        # 5. Vẽ vector dòng chảy (Occupancy Flow)
        flow_vecs = flow_pred[:, z, y, x].T # [N, 3]
        rr.log("world/occupancy_flow", rr.Arrows3D(origins=centers, vectors=flow_vecs * 0.5, colors=[255, 255, 0]))
```

---

## 7. BỘ CÔNG CỤ BENCHMARK ĐỐI ĐẦU ĐA CẢM BIẾN (BENCHMARKING SUITE)

### 7.1. Script `benchmarking/run_benchmark.py`
Tự động ngắt mở cảm biến theo cờ tham số để đối soát 4 cấu hình:
```python
import argparse
import time
import torch
import numpy as np

def run_sensor_benchmark(model, dataloader, config_mode='tri_modal'):
    """
    config_mode: 'cam_only', 'cam_radar', 'cam_lidar', 'tri_modal'
    """
    model.eval()
    total_time = 0
    total_frames = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Vô hiệu hóa có chủ đích cảm biến theo chế độ
            if config_mode == 'cam_only':
                batch['lidar_data'] = None
                batch['radar_data'] = None
            elif config_mode == 'cam_radar':
                batch['lidar_data'] = None
            elif config_mode == 'cam_lidar':
                batch['radar_data'] = None

            start = time.perf_counter()
            occ_pred, flow_pred = model(batch)
            torch.cuda.synchronize()
            total_time += (time.perf_counter() - start)
            total_frames += 1

    avg_latency = (total_time / total_frames) * 1000 # ms
    print(f"[{config_mode.upper()}] Latency trung bình: {avg_latency:.2f} ms (~{1000/avg_latency:.1f} FPS)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=['cam_only', 'cam_radar', 'cam_lidar', 'tri_modal'], default='tri_modal')
    args = parser.parse_args()
```

Lệnh thực thi so sánh đối đầu:
```bash
python benchmarking/run_benchmark.py --config cam_only
python benchmarking/run_benchmark.py --config tri_modal
```

---

## 8. LỊCH TRÌNH THỰC HIỆN 6 TUẦN CHI TIẾT (WEEK-BY-WEEK SPRINT)

| Tuần | Mục tiêu chính | Nhiệm vụ kỹ thuật cụ thể | Sản phẩm bàn giao (Deliverables) |
| :---: | :--- | :--- | :--- |
| **Tuần 1** | **Setup & Dữ liệu** | Dựng Docker GPU, tải `nuScenes-mini` + `Occ3D-mini`, viết `create_data.py` sinh cache `.pkl`, hoàn thiện module ẩn danh `anonymizer.py`. | Container Docker chạy mượt mà, file `nuscenes_infos_val.pkl`, demo ảnh camera được blur mặt/biển số. |
| **Tuần 2** | **Baseline Model** | Tải checkpoint pretrained `Cam4DOcc` / `BEVFusion`, nạp thử nghiệm tập validation, thiết lập kết nối Weights & Biases log metrics. | Script test chạy thành công, log baseline IoU và mAP lên W&B dashboard. |
| **Tuần 3** | **4D Temporal & Flow** | Cài đặt `EgoSpatialAlignment` (Eq. 9), nối Temporal Queue, gắn `DualOccFlowHead`, chạy fine-tuning nhẹ 8 epochs với hàm mất mát tổng hợp. | Checkpoint mô hình 4D có khả năng dự đoán đồng thời lưới voxel và vector vận tốc $(v_x, v_y, v_z)$. |
| **Tuần 4** | **3D Visualizer Lab** | Tích hợp Rerun.io SDK: Stream video 6 camera, point cloud, khối lập phương voxel 3D, mũi tên flow và cảnh báo đỏ rủi ro va chạm. | Video demo và giao diện 3D tương tác xoay 360°, tua thời gian mượt mà $\ge 30\text{ FPS}$. |
| **Tuần 5** | **Sensor Benchmark** | Lập trình module ngắt mở cảm biến (`cam_only`, `cam_radar`, `cam_lidar`, `tri_modal`), đo đạc Voxel IoU, mAVE, OccScore, tỉ lệ phanh ma (FP) và bỏ sót (FN). | Bảng ma trận đối đầu định lượng 4 cấu hình cảm biến kèm biểu đồ phân tích đánh đổi (Trade-off Matrix). |
| **Tuần 6** | **Tối ưu & Đóng gói** | Đo latency profiling chi tiết từng lớp mạng bằng `torch.cuda.Event`, tối ưu lượng hóa FP16, đóng gói Docker 1-click và hoàn thiện báo cáo nghiệm thu. | File Docker Image hoàn chỉnh, Runbook hướng dẫn sử dụng, file Báo cáo Thẩm định Kỹ thuật cho VinFast ADAS. |

---

## 9. LỆNH DEMO 1-CLICK CUỐI DỰ ÁN

Để chạy thử nghiệm toàn bộ hệ thống từ đầu đến cuối:
```bash
# 1. Chạy demo hiển thị bản đồ 3D tương tác trên Rerun
python tools/demo_inference.py --scene scene-0061 --checkpoint models/checkpoints/best_4docc.pth --visualize

# 2. Chạy ma trận đánh giá đối soát toàn diện
python benchmarking/run_benchmark.py --compare cam_only tri_modal --output-report reports/benchmark_vinfast.md
```
