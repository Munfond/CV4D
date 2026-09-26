# VinFast ADAS: BEV/4D Occupancy & Flow Lab (RAV-25)

Hệ thống huấn luyện, kiểm thử và trực quan hóa mô hình nhận thức không gian 3D/4D từ đa cảm biến (**Camera 360°, LiDAR và Radar**) phục vụ xe tự hành VinFast.

---

## 🚀 HƯỚNG DẪN CHẠY TRÊN KAGGLE (MIỄN PHÍ - 1 CLICK)

### Cách 1: Tải trực tiếp file Notebook lên Kaggle
1. Vào [Kaggle.com](https://www.kaggle.com/) -> Bấm **Create** -> **New Notebook**.
2. Chọn **File** -> **Upload Notebook** -> Chọn file [`kaggle_notebook.ipynb`](file:///C:/Users/nguye/.gemini/antigravity/scratch/vinfast-bev-occupancy-lab/kaggle_notebook.ipynb).
3. Bật **GPU T4 x 2** và gạt **Internet: ON** ở menu bên phải.
4. Bấm **Run All**!

### Cách 2: Chạy dòng lệnh trong Kaggle
```bash
# 1. Cài đặt thư viện
!pip install -q pyquaternion nuscenes-devkit einops

# 2. Chạy Master Pipeline (Tự động nạp dữ liệu, Train FP16, Benchmark 4 cấu hình và xuất ảnh BEV)
!python kaggle_run_all.py
```

---

## 📁 CẤU TRÚC CODEBASE

- **`configs/`**: Cấu hình không gian Voxel ($200 \times 200 \times 16$), voxel size $0.4\text{m}$, 18 classes.
- **`models/`**:
  - `backbones/`: ResNet-18/50 (Camera), PillarNet (LiDAR), Radar PointNet (Doppler velocity).
  - `fusion/`: LSS Depth Projection (BEVPooling) & ConvFuser (SE-Block).
  - `temporal/`: Gióng hàng tọa độ thời gian thân xe `EgoSpatialAlignment` (Eq. 9) & `TemporalFusionQueue`.
  - `heads/`: `DualOccFlowHead` (Dự đoán đồng thời lưới voxel $200 \times 200 \times 16$ và vector vận tốc 3D $v_x, v_y, v_z$).
  - `full_4docc_model.py`: Mạng nơ-ron tổng hợp toàn trình `VinFast4DOccModel`.
- **`pipelines/`**: DataLoader nạp dữ liệu đa cảm biến nuScenes (kèm module tự động tạo mock data giả lập để test nhanh không cần tải 4GB), module `PrivacyAnonymizer` tự động làm mờ mặt và biển số xe.
- **`benchmarking/`**: Đo đạc độ trễ inference (ms), VRAM (GB), Voxel IoU, mIoU, mAVE và `OccScore` theo chuẩn CVPR 2024.
- **`visualization/`**: Vẽ bản đồ BEV Map Snapshot chuyên nghiệp (Ego car, LiDAR points, Radar markers, Voxel boxes, Flow arrows) tương tự các hệ thống xe tự hành thực tế.
- **`tools/`**:
  - `create_data.py`: Pre-compile metadata sang file `.pkl`.
  - `train_finetune.py`: Script huấn luyện đóng băng backbone, bật FP16 AMP.
  - `demo_inference.py`: Chạy thử 1-click xuất file ảnh `bev_output.png`.
