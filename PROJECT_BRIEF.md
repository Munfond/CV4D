# PROJECT BRIEF: VINFAST ADAS BEV/4D OCCUPANCY BENCHMARKING LAB
**Mã dự án:** RAV-25  
**Lĩnh vực:** Robot & Xe tự hành (VinFast Autonomous Driving / ADAS)  
**Thời gian thực hiện:** 6 tuần  
**Phiên bản tài liệu:** v1.0  
**Tình trạng:** Ready for Kick-off  

---

## 1. BỐI CẢNH & TÍNH CẤP THIẾT (BACKGROUND & CONTEXT)

Trong môi trường giao thông đô thị phức tạp tại Việt Nam và các thị trường trọng điểm của VinFast (mật độ xe máy cao, phương tiện lấn làn, chướng ngại vật hình dạng bất định, điểm mù do xe buýt/xe tải che khuất):
- **Hạn chế của phương pháp truyền thống:** Việc chỉ dựa vào mô hình nhận diện hộp bao 3D (3D Bounding Box Object Detection) từ camera hoặc LiDAR đơn lẻ không đủ bảo đảm an toàn cho các tác vụ lập quỹ đạo chuyển động (Motion Planning). 3D Box chỉ mô hình hóa được các vật thể định danh (ô tô, xe máy, người đi bộ), bỏ sót các vật thể không định hình (mảnh vỡ, hàng hóa rơi, cọc tiêu, công trình) và không thể hiện được thể tích chiếm dụng chi tiết (occlusion/free-space boundary).
- **Xu hướng công nghệ:** Mô hình **BEV (Bird's Eye View) kết hợp 4D Occupancy Grid & Occupancy Flow** từ dữ liệu hợp nhất đa cảm biến (Camera + LiDAR + Radar) đang là kiến trúc perception hàng đầu thế giới (Tesla, Waymo, Horizon Robotics).
- **Mục tiêu dự án:** Xây dựng một **Môi trường Thực nghiệm (Lab) & Nền tảng Trực quan hóa/Đánh giá (Benchmarking Web Tool)** mô hình BEV/4D Occupancy Fusion, hỗ trợ đội ngũ kỹ sư ADAS VinFast so sánh, thẩm định các cấu hình cảm biến (Camera-only, Camera+Radar, Camera+LiDAR, Tri-modal) nhằm tối ưu giữa độ chính xác an toàn, độ trễ thời gian thực và chi phí phần cứng xe thương mại.

---

## 2. TUYÊN BỐ MỤC TIÊU DỰ ÁN (PROJECT OBJECTIVES)

1. **Về Kỹ thuật & Mô hình (AI Perception):**
   - Triển khai và chuẩn hóa pipeline trích xuất không gian BEV và lưới chiếm dụng 4D (spatio-temporal occupancy) cùng trường vận tốc (occupancy flow) từ luồng dữ liệu hợp nhất Camera, LiDAR, Radar.
   - Hỗ trợ benchmark các mô hình SOTA (dựa trên kiến trúc BEVFusion, BEVFormer, OccNet-style).
2. **Về Công cụ & Trực quan hóa (Software/Web Platform):**
   - Xây dựng Web App tương tác cao (FastAPI + React/Next.js + Three.js/WebGL) cho phép hiển thị đồng bộ: Video camera góc rộng, Point cloud LiDAR, Radar Doppler vectors, Lưới 3D Voxel/Occupancy, Occupancy Flow và Vùng cảnh báo rủi ro va chạm.
3. **Về Thẩm định & Ra quyết định (Benchmarking & Decision Support):**
   - Cung cấp ma trận so sánh định lượng đa chiều giữa các cấu hình sensor: Độ chính xác (Ray/Voxel IoU, mAP), Sai số chuyển động (ADE/FDE), Độ trễ (Inference Latency) và Mức tiêu thụ tài nguyên GPU/chi phí tính toán.

---

## 3. PHẠM VI DỰ ÁN (SCOPE OF WORK)

### Trong phạm vi (In-Scope):
- Xử lý và chuẩn hóa dữ liệu mô phỏng / benchmark công khai và dữ liệu tương thích: **nuScenes**, **Waymo Open Dataset**, **KITTI** (hỗ trợ camera ring, LiDAR 32-128 chùm tia, Radar point cloud).
- Huấn luyện / Fine-tune / Benchmark mô hình pretrained BEVFusion / BEVFormer / OccNet.
- Pipeline ước lượng Occupancy Flow theo chuỗi thời gian ($T-k$ đến $T$).
- Trực quan hóa 3D không gian thực tế với WebGL/Three.js (xoay, zoom, lọc voxel theo label/confidence, hiển thị vector vận tốc).
- Báo cáo phân tích so sánh: Camera-only vs Camera+LiDAR vs Camera+Radar vs Full Tri-modal.
- Module che mờ/ẩn danh thông tin nhạy cảm (biển số xe, khuôn mặt người đi đường) trên giao diện.

### Ngoài phạm vi (Out-of-Scope):
- Không can thiệp hoặc điều khiển trực tiếp hệ thống phần cứng/cơ cấu chấp hành của xe thật (Drive-by-wire / CAN bus).
- Không triển khai hệ thống nhúng ASIL-D production-grade trong phạm vi 6 tuần (chỉ đo đạc profiling latency giả lập trên GPU workstation/server).
- Không tự thu thập dữ liệu xe chạy thực tế trên đường (sử dụng dữ liệu open-standard và log có sẵn).

---

## 4. RÀNG BUỘC & QUY TẮC AN TOÀN (CONSTRAINTS & POLICIES)

1. **Human-in-the-loop:** Công cụ đóng vai trò phân tích, hỗ trợ kỹ sư ra quyết định; mọi kết quả output phải qua kỹ sư review trước khi đưa vào module lập kế hoạch (planning) hạ nguồn.
2. **Bảo mật & Quyền riêng tư:** Tự động phát hiện và ẩn danh (blur) biển số xe và khuôn mặt trên các camera feeds hiển thị ra giao diện web.
3. **Tối ưu chi phí hạ tầng:** Tối ưu hóa pipeline suy luận (TensorRT / ONNX / FP16/INT8 quantize cơ bản) để kiểm soát chi phí GPU cloud/lab.

---

## 5. KẾ HOẠCH TRIỂN KHAI 6 TUẦN (6-WEEK ROADMAP)

```
Tuần 1: Khởi động, Khảo sát & Chuẩn hóa Data Pipeline (nuScenes/Waymo)
Tuần 2: Thiết lập Baseline BEV Fusion Model (MMDetection3D / PyTorch)
Tuần 3: Tích hợp 4D Spatio-Temporal Occupancy Grid & Flow Estimation
Tuần 4: Phát triển Web Dashboard & 3D Interactive Viewer (FastAPI + Three.js)
Tuần 5: Triển khai Module Benchmarking Sensor Fusion & Phân tích Đánh đổi (Trade-off)
Tuần 6: Tối ưu Hóa Profiling (GPU/Latency), Kiểm thử & Đóng gói Bàn giao (Docker)
```

| Tuần | Mục tiêu chính | Kết quả bàn giao (Deliverables) |
| :--- | :--- | :--- |
| **Tuần 1** | Thiết lập môi trường GPU, Docker, Data Preprocessing pipeline cho nuScenes / Waymo; Module anonymization. | Pipeline nạp dữ liệu Camera + LiDAR + Radar đồng bộ timestamp; Script tự động blur mặt & biển số xe. |
| **Tuần 2** | Triển khai mô hình baseline BEVFusion / BEVFormer; đo mAP 3D detection trên tập dữ liệu chuẩn. | Mô hình baseline chạy ổn định trên PyTorch; Weights & Biases tracking; Báo cáo baseline mAP. |
| **Tuần 3** | Nâng cấp mô hình lên lưới Occupancy 3D và Occupancy Flow 4D (vận tốc di chuyển voxel $v_x, v_y, v_z$). | Checkpoint mô hình 4D Occupancy; Metric đánh giá Ray/Voxel IoU và ADE/FDE cho luồng chuyển động. |
| **Tuần 4** | Xây dựng API Backend (FastAPI) và Frontend WebGL/Three.js hiển thị không gian 3D tương tác. | Giao diện Web: Chế độ Top-down BEV, 3D Voxel viewer, camera carousel, scrubber tua thời gian playback. |
| **Tuần 5** | Xây dựng ma trận benchmark so sánh các cấu hình cảm biến (Camera vs LiDAR vs Radar); phân tích False Positive / False Negative. | Bảng điều khiển phân tích so sánh song song 2 cấu hình; biểu đồ phân tích lỗi theo điều kiện thời tiết/ban đêm. |
| **Tuần 6** | Đo đạc latency inference chi tiết, profiling chi phí GPU; đóng gói Docker Compose và hoàn thiện tài liệu. | Docker image hoàn chỉnh (Backend + Frontend + GPU worker); Báo cáo kỹ thuật tổng kết dự án và video demo. |

---

## 6. TIÊU CHÍ ĐÁNH GIÁ THÀNH CÔNG (SUCCESS METRICS & KPIS)

| Hạng mục | Chỉ số đo lường (KPI) | Mục tiêu tối thiểu (MVP) | Mục tiêu kỳ vọng (Target) |
| :--- | :--- | :--- | :--- |
| **Độ chính xác Occupancy** | Geometric Ray/Voxel IoU | $\ge 40\%$ | $\ge 50\%$ |
| **Độ chính xác 3D Box** | mAP (Mean Average Precision) | $\ge 55\%$ (nuScenes val) | $\ge 65\%$ (nuScenes val) |
| **Độ trôi Occupancy Flow**| ADE / FDE (Average / Final Displacement Error)| $\le 0.8\text{m} / \le 1.5\text{m}$ | $\le 0.5\text{m} / \le 1.0\text{m}$ |
| **Hiệu năng Web UI** | Tốc độ khung hình Render Three.js | $\ge 30\text{ fps}$ | $\ge 60\text{ fps}$ |
| **Tốc độ Inference** | Latency trên GPU NVIDIA RTX/A-series | $\le 100\text{ ms/frame}$ | $\le 45\text{ ms/frame}$ (real-time capable) |
| **Bảo mật dữ liệu** | Tỉ lệ ẩn danh khuôn mặt / biển số | $100\%$ camera test view hiển thị | $100\%$ tự động blur |

---

## 7. CƠ CẤU NHÂN SỰ & VAI TRÒ (TEAM STRUCTURE)

- **1 Lead Perception / AI Engineer:** Chịu trách nhiệm kiến trúc mô hình BEV/Occupancy, pipeline PyTorch/MMDetection3D, metrics IoU/mAP/Flow.
- **1 Full-stack / 3D Visualization Engineer:** Chịu trách nhiệm phát triển FastAPI server, Three.js WebGL rendering, giao diện dashboard Next.js/React.
- **1 Data & MLOps Engineer:** Chịu trách nhiệm tiền xử lý dữ liệu sensor (sync timestamp, radar doppler), setup Docker GPU, Weights & Biases và latency profiling.
