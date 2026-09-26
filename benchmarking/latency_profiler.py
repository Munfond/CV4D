"""
Latency Profiler: Đo đạc chi tiết thời gian suy luận (Breakdown Latency) từng tầng mạng
Sử dụng torch.cuda.Event để đạt độ chính xác micro-second trên GPU
Phục vụ mục tiêu tối ưu hóa phần cứng và định mức GPU cho VinFast ADAS
"""
import time
import torch
import numpy as np

class HardwareLatencyProfiler:
    def __init__(self, device='cuda'):
        self.device = device
        self.is_cuda = torch.cuda.is_available() and 'cuda' in device

    def profile_model_breakdown(self, model, sample_batch, num_warmup=5, num_runs=20, mode='tri_modal'):
        """
        Đo chi tiết thời gian chạy từng khâu:
        1. Camera Backbone + BEV Pool
        2. LiDAR Backbone
        3. Radar Backbone
        4. ConvFuser (Sensor Fusion)
        5. Temporal Alignment & Queue
        6. Dual-Heads (Occupancy & Flow)
        """
        model.eval()
        imgs = sample_batch['imgs'].to(self.device)
        lidar_pts = [pts.to(self.device) for pts in sample_batch['lidar_pts']] if 'lidar_pts' in sample_batch else None
        radar_pts = [pts.to(self.device) for pts in sample_batch['radar_pts']] if 'radar_pts' in sample_batch else None
        delta_transform = sample_batch.get('delta_transform', torch.eye(4)).to(self.device)

        # 1. Warmup GPU
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(sample_batch, mode=mode)
            if self.is_cuda:
                torch.cuda.synchronize()

        times = {
            'cam_branch': [],
            'lidar_branch': [],
            'radar_branch': [],
            'fusion_neck': [],
            'temporal_module': [],
            'heads': [],
            'total_e2e': []
        }

        with torch.no_grad():
            for _ in range(num_runs):
                t_start = time.perf_counter()

                # A. Camera
                t0 = time.perf_counter()
                cam_feat = model.camera_backbone(imgs)
                cam_bev = model.cam_bev_pool(cam_feat)
                if self.is_cuda: torch.cuda.synchronize()
                times['cam_branch'].append((time.perf_counter() - t0) * 1000)

                # B. LiDAR
                t0 = time.perf_counter()
                lidar_bev = model.lidar_backbone(lidar_pts) if (lidar_pts and mode in ['cam_lidar', 'tri_modal']) else None
                if self.is_cuda: torch.cuda.synchronize()
                times['lidar_branch'].append((time.perf_counter() - t0) * 1000)

                # C. Radar
                t0 = time.perf_counter()
                radar_bev = model.radar_backbone(radar_pts) if (radar_pts and mode in ['cam_radar', 'tri_modal']) else None
                if self.is_cuda: torch.cuda.synchronize()
                times['radar_branch'].append((time.perf_counter() - t0) * 1000)

                # D. Fusion Neck
                t0 = time.perf_counter()
                fused_bev = model.conv_fuser(cam_bev, lidar_bev, radar_bev, mode=mode)
                if self.is_cuda: torch.cuda.synchronize()
                times['fusion_neck'].append((time.perf_counter() - t0) * 1000)

                # E. Temporal
                t0 = time.perf_counter()
                fused_4d = model.temporal_queue(fused_bev, None)
                if self.is_cuda: torch.cuda.synchronize()
                times['temporal_module'].append((time.perf_counter() - t0) * 1000)

                # F. Heads
                t0 = time.perf_counter()
                occ_logits, flow_vecs = model.head(fused_4d)
                if self.is_cuda: torch.cuda.synchronize()
                times['heads'].append((time.perf_counter() - t0) * 1000)

                times['total_e2e'].append((time.perf_counter() - t_start) * 1000)

        # Tổng hợp kết quả
        breakdown = {k: float(np.mean(v)) for k, v in times.items()}
        breakdown['fps'] = 1000.0 / breakdown['total_e2e'] if breakdown['total_e2e'] > 0 else 0.0

        if self.is_cuda:
            breakdown['peak_vram_gb'] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            breakdown['peak_vram_gb'] = 0.0

        return breakdown

    def print_report(self, breakdown, mode='tri_modal'):
        print("\n" + "="*65)
        print(f"       BÁO CÁO PHÂN TÍCH ĐỘ TRỄ PHẦN CỨNG [{mode.upper()}]")
        print("="*65)
        print(f"1. Nhánh Camera (ResNet + LSS BEVPool) : {breakdown['cam_branch']:>6.2f} ms")
        print(f"2. Nhánh LiDAR (PillarNet)             : {breakdown['lidar_branch']:>6.2f} ms")
        print(f"3. Nhánh Radar (Doppler PointNet)      : {breakdown['radar_branch']:>6.2f} ms")
        print(f"4. Tầng Hợp nhất (ConvFuser + SE-Block) : {breakdown['fusion_neck']:>6.2f} ms")
        print(f"5. Tầng Chuỗi thời gian 4D (Temporal)   : {breakdown['temporal_module']:>6.2f} ms")
        print(f"6. Tầng Dự đoán (Occ & Flow Heads)     : {breakdown['heads']:>6.2f} ms")
        print("-" * 65)
        print(f" TỔNG ĐỘ TRỄ SUY LUẬN (End-to-End)     : {breakdown['total_e2e']:>6.2f} ms")
        print(f" TỐC ĐỘ XỬ LÝ KHUNG HÌNH (FPS)         : {breakdown['fps']:>6.1f} FPS")
        print(f" DUNG LƯỢNG VRAM GPU TIÊU THỤ          : {breakdown['peak_vram_gb']:>6.2f} GB")
        print("="*65)
