"""
Benchmark Suite: So sánh đối đầu đa cấu hình cảm biến cho VinFast ADAS
- Camera-only
- Camera + Radar
- Camera + LiDAR
- Tri-modal (Camera + LiDAR + Radar)
"""
import os
import sys
import argparse
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader

from configs.base_config import Config
from models.full_4docc_model import VinFast4DOccModel
from pipelines.dataset_loader import NuScenesOccupancyDataset, collate_fn_4docc
from benchmarking.metrics import OccupancyMetrics

def evaluate_configuration(model, dataloader, mode='tri_modal', device='cuda', max_samples=20):
    model.eval()
    metrics = OccupancyMetrics()
    latencies = []
    
    # Đo bộ nhớ VRAM đỉnh
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    prev_bev = None
    sample_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= max_samples:
                break

            # Đưa dữ liệu lên GPU
            batch['imgs'] = batch['imgs'].to(device)
            batch['lidar_pts'] = [pts.to(device) for pts in batch['lidar_pts']]
            batch['radar_pts'] = [pts.to(device) for pts in batch['radar_pts']]
            delta_transform = batch['delta_transform'].to(device)

            # Đo độ trễ suy luận
            if torch.cuda.is_available():
                start_event.record()
                outputs = model(batch, prev_bev=prev_bev, delta_transform=delta_transform, mode=mode)
                end_event.record()
                torch.cuda.synchronize()
                latency_ms = start_event.elapsed_time(end_event)
            else:
                t0 = time.perf_counter()
                outputs = model(batch, prev_bev=prev_bev, delta_transform=delta_transform, mode=mode)
                latency_ms = (time.perf_counter() - t0) * 1000

            latencies.append(latency_ms)
            prev_bev = outputs['bev_feat']

            # Lấy dự đoán
            occ_logits = outputs['occ_logits'][0] # [18, 16, 200, 200]
            occ_pred = torch.argmax(occ_logits, dim=0) # [16, 200, 200]
            flow_pred = outputs['flow_vectors'][0]     # [3, 16, 200, 200]

            occ_gt = batch['gt_occ'][0]
            flow_gt = batch['gt_flow'][0]

            metrics.update(occ_pred, occ_gt, flow_pred, flow_gt)
            sample_count += 1

    results = metrics.compute()
    results['avg_latency_ms'] = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
    results['fps'] = 1000.0 / results['avg_latency_ms'] if results['avg_latency_ms'] > 0 else 0.0
    results['vram_gb'] = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    
    return results

def main():
    parser = argparse.ArgumentParser(description="VinFast ADAS Multi-Sensor Benchmark Suite")
    parser.add_argument('--checkpoint', type=str, default=None, help="Đường dẫn file .pth trọng số")
    parser.add_argument('--synthetic', action='store_true', help="Dùng dữ liệu giả lập để test nhanh")
    parser.add_argument('--max-samples', type=int, default=15, help="Số frame test")
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"[Benchmark] Sử dụng thiết bị: {device}")

    # 1. Khởi tạo mô hình
    model = VinFast4DOccModel(config=Config, pretrained_cam=False).to(device)
    if args.checkpoint:
        print(f"[Benchmark] Nạp trọng số từ {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)

    # 2. Khởi tạo Dataset
    dataset = NuScenesOccupancyDataset(is_synthetic=args.synthetic)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_4docc)

    configs_to_test = ['cam_only', 'cam_radar', 'cam_lidar', 'tri_modal']
    all_results = {}

    print("\n" + "="*80)
    print("         BẮT ĐẦU CHẠY MA TRẬN ĐỐI SOÁT CẢM BIẾN CHO VINFAST ADAS")
    print("="*80)

    for cfg in configs_to_test:
        print(f"\n--> Đang kiểm thử cấu hình: [{cfg.upper()}] ...")
        res = evaluate_configuration(model, dataloader, mode=cfg, device=device, max_samples=args.max_samples)
        all_results[cfg] = res

    # 3. Xuất bảng so sánh tổng hợp
    print("\n" + "="*85)
    print(f"{'CẤU HÌNH CẢM BIẾN':<20} | {'Voxel IoU':<10} | {'mIoU':<8} | {'mAVE':<8} | {'OccScore':<10} | {'Latency':<10} | {'VRAM':<8}")
    print("-" * 85)
    for cfg in configs_to_test:
        r = all_results[cfg]
        print(f"{cfg.upper():<20} | {r['voxel_iou']:>8.2f}% | {r['miou']:>6.2f}% | {r['mave']:>6.2f}m | {r['occ_score']:>8.2f} | {r['avg_latency_ms']:>6.1f}ms | {r['vram_gb']:>5.2f}GB")
    print("="*85)

if __name__ == '__main__':
    import numpy as np
    main()
