"""
Script CLI: Chạy Suy Luận Chuỗi Thời Gian 4D & Xuất Trọn Bộ Artifact
Tạo cây thư mục 'model_output/' đầy đủ 100% theo đặc tả yêu cầu:
- occupancy/ (occupancy_probability.npy, occupancy_label.npy)
- semantic/ (semantic_logits.npy, semantic_probability.npy, semantic_label.npy)
- motion/ (flow.npy, velocity.npy, motion_mask.npy)
- instance/ (instance_id.npy, bounding_box.json, trajectory.json)
- uncertainty/ (occupancy_uncertainty.npy, semantic_uncertainty.npy, motion_uncertainty.npy)
- visualization/ (bev.png, occupancy_3d.ply, temporal.gif)
- evaluation/ (metrics.json, confusion_matrix.png, per_class_metrics.csv)
- metadata/ (scene_metadata.yaml)
"""
import os
import sys
import argparse
import time
import torch
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.base_config import Config
from models.full_4docc_model import VinFast4DOccModel
from pipelines.dataset_loader import NuScenesOccupancyDataset, collate_fn_4docc
from pipelines.artifact_exporter import ArtifactExporter
from benchmarking.metrics import OccupancyMetrics

def main():
    parser = argparse.ArgumentParser(description="VinFast 4D Occupancy Artifact Exporter")
    parser.add_argument('--checkpoint', type=str, default=None, help="File .pth trọng số mô hình")
    parser.add_argument('--data-root', type=str, default='data/nuscenes', help="Thư mục nuScenes")
    parser.add_argument('--cache-path', type=str, default='data/cache/nuscenes_infos_val.pkl', help="Cache metadata .pkl")
    parser.add_argument('--occ-gt-root', type=str, default='data/occ3d_cam4d', help="Thư mục nhãn Occ3D")
    parser.add_argument('--temporal-window', type=int, default=5, help="Số timestep T trong chuỗi 4D (mặc định 5 frames)")
    parser.add_argument('--output-dir', type=str, default='model_output', help="Thư mục xuất package artifact")
    parser.add_argument('--synthetic', action='store_true', default=False, help="Dùng data giả lập nếu chưa có nuScenes")
    parser.add_argument('--mode', type=str, default='tri_modal', choices=['cam_only', 'cam_radar', 'cam_lidar', 'tri_modal'])
    args = parser.parse_args()

    device = Config.DEVICE
    print("="*80)
    print("  VINFAST ADAS: 4D OCCUPANCY ARTIFACT GENERATOR & EXPORTER PIPELINE")
    print(f"  Thiết bị: {device} | Cửa sổ thời gian T = {args.temporal_window} frames (Δt = {Config.DELTA_T}s)")
    print(f"  Thư mục đích: {args.output_dir}/")
    print("="*80)

    # 1. Khởi tạo Dataset
    dataset = NuScenesOccupancyDataset(
        data_root=args.data_root,
        info_path=args.cache_path,
        occ_gt_root=args.occ_gt_root,
        is_synthetic=args.synthetic,
        synthetic_len=max(20, args.temporal_window)
    )

    # 2. Khởi tạo Mô hình
    model = VinFast4DOccModel(config=Config, pretrained_cam=False).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[Model] Đang nạp trọng số từ: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    # 3. Chạy suy luận chuỗi thời gian liên tục T frames
    T = min(args.temporal_window, len(dataset))
    print(f"\n--> Bắt đầu chạy suy luận liên tục {T} frames để thu thập chuỗi thời gian 4D...")

    logits_list = []
    flow_list = []
    gt_occ_list = []
    gt_flow_list = []
    frame_ids = []
    timestamps = []

    metrics_tracker = OccupancyMetrics()
    prev_bev = None
    latencies = []

    with torch.no_grad():
        for t in range(T):
            sample = dataset[t]
            batch = collate_fn_4docc([sample])
            batch['imgs'] = batch['imgs'].to(device)
            batch['lidar_pts'] = [pts.to(device) for pts in batch['lidar_pts']]
            batch['radar_pts'] = [pts.to(device) for pts in batch['radar_pts']]
            delta_transform = batch['delta_transform'].to(device)

            t0 = time.perf_counter()
            outputs = model(batch, prev_bev=prev_bev, delta_transform=delta_transform, mode=args.mode)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            # Cập nhật hàng đợi đặc trưng thời gian
            prev_bev = outputs['bev_feat'].detach()

            occ_logits = outputs['occ_logits'][0]    # [18, 16, 200, 200]
            flow_vectors = outputs['flow_vectors'][0] # [3, 16, 200, 200]
            gt_occ = batch['gt_occ'][0]               # [16, 200, 200]
            gt_flow = batch['gt_flow'][0]             # [3, 16, 200, 200]

            logits_list.append(occ_logits.cpu())
            flow_list.append(flow_vectors.cpu())
            gt_occ_list.append(gt_occ.cpu())
            gt_flow_list.append(gt_flow.cpu())

            frame_tok = batch['tokens'][0]
            frame_ids.append(frame_tok)
            timestamps.append(int(time.time() * 1e6) + t * int(Config.DELTA_T * 1e6))

            # Đo đạc chất lượng
            pred_label = torch.argmax(occ_logits, dim=0)
            metrics_tracker.update(pred_label, gt_occ, flow_vectors, gt_flow)
            print(f"  Frame [{t+1}/{T}] ({frame_tok[:16]}...) - Độ trễ: {lat:.1f}ms")

    # 4. Tính toán kết quả đánh giá
    eval_results = metrics_tracker.compute()
    eval_results['latency_ms'] = float(np.mean(latencies))
    eval_results['fps'] = float(1000.0 / eval_results['latency_ms'])
    eval_results['confusion_matrix'] = metrics_tracker.confusion_matrix

    print("\n" + "-"*60)
    print(f"  Voxel IoU    : {eval_results['voxel_iou']:.2f}%")
    print(f"  Semantic mIoU: {eval_results['miou']:.2f}%")
    print(f"  Motion mAVE  : {eval_results['mave']:.2f} m/s")
    print(f"  OccScore     : {eval_results['occ_score']:.2f}")
    print(f"  Độ trễ TB    : {eval_results['latency_ms']:.1f} ms (~{eval_results['fps']:.1f} FPS)")
    print("-"*60)

    # 5. Xuất trọn bộ Artifact
    temporal_predictions = {
        'logits': logits_list,
        'flow': flow_list,
        'gt_occ': gt_occ_list,
        'gt_flow': gt_flow_list
    }
    temporal_metadata = {
        'scene_id': 'scene-0061' if not args.synthetic else 'scene-synthetic-0',
        'frame_ids': frame_ids,
        'timestamps': timestamps,
        'delta_t': Config.DELTA_T,
        'split': 'v1.0-mini'
    }

    exporter = ArtifactExporter(output_dir=args.output_dir, config=Config)
    exporter.export_all(temporal_predictions, temporal_metadata, evaluation_data=eval_results)

    # 6. In cây thư mục kết quả
    print_directory_tree(args.output_dir)

def print_directory_tree(root_dir):
    print("\nCẤU TRÚC GÓI ARTIFACT ĐÃ XUẤT:")
    print(f"{root_dir}/")
    for root, dirs, files in os.walk(root_dir):
        rel_path = os.path.relpath(root, root_dir)
        if rel_path == ".":
            indent = "├── "
        else:
            indent = "│   ├── "
            print(f"├── {os.path.basename(root)}/")
        for f in files:
            f_path = os.path.join(root, f)
            sz = os.path.getsize(f_path)
            if sz > 1024*1024:
                sz_str = f"{sz/(1024*1024):.2f} MB"
            elif sz > 1024:
                sz_str = f"{sz/1024:.1f} KB"
            else:
                sz_str = f"{sz} B"
            print(f"│   │   ├── {f} ({sz_str})")
    print("\n TOÀN BỘ ARTIFACT 4D ĐÃ ĐƯỢC TẠO HOÀN CHỈNH!")

if __name__ == '__main__':
    import numpy as np
    main()
