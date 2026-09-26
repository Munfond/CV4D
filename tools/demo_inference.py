"""
Script Demo Suy luận 1-Click (Demo Inference & BEV Map Export)
Chạy suy luận qua mô hình 4D-OccFusion và xuất trực tiếp bản đồ BEV Map ra file ảnh!
"""
import os
import sys
import argparse
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch

from configs.base_config import Config
from models.full_4docc_model import VinFast4DOccModel
from pipelines.dataset_loader import NuScenesOccupancyDataset, collate_fn_4docc
from visualization.visualizer import OccupancyVisualizer
from benchmarking.metrics import OccupancyMetrics

def main():
    parser = argparse.ArgumentParser(description="VinFast 4DOcc 1-Click Demo")
    parser.add_argument('--checkpoint', type=str, default=None, help="File .pth trọng số (nếu có)")
    parser.add_argument('--data-root', type=str, default='data/nuscenes', help="Thư mục nuScenes")
    parser.add_argument('--cache-path', type=str, default='data/cache/nuscenes_infos_val.pkl', help="Cache metadata .pkl")
    parser.add_argument('--occ-gt-root', type=str, default='data/occ3d_cam4d', help="Thư mục nhãn Occ3D")
    parser.add_argument('--synthetic', action='store_true', default=False, help="Dùng dữ liệu giả lập để test nhanh")
    parser.add_argument('--mode', type=str, default='tri_modal', choices=['cam_only', 'cam_radar', 'cam_lidar', 'tri_modal'])
    parser.add_argument('--output-img', type=str, default='bev_output.png', help="File ảnh output BEV")
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"[Demo] Khởi động VinFast ADAS 4D Occupancy Lab trên thiết bị: {device}")

    # 1. Khởi tạo mô hình
    model = VinFast4DOccModel(config=Config, pretrained_cam=False).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[Demo] Nạp trọng số từ: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    model.eval()

    # 2. Lấy 1 mẫu dữ liệu
    dataset = NuScenesOccupancyDataset(
        data_root=args.data_root,
        info_path=args.cache_path,
        occ_gt_root=args.occ_gt_root,
        is_synthetic=args.synthetic
    )
    sample = dataset[0]
    batch = collate_fn_4docc([sample])

    batch['imgs'] = batch['imgs'].to(device)
    batch['lidar_pts'] = [pts.to(device) for pts in batch['lidar_pts']]
    batch['radar_pts'] = [pts.to(device) for pts in batch['radar_pts']]
    delta_transform = batch['delta_transform'].to(device)

    # 3. Chạy suy luận & Đo thời gian
    print(f"[Demo] Đang chạy suy luận với cấu hình: [{args.mode.upper()}] ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(batch, delta_transform=delta_transform, mode=args.mode)
    latency_ms = (time.perf_counter() - t0) * 1000

    occ_logits = outputs['occ_logits'][0]    # [18, 16, 200, 200]
    occ_pred = torch.argmax(occ_logits, dim=0) # [16, 200, 200]
    flow_pred = outputs['flow_vectors'][0]     # [3, 16, 200, 200]

    # 4. Đánh giá nhanh
    metrics = OccupancyMetrics()
    metrics.update(occ_pred, batch['gt_occ'][0], flow_pred, batch['gt_flow'][0])
    res = metrics.compute()

    print("\n" + "="*60)
    print("           KẾT QUẢ SUY LUẬN VINFAST ADAS 4D LAB")
    print("="*60)
    print(f"Cấu hình cảm biến : {args.mode.upper()}")
    print(f"Độ trễ suy luận   : {latency_ms:.2f} ms (~{1000/latency_ms:.1f} FPS)")
    print(f"Voxel IoU (Hình học) : {res['voxel_iou']:.2f}%")
    print(f"mIoU (Ngữ nghĩa)     : {res['miou']:.2f}%")
    print(f"Sai số vận tốc mAVE  : {res['mave']:.2f} m/s")
    print(f"OccScore tổng hợp   : {res['occ_score']:.2f}")
    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"VRAM GPU tiêu thụ  : {vram:.2f} GB")
    print("="*60)

    # 5. Phân tích vùng nguy cơ va chạm hình học (Collision Risk Assessment)
    from visualization.risk_assessment import CollisionRiskAssessment
    risk_analyzer = CollisionRiskAssessment(horizon_sec=1.5)
    risk_mask = risk_analyzer.evaluate_risk(occ_pred, flow_pred, ego_speed_mps=8.0)
    high_risk_count = int((risk_mask == 2).sum())
    caution_count = int((risk_mask == 1).sum())
    print(f"Cảnh báo va chạm : {high_risk_count} voxels nguy cơ cao (Đỏ) | {caution_count} voxels chú ý (Vàng)")

    # 6. Xuất hình ảnh bản đồ BEV
    visualizer = OccupancyVisualizer(use_rerun=False)
    visualizer.plot_bev_snapshot(
        occ_pred=occ_pred,
        occ_gt=batch['gt_occ'][0],
        flow_pred=flow_pred,
        lidar_pts=batch['lidar_pts'][0],
        radar_pts=batch['radar_pts'][0],
        risk_mask=risk_mask,
        save_path=args.output_img
    )
    print(f"\n[Demo] ĐÃ XUẤT THÀNH CÔNG BẢN ĐỒ BEV RA FILE: {args.output_img}")

if __name__ == '__main__':
    main()
