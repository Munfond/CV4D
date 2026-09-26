"""
Script Huấn luyện / Fine-tune Mô hình 4D-OccFusion trên Kaggle
Áp dụng:
- Đóng băng Backbone (Freeze Backbones) để nhẹ RAM
- Huấn luyện FP16 Automatic Mixed Precision (AMP)
- Tích lũy Gradient (Gradient Accumulation)
"""
import os
import sys
import argparse
import time

# Tự động thêm thư mục gốc dự án vào PYTHONPATH để tránh lỗi ModuleNotFoundError
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast, GradScaler
    def make_scaler(dev, enabled):
        return GradScaler(dev, enabled=enabled)
    def make_autocast(dev, enabled):
        return autocast(dev, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    def make_scaler(dev, enabled):
        return GradScaler(enabled=enabled)
    def make_autocast(dev, enabled):
        return autocast(enabled=enabled)

from configs.base_config import Config
from models.full_4docc_model import VinFast4DOccModel
from pipelines.dataset_loader import NuScenesOccupancyDataset, collate_fn_4docc

def freeze_backbones(model):
    """Đóng băng trọng số trích xuất đặc trưng của Camera và LiDAR"""
    for param in model.camera_backbone.parameters():
        param.requires_grad = False
    for param in model.lidar_backbone.parameters():
        param.requires_grad = False
    print("[Train] Đã đóng băng Camera & LiDAR Backbone (Tiết kiệm 70% VRAM)")

def main():
    parser = argparse.ArgumentParser(description="VinFast 4DOcc Fine-tuning Script")
    parser.add_argument('--data-root', type=str, default='data/nuscenes')
    parser.add_argument('--cache-path', type=str, default='data/cache/nuscenes_infos_val.pkl')
    parser.add_argument('--occ-gt-root', type=str, default='data/occ3d_cam4d', help="Thư mục nhãn Occ3D")
    parser.add_argument('--epochs', type=int, default=8, help="Số epoch huấn luyện")
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--grad-accum', type=int, default=Config.GRAD_ACCUM_STEPS)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    parser.add_argument('--amp', action='store_true', default=Config.USE_AMP, help="Bật FP16 AMP")
    parser.add_argument('--save-dir', type=str, default='checkpoints', help="Thư mục lưu model")
    parser.add_argument('--synthetic', action='store_true', help="Dùng data giả lập nếu chưa có nuScenes")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = Config.DEVICE
    print(f"[Train] Thiết bị huấn luyện: {device}")

    # 1. Dataset & DataLoader
    dataset = NuScenesOccupancyDataset(
        data_root=args.data_root,
        info_path=args.cache_path,
        occ_gt_root=args.occ_gt_root,
        is_synthetic=args.synthetic
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if not args.synthetic else 0,
        collate_fn=collate_fn_4docc,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 2. Khởi tạo Mô hình
    model = VinFast4DOccModel(config=Config, pretrained_cam=True).to(device)
    freeze_backbones(model)

    # 3. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = make_scaler(device_type, enabled=args.amp)

    # 4. Hàm mất mát (Loss)
    criterion_occ = nn.CrossEntropyLoss(ignore_index=255)
    criterion_flow = nn.SmoothL1Loss(reduction='none')

    print(f"\n[Train] Bắt đầu quá trình Fine-tuning ({args.epochs} epochs)...")
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()

        prev_bev = None
        for step, batch in enumerate(dataloader):
            batch['imgs'] = batch['imgs'].to(device)
            batch['lidar_pts'] = [pts.to(device) for pts in batch['lidar_pts']]
            batch['radar_pts'] = [pts.to(device) for pts in batch['radar_pts']]
            delta_transform = batch['delta_transform'].to(device)
            gt_occ = batch['gt_occ'].to(device)
            gt_flow = batch['gt_flow'].to(device)

            with make_autocast(device_type, enabled=args.amp):
                outputs = model(batch, prev_bev=prev_bev, delta_transform=delta_transform)
                prev_bev = outputs['bev_feat'].detach()

                occ_logits = outputs['occ_logits']    # [B, 18, 16, 200, 200]
                flow_vectors = outputs['flow_vectors'] # [B, 3, 16, 200, 200]

                loss_occ = criterion_occ(occ_logits, gt_occ)

                # Chỉ tính Flow loss trên các voxel có vật cản di động (classes 2->10)
                dynamic_mask = ((gt_occ >= 2) & (gt_occ <= 10)).unsqueeze(1) # [B, 1, 16, 200, 200]
                loss_flow = (criterion_flow(flow_vectors, gt_flow) * dynamic_mask).sum() / (dynamic_mask.sum() + 1e-4)

                loss = (loss_occ + 0.2 * loss_flow) / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum

        scheduler.step()
        epoch_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {epoch_loss:.4f} - Thời gian: {elapsed:.1f}s - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Lưu checkpoint tốt nhất
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            ckpt_path = os.path.join(args.save_dir, 'best_4docc.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"--> Đã lưu checkpoint mới tốt nhất: {ckpt_path}")

    print("\n[Train] QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")

if __name__ == '__main__':
    main()
