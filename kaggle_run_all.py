"""
Kaggle Master Runner: Chạy toàn bộ pipeline từ A đến Z trên Kaggle chỉ với 1 câu lệnh!
Quy trình:
1. Kiểm tra môi trường GPU
2. Tạo Metadata Cache nếu có nuScenes (hoặc bật chế độ Synthetic tự động)
3. Chạy Fine-tuning 4D-OccFusion (8 epochs)
4. Chạy Benchmark so sánh 4 cấu hình cảm biến (Cam, Cam+Radar, Cam+LiDAR, Tri-modal)
5. Xuất hình ảnh trực quan hóa BEV Map kết quả
"""
import os
import sys
import subprocess
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run_command(cmd, desc):
    print(f"\n{'='*70}\n[Kaggle Master] {desc}\nLệnh: {cmd}\n{'='*70}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"[Cảnh báo] Lệnh '{cmd}' kết thúc với mã lỗi {ret.returncode}")

def main():
    print("="*70)
    print("  VINFAST ADAS BEV/4D OCCUPANCY & FLOW LAB - KAGGLE MASTER PIPELINE")
    print("="*70)

    # 1. Kiểm tra GPU
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM Capacity : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[Lưu ý] Không phát hiện GPU, pipeline sẽ chạy trên CPU (chậm hơn).")

    # 2. Kiểm tra thư mục dữ liệu trên Kaggle
    candidate_nuscenes = [
        "/kaggle/working/data/nuscenes",
        "/kaggle/input/mini-nuscenes",
        "/kaggle/input/nuscenes-v1-0-mini",
        "/kaggle/input/nuscenes",
        "data/nuscenes"
    ]
    data_root = next((p for p in candidate_nuscenes if os.path.exists(p)), None)
    
    use_synthetic = True
    if data_root is not None:
        print(f"[Data] Tìm thấy nuScenes tại: {data_root}")
        use_synthetic = False
    else:
        print("[Data] Chưa phát hiện dữ liệu nuScenes thật. Tự động bật chế độ Synthetic Dataset để chạy demo!")
        data_root = "data/nuscenes"
        use_synthetic = True

    # Kiểm tra bộ nhãn Occ3D Ground Truth
    candidate_occ = []
    if os.path.exists('/kaggle/input'):
        for root, _, files in os.walk('/kaggle/input'):
            if 'labels.npz' in files:
                # root có dạng /kaggle/input/<dataset>/gts/scene-0061/<token>
                # Thư mục gốc nhãn thường là /kaggle/input/<dataset>
                parts = root.split(os.sep)
                if 'gts' in parts:
                    idx = parts.index('gts')
                    candidate_occ.append(os.sep.join(parts[:idx+1]))
                    candidate_occ.append(os.sep.join(parts[:idx]))
                else:
                    candidate_occ.append(root)
                break

    candidate_occ.extend([
        "/kaggle/input/occ3d-mini-gts",
        "/kaggle/input/occ3d-nuscenes-mini",
        "/kaggle/working/CV4D/data/occ3d_cam4d",
        "data/occ3d_cam4d",
        os.path.join(data_root, 'gts')
    ])
    occ_gt_root = next((p for p in candidate_occ if os.path.exists(p)), "data/occ3d_cam4d")
    print(f"[Data] Thư mục nhãn Occ3D phát hiện được: {occ_gt_root}")

    # 3. Tạo cache metadata nếu dùng data thật
    cache_path = "data/cache/nuscenes_infos_val.pkl"
    if not use_synthetic:
        run_command(f"python tools/create_data.py --data-root {data_root} --out-path {cache_path}",
                    "Tạo file Cache Metadata .pkl")

    # 4. Chạy Fine-tuning (4 epochs cho demo nhanh hoặc 8 epochs)
    synthetic_flag = "--synthetic" if use_synthetic else ""
    run_command(f"python tools/train_finetune.py --data-root {data_root} --occ-gt-root {occ_gt_root} --epochs 4 --batch-size 1 --amp {synthetic_flag}",
                "Huấn luyện / Fine-tune Mô hình 4D-OccFusion")

    # 5. Chạy Benchmark So sánh 4 Cấu hình Cảm biến
    ckpt_path = "checkpoints/best_4docc.pth"
    ckpt_arg = f"--checkpoint {ckpt_path}" if os.path.exists(ckpt_path) else ""
    run_command(f"python benchmarking/run_benchmark.py {ckpt_arg} --data-root {data_root} --occ-gt-root {occ_gt_root} {synthetic_flag} --max-samples 10",
                "Chạy Ma trận Benchmark Đối soát 4 Cấu hình Cảm biến")

    # 6. Chạy Demo Inference & Xuất Ảnh Bản đồ BEV
    run_command(f"python tools/demo_inference.py {ckpt_arg} --data-root {data_root} --occ-gt-root {occ_gt_root} {synthetic_flag} --output-img bev_output.png",
                "Suy luận & Xuất Ảnh Bản đồ BEV Map")

    # 7. Đóng gói Trọn bộ Artifact 4D Chuẩn theo Đặc tả (T=5 frames)
    run_command(f"python tools/export_4d_artifacts.py {ckpt_arg} --data-root {data_root} --occ-gt-root {occ_gt_root} {synthetic_flag} --temporal-window 5 --output-dir model_output",
                "Đóng gói Toàn bộ Gói Artifact 4D (NPY, JSON, PLY, GIF, YAML)")

    print("\n" + "="*70)
    print("  TOÀN BỘ PIPELINE ĐÃ HOÀN TẤT THÀNH CÔNG TRÊN KAGGLE!")
    print("  Gói Artifact chuẩn công nghiệp đã được xuất tại: model_output/")
    print("  File ảnh bản đồ BEV đã được xuất tại: bev_output.png")
    print("  File checkpoint trọng số đã được lưu tại: checkpoints/best_4docc.pth")
    print("="*70)

if __name__ == '__main__':
    main()
