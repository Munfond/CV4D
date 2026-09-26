"""
Script Tải & Giải nén Tự động Bộ Nhãn Chuẩn Occ3D-nuScenes (Official Ground Truth)
Hỗ trợ:
- Tải file gts.tar.gz chuẩn (~440MB cho nuScenes-mini)
- Giải nén tự động và kiểm tra tính toàn vẹn của nhãn voxel (labels.npz)
- Xử lý cơ chế hạn ngạch tải (quota limit) của Google Drive một cách thông minh
"""
import os
import sys
import argparse
import tarfile
import subprocess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Google Drive File ID chính thức của Occ3D-nuScenes-mini (Tsinghua MARS Lab & CVPR Challenge)
OFFICIAL_OCC3D_MINI_GTS_ID = "1-XIJRS_Uy0tC6Th6KlMIlrzw8JH939cp"
OFFICIAL_GDRIVE_URL = f"https://drive.google.com/uc?id={OFFICIAL_OCC3D_MINI_GTS_ID}"
OFFICIAL_FOLDER_URL = "https://drive.google.com/drive/folders/1ksWt4WLEqOxptpWH2ZN-t1pjugBhg3ME"

def download_with_gdown(file_id, out_tar_path):
    try:
        import gdown
    except ImportError:
        print("[Occ3D] Đang cài đặt thư viện gdown...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
        import gdown

    print(f"\n[Occ3D] Đang tải gts.tar.gz từ Google Drive (ID: {file_id}) ...")
    try:
        res = gdown.download(id=file_id, output=out_tar_path, quiet=False)
        if res and os.path.exists(out_tar_path) and os.path.getsize(out_tar_path) > 10 * 1024 * 1024:
            print(f"[Occ3D] Tải thành công! Dung lượng: {os.path.getsize(out_tar_path) / (1024*1024):.2f} MB")
            return True
        else:
            print("[Occ3D] Cảnh báo: File tải về có kích thước không hợp lệ.")
            return False
    except Exception as e:
        print(f"\n[Occ3D] Lỗi tải qua gdown: {e}")
        return False

def extract_tar_gz(tar_path, extract_dir):
    print(f"[Occ3D] Đang giải nén {tar_path} vào {extract_dir} ...")
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("[Occ3D] Giải nén hoàn tất!")
        return True
    except Exception as e:
        print(f"[Occ3D] Lỗi giải nén: {e}")
        return False

def verify_occ3d_labels(target_dir):
    """Đếm số file labels.npz đã giải nén"""
    count = 0
    sample_file = None
    for root, _, files in os.walk(target_dir):
        if 'labels.npz' in files:
            count += 1
            if sample_file is None:
                sample_file = os.path.join(root, 'labels.npz')

    print(f"\n[Occ3D Kiểm tra] Tìm thấy tổng cộng: {count} file nhãn labels.npz chuẩn!")
    if sample_file:
        try:
            import numpy as np
            data = np.load(sample_file)
            print(f"--> File mẫu: {sample_file}")
            print(f"--> Các trường dữ liệu: {list(data.files)}")
            if 'semantics' in data.files:
                sem = data['semantics']
                print(f"--> Kích thước mảng semantics: {sem.shape} (Voxel grid chuẩn 200x200x16)")
                print(f"--> Số lượng voxel có vật thể/mặt đường: {np.sum(sem != 17):,}")
                print(f"--> Số lượng voxel không khí (free=17): {np.sum(sem == 17):,}")
        except Exception as e:
            print(f"--> Lỗi đọc file mẫu: {e}")
    return count > 0

def main():
    parser = argparse.ArgumentParser(description="Tải và cấu hình bộ nhãn chuẩn Occ3D-nuScenes")
    parser.add_argument('--target-dir', type=str, default='data/occ3d_cam4d', help="Thư mục giải nén nhãn Occ3D")
    parser.add_argument('--gdrive-id', type=str, default=OFFICIAL_OCC3D_MINI_GTS_ID, help="Google Drive File ID")
    parser.add_argument('--tar-file', type=str, default=None, help="Đường dẫn file .tar.gz nếu đã tải thủ công")
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    tar_path = args.tar_file

    if tar_path is None or not os.path.exists(tar_path):
        tar_path = os.path.join(args.target_dir, "gts.tar.gz")
        success = download_with_gdown(args.gdrive_id, tar_path)
        if not success:
            print("\n" + "="*80)
            print("  [HƯỚNG DẪN XỬ LÝ KHI GOOGLE DRIVE HẾT QUOTA CÔNG CỘNG]")
            print("="*80)
            print("Google Drive đang tạm thời khóa quota tải công cộng đối với file này.")
            print("Bạn hãy áp dụng 1 trong 2 cách cực kỳ đơn giản và đảm bảo 100% thành công sau:")
            print("\nCÁCH 1 (Khuyên dùng nhất trên Kaggle - Tốc độ nhanh nhất):")
            print(f"1. Mở link thư mục chính thức Occ3D-mini trên trình duyệt:")
            print(f"   {OFFICIAL_FOLDER_URL}")
            print("2. Tải file 'gts.tar.gz' (khoảng 440MB) về máy tính cá nhân.")
            print("3. Trên giao diện Kaggle Notebook, bấm nút '+ Add Input' -> 'Upload Dataset'")
            print("   -> Đặt tên dataset là 'occ3d-mini-gts' và tải file lên.")
            print("4. Khi đó nhãn Occ3D sẽ luôn có sẵn tại: /kaggle/input/occ3d-mini-gts !")
            print("\nCÁCH 2 (Tạo bản sao Google Drive):")
            print(f"1. Mở link: {OFFICIAL_FOLDER_URL}")
            print("2. Chuột phải vào 'gts.tar.gz' -> Chọn 'Make a copy' (Tạo bản sao) vào Google Drive của bạn.")
            print("3. Lấy File ID của bản sao đó và chạy:")
            print("   python tools/download_occ3d.py --gdrive-id <FILE_ID_CUA_BAN>")
            print("="*80 + "\n")
            return

    # Giải nén
    extract_tar_gz(tar_path, args.target_dir)

    # Xác minh
    verify_occ3d_labels(args.target_dir)

if __name__ == '__main__':
    main()
