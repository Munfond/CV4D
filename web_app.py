"""
VinFast ADAS 4D-OccFusion Lab - Local Web Portal Server
Cung cấp web dashboard trực quan hóa toàn bộ artifacts của dự án:
- Master Multi-Modal ADAS Dashboard
- LiDAR BEV Point Cloud & 3D Bounding Boxes (GT vs Pred)
- Real Camera RGB with 3D Wireframe Bounding Box Projections
- Interactive 360° 3D Voxel Occupancy (WebGL)
- 4D Spatio-Temporal Perception Video & Flow Player
- Model Benchmark Metrics & Confusion Matrix
- File Explorer & One-Click Artifact Downloads
"""

import os
import sys
import json
import csv
import argparse
import webbrowser
import threading
import time
from pathlib import Path
from flask import Flask, render_template, jsonify, send_from_directory, request, send_file, abort

# Cấu hình encoding UTF-8 trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"
MODEL_OUTPUT_DIR = BASE_DIR / "model_output"
VIS_DIR = MODEL_OUTPUT_DIR / "visualization"
EVAL_DIR = MODEL_OUTPUT_DIR / "evaluation"
INSTANCE_DIR = MODEL_OUTPUT_DIR / "instance"
METADATA_DIR = MODEL_OUTPUT_DIR / "metadata"
DATA_DIR = BASE_DIR / "data" / "nuscenes"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR)
)

# ----------------- HELPER FUNCTIONS ----------------- #

def load_json_safe(file_path):
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Error reading {file_path}: {e}")
    return {}

def format_size(bytes_num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.1f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.1f} TB"

# ----------------- REST API ROUTES ----------------- #

@app.route('/')
def index():
    """Trang chủ ADAS Web Dashboard"""
    return render_template('index.html')

@app.route('/api/summary')
def api_summary():
    """Tóm tắt thông tin scene, chỉ số mô hình và trạng thái hệ thống"""
    metrics = load_json_safe(EVAL_DIR / "metrics.json")
    
    # Metadata cơ bản
    metadata = {
        "project": "VinFast ADAS 4D-OccFusion Lab",
        "scene_id": "scene-0061",
        "dataset": "nuScenes v1.0-mini (Boston/Singapore)",
        "model_architecture": "4D-OccNet (Swin-T + Deformable Temporal Cross-Attention)",
        "spatial_grid": "200 x 200 x 16 (0.4m voxel)",
        "spatial_range": "X: [-40m, 40m], Y: [-40m, 40m], Z: [-1.0m, 5.4m]",
        "temporal_window": "5 frames (2.0 Hz, delta_t = 0.5s)",
        "ego_vehicle": "VinFast VF8 Prototype Platform"
    }

    # Đếm số lượng vật thể phát hiện
    boxes_data = load_json_safe(INSTANCE_DIR / "bounding_box.json")
    total_detections = 0
    if boxes_data:
        first_frame = next(iter(boxes_data.values()), [])
        total_detections = len(first_frame)

    return jsonify({
        "status": "success",
        "metadata": metadata,
        "metrics": metrics,
        "total_detections": total_detections,
        "artifacts_dir": str(VIS_DIR)
    })

@app.route('/api/detections')
def api_detections():
    """Danh sách vật thể nhận diện với tọa độ, khoảng cách và vận tốc"""
    boxes_data = load_json_safe(INSTANCE_DIR / "bounding_box.json")
    results = []
    
    if boxes_data:
        # Lấy frame đầu tiên hoặc frame chỉ định
        frame_id = request.args.get('frame_id')
        if not frame_id or frame_id not in boxes_data:
            frame_id = next(iter(boxes_data.keys()))
            
        objs = boxes_data.get(frame_id, [])
        for obj in objs:
            cx, cy, cz = obj.get("centroid", [0, 0, 0])
            dist = (cx**2 + cy**2)**0.5
            vx, vy, vz = obj.get("velocity", [0, 0, 0])
            speed = (vx**2 + vy**2)**0.5
            
            # Phân loại độ ưu tiên va chạm
            threat = "Low"
            if dist < 15.0 and speed > 0.1:
                threat = "High"
            elif dist < 30.0:
                threat = "Medium"

            results.append({
                "id": obj.get("instance_id", "unknown"),
                "class": obj.get("class", "unknown"),
                "class_id": obj.get("class_id", 0),
                "distance_m": round(dist, 2),
                "speed_mps": round(speed, 2),
                "speed_kmh": round(speed * 3.6, 1),
                "position": [round(cx, 2), round(cy, 2), round(cz, 2)],
                "dimensions_lwh": [round(x, 2) for x in obj.get("dimensions_lwh", [0, 0, 0])],
                "velocity_xyz": [round(x, 2) for x in obj.get("velocity", [0, 0, 0])],
                "voxel_count": obj.get("voxel_count", 0),
                "threat_level": threat
            })

    # Sắp xếp theo khoảng cách gần nhất
    results.sort(key=lambda x: x["distance_m"])
    return jsonify({
        "status": "success",
        "count": len(results),
        "detections": results
    })

@app.route('/api/metrics')
def api_metrics():
    """Bảng chi tiết chỉ số từng phân lớp (Per-class IoU, Precision, Recall, F1)"""
    csv_path = EVAL_DIR / "per_class_metrics.csv"
    per_class = []
    if csv_path.exists():
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    per_class.append({
                        "class_id": int(row.get("Class_ID", 0)),
                        "class_name": row.get("Class_Name", ""),
                        "iou": float(row.get("IoU_Percent", 0)),
                        "precision": float(row.get("Precision", 0)),
                        "recall": float(row.get("Recall", 0)),
                        "f1_score": float(row.get("F1_Score", 0)),
                        "tp": int(row.get("TP", 0)),
                        "fp": int(row.get("FP", 0)),
                        "fn": int(row.get("FN", 0)),
                    })
        except Exception as e:
            print(f"[WARN] Error reading {csv_path}: {e}")
            
    return jsonify({
        "status": "success",
        "per_class": per_class
    })

@app.route('/api/artifacts')
def api_artifacts():
    """Liệt kê toàn bộ files kết quả trực quan hóa và model output để tải về"""
    items = []
    
    # Danh mục file trọng tâm
    featured = [
        {"name": "bev_lidar_groundtruth_pred.png", "folder": VIS_DIR, "type": "Image", "category": "LiDAR BEV", "desc": "BEV LiDAR Point Cloud vs 3D Boxes (Ground Truth vs Model Prediction)"},
        {"name": "camera_front_3d_boxes.png", "folder": VIS_DIR, "type": "Image", "category": "Camera", "desc": "RGB Camera Front với 3D Wireframe Bounding Box chiếu hình học"},
        {"name": "full_scene_multimodal_video.mp4", "folder": VIS_DIR, "type": "Video", "category": "Full Scene 39F", "desc": "Video toàn cảnh 39 frames liên tiếp tối đa (Master Dashboard, 4 FPS)"},
        {"name": "full_scene_camera_video.mp4", "folder": VIS_DIR, "type": "Video", "category": "Full Scene 39F", "desc": "Video Camera trước 39 frames liên tiếp với 3D Wireframe Boxes (720p HD)"},
        {"name": "full_scene_bev_lidar_video.mp4", "folder": VIS_DIR, "type": "Video", "category": "Full Scene 39F", "desc": "Video BEV LiDAR Point Cloud 39 frames liên tiếp kèm hộp theo dõi"},
        {"name": "multimodal_perception_video.mp4", "folder": VIS_DIR, "type": "Video", "category": "Video 4D", "desc": "Video 4D đồng bộ đa cảm biến theo thời gian (Master sequence)"},
        {"name": "multimodal_perception_animation.gif", "folder": VIS_DIR, "type": "Animation", "category": "Animation", "desc": "Ảnh động GIF chuỗi nhận diện thời gian thực"},
        {"name": "interactive_3d_occupancy.html", "folder": VIS_DIR, "type": "HTML/WebGL", "category": "Interactive 3D", "desc": "Mô hình 3D Voxel xoay 360 độ tương tác Plotly WebGL"},
        {"name": "flow_4d_video.mp4", "folder": VIS_DIR, "type": "Video", "category": "Video 4D", "desc": "Video trường chuyển động vector 4D Occupancy Flow"},
        {"name": "flow_4d_animation.gif", "folder": VIS_DIR, "type": "Animation", "category": "Animation", "desc": "GIF 4D Flow trường chuyển động các đối tượng động"},
        {"name": "occupancy_3d_isometric.png", "folder": VIS_DIR, "type": "Image", "category": "Isometric 3D", "desc": "Ảnh phối cảnh 3D Isometric độ phân giải cao"},
        {"name": "occupancy_3d.ply", "folder": VIS_DIR, "type": "3D Point Cloud", "category": "3D Mesh", "desc": "Dữ liệu đám mây điểm 3D dạng PLY cho MeshLab/CloudCompare"},
        {"name": "confusion_matrix.png", "folder": EVAL_DIR, "type": "Image", "category": "Evaluation", "desc": "Ma trận nhầm lẫn (Confusion Matrix) 18 phân lớp semantic"},
        {"name": "metrics.json", "folder": EVAL_DIR, "type": "JSON", "category": "Evaluation", "desc": "Các chỉ số tổng hợp chuẩn CVPR 2023 Occupancy Challenge"},
        {"name": "per_class_metrics.csv", "folder": EVAL_DIR, "type": "CSV", "category": "Evaluation", "desc": "Bảng chi tiết IoU, Precision, Recall, F1 theo từng class"},
        {"name": "bounding_box.json", "folder": INSTANCE_DIR, "type": "JSON", "category": "Detections", "desc": "Tọa độ 3D Bounding Boxes các đối tượng quanh xe"},
    ]
    
    for item in featured:
        file_path = item["folder"] / item["name"]
        if file_path.exists():
            stat = file_path.stat()
            rel_folder = "visualization" if item["folder"] == VIS_DIR else ("evaluation" if item["folder"] == EVAL_DIR else "instance")
            items.append({
                "name": item["name"],
                "category": item["category"],
                "type": item["type"],
                "description": item["desc"],
                "size": format_size(stat.st_size),
                "bytes": stat.st_size,
                "url": f"/files/{rel_folder}/{item['name']}",
                "download_url": f"/download/{rel_folder}/{item['name']}"
            })
            
    return jsonify({
        "status": "success",
        "count": len(items),
        "artifacts": items
    })

# ----------------- STATIC ASSET SERVERS ----------------- #

@app.route('/files/<folder>/<filename>')
def serve_file(folder, filename):
    """Phục vụ file tĩnh từ thư mục model_output"""
    target_dir = MODEL_OUTPUT_DIR / folder
    if not target_dir.exists() or not (target_dir / filename).exists():
        abort(404)
    return send_from_directory(str(target_dir), filename)

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    """Tải file đính kèm trực tiếp"""
    target_dir = MODEL_OUTPUT_DIR / folder
    if not target_dir.exists() or not (target_dir / filename).exists():
        abort(404)
    return send_from_directory(str(target_dir), filename, as_attachment=True)

@app.route('/sample_cam/<camera_name>/<filename>')
def serve_sample_cam(camera_name, filename):
    """Phục vụ ảnh camera nguyên bản từ nuScenes dataset"""
    target_dir = DATA_DIR / "samples" / camera_name
    if not target_dir.exists() or not (target_dir / filename).exists():
        abort(404)
    return send_from_directory(str(target_dir), filename)

# ----------------- RUNNER ----------------- #

def open_browser_delayed(url, delay=1.2):
    time.sleep(delay)
    print(f"\n[INFO] Đang tự động mở trình duyệt web: {url}")
    webbrowser.open(url)

def main():
    parser = argparse.ArgumentParser(description="VinFast ADAS 4D-OccFusion Local Web Portal")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP (mặc định: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Cổng chạy server (mặc định: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Không tự động mở trình duyệt")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 70)
    print(" 🚗 VINFAST ADAS 4D-OCCFUSION LAB - LOCAL WEB DASHBOARD")
    print("=" * 70)
    print(f" [URL]   {url}")
    print(f" [Path]  {BASE_DIR}")
    print(f" [Files] {VIS_DIR}")
    print("=" * 70)
    print(" Nhấn Ctrl+C để dừng web server.\n")

    if not args.no_browser:
        threading.Thread(target=open_browser_delayed, args=(url,), daemon=True).start()

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
