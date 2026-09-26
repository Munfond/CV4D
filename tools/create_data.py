"""
Script Tiền xử lý Dữ liệu: Pre-compile metadata nuScenes thành file cache .pkl
Sử dụng Pure Python JSON Parser: KHÔNG CẦN nuScenes-devkit, KHÔNG CẦN sklearn/scipy.
Chạy được trên mọi phiên bản Python / NumPy mà không bao giờ bị lỗi dependency!
"""
import os
import sys
import json
import argparse
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def find_json_dir(data_root, version='v1.0-mini'):
    """Tìm thư mục chứa các file json của nuScenes"""
    candidate_paths = [
        os.path.join(data_root, version),
        data_root,
        os.path.join(data_root, 'mini-nuscenes', version),
        os.path.join(data_root, 'nuscenes-v1.0-mini', version),
        os.path.join(data_root, 'v1.0-mini')
    ]
    for p in candidate_paths:
        if os.path.exists(os.path.join(p, 'sample.json')):
            return p
    return None

def parse_nuscenes_pure_json(json_dir):
    """Đọc trực tiếp các bảng JSON nuScenes bằng thư viện json có sẵn của Python"""
    print(f"[PureJSON] Đang đọc các bảng JSON từ: {json_dir} ...")
    
    with open(os.path.join(json_dir, 'sample.json'), 'r') as f:
        samples = json.load(f)
    with open(os.path.join(json_dir, 'sample_data.json'), 'r') as f:
        sample_data = json.load(f)
    with open(os.path.join(json_dir, 'calibrated_sensor.json'), 'r') as f:
        calibrated_sensors = json.load(f)
    with open(os.path.join(json_dir, 'ego_pose.json'), 'r') as f:
        ego_poses = json.load(f)

    # Đọc thêm scene.json để ánh xạ sang tên scene (ví dụ: scene-0061) cho bộ nhãn Occ3D
    scene_name_map = {}
    scene_file = os.path.join(json_dir, 'scene.json')
    if os.path.exists(scene_file):
        with open(scene_file, 'r') as f:
            scenes = json.load(f)
            scene_name_map = {item['token']: item['name'] for item in scenes}

    # Đánh chỉ mục dictionary bằng token để tra cứu O(1)
    sd_map = {item['token']: item for item in sample_data}
    calib_map = {item['token']: item for item in calibrated_sensors}
    ego_map = {item['token']: item for item in ego_poses}

    # Xây dựng bảng ánh xạ foreign key: sample_token -> {channel: sample_data}
    sample_to_sd = {}
    for sd in sample_data:
        s_tok = sd.get('sample_token')
        chan = sd.get('channel')
        if s_tok and chan:
            if s_tok not in sample_to_sd:
                sample_to_sd[s_tok] = {}
            if sd.get('is_key_frame', True) or chan not in sample_to_sd[s_tok]:
                sample_to_sd[s_tok][chan] = sd

    infos = []
    cam_types = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    for sample in samples:
        s_tok = sample['token']
        # Hỗ trợ cả raw nuScenes JSON (sample không có trường 'data') lẫn devkit JSON
        if 'data' in sample and isinstance(sample['data'], dict):
            sensor_map = {chan: sd_map[tok] for chan, tok in sample['data'].items() if tok in sd_map}
        else:
            sensor_map = sample_to_sd.get(s_tok, {})

        cams = {}
        for cam in cam_types:
            if cam in sensor_map:
                sd = sensor_map[cam]
                calib = calib_map.get(sd.get('calibrated_sensor_token', ''), {})
                cams[cam] = {
                    'data_path': sd.get('filename', ''),
                    'sensor2ego_translation': calib.get('translation', [0, 0, 0]),
                    'sensor2ego_rotation': calib.get('rotation', [1, 0, 0, 0]),
                    'cam_intrinsic': calib.get('camera_intrinsic', None)
                }
            else:
                cams[cam] = {
                    'data_path': '',
                    'sensor2ego_translation': [0, 0, 0],
                    'sensor2ego_rotation': [1, 0, 0, 0],
                    'cam_intrinsic': None
                }

        # LiDAR
        if 'LIDAR_TOP' in sensor_map:
            lidar_sd = sensor_map['LIDAR_TOP']
            lidar_calib = calib_map.get(lidar_sd.get('calibrated_sensor_token', ''), {})
            lidar_info = {
                'data_path': lidar_sd.get('filename', ''),
                'sensor2ego_translation': lidar_calib.get('translation', [0, 0, 0]),
                'sensor2ego_rotation': lidar_calib.get('rotation', [1, 0, 0, 0]),
            }
            ego_pose = ego_map.get(lidar_sd.get('ego_pose_token', ''), {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]})
        else:
            lidar_info = {'data_path': '', 'sensor2ego_translation': [0, 0, 0], 'sensor2ego_rotation': [1, 0, 0, 0]}
            ego_pose = {'translation': [0, 0, 0], 'rotation': [1, 0, 0, 0]}

        # Radar (Lấy Radar Front nếu có)
        radar_chans = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        radar_sd = next((sensor_map[r] for r in radar_chans if r in sensor_map), None)
        if radar_sd is not None:
            radar_calib = calib_map.get(radar_sd.get('calibrated_sensor_token', ''), {})
            radar_info = {
                'data_path': radar_sd.get('filename', ''),
                'sensor2ego_translation': radar_calib.get('translation', [0, 0, 0]),
                'sensor2ego_rotation': radar_calib.get('rotation', [1, 0, 0, 0]),
            }
        else:
            radar_info = {'data_path': '', 'sensor2ego_translation': [0, 0, 0], 'sensor2ego_rotation': [1, 0, 0, 0]}

        info = {
            'token': sample['token'],
            'timestamp': sample['timestamp'],
            'scene_token': sample.get('scene_token', ''),
            'scene_name': scene_name_map.get(sample.get('scene_token', ''), ''),
            'ego2global_translation': ego_pose.get('translation', [0, 0, 0]),
            'ego2global_rotation': ego_pose.get('rotation', [1, 0, 0, 0]),
            'cams': cams,
            'lidar': lidar_info,
            'radar': radar_info
        }
        infos.append(info)

    return infos

def main():
    parser = argparse.ArgumentParser(description="Tạo file cache metadata cho nuScenes (Pure Python)")
    parser.add_argument('--data-root', type=str, default='/kaggle/input/mini-nuscenes', help="Thư mục gốc nuScenes")
    parser.add_argument('--version', type=str, default='v1.0-mini', help="Phiên bản nuScenes")
    parser.add_argument('--out-path', type=str, default='data/cache/nuscenes_infos_val.pkl', help="File output .pkl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    json_dir = find_json_dir(args.data_root, args.version)

    if json_dir is not None:
        try:
            infos = parse_nuscenes_pure_json(json_dir)
            with open(args.out_path, 'wb') as f:
                pickle.dump(infos, f)
            print(f"\n[CreateData] THÀNH CÔNG RỰC RỠ! Đã nạp {len(infos)} frames dữ liệu nuScenes thật vào {args.out_path}")
            return
        except Exception as e:
            import traceback
            print(f"[CreateData] Lỗi đọc JSON ({e}), chuyển sang phương án fallback.")
            traceback.print_exc()

    print(f"[CreateData] Không tìm thấy thư mục JSON tại {args.data_root}. Tạo dữ liệu mẫu 20 frames...")
    dummy_infos = [{'token': f'sample_{i}', 'scene_token': 'scene_0'} for i in range(20)]
    with open(args.out_path, 'wb') as f:
        pickle.dump(dummy_infos, f)
    print(f"[CreateData] Đã tạo cache mẫu tại: {args.out_path}")

if __name__ == '__main__':
    main()
