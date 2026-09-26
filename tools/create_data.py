"""
Script Tiền xử lý Dữ liệu: Pre-compile metadata nuScenes thành file cache .pkl
Giúp nạp dữ liệu siêu tốc trên Kaggle mà không bị tràn RAM.
"""
import os
import argparse
import pickle

def main():
    parser = argparse.ArgumentParser(description="Tạo file cache metadata cho nuScenes")
    parser.add_argument('--data-root', type=str, default='data/nuscenes', help="Thư mục gốc nuScenes")
    parser.add_argument('--version', type=str, default='v1.0-mini', help="Phiên bản nuScenes")
    parser.add_argument('--out-path', type=str, default='data/cache/nuscenes_infos_val.pkl', help="File output .pkl")
    args = parser.parse_args()

    print(f"[CreateData] Đang quét thư mục {args.data_root} (phiên bản {args.version})...")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    except Exception as e:
        print(f"[CreateData] Không thể nạp nuScenes-devkit ({e}).")
        print("[CreateData] Tạo file cache giả lập để nhóm có thể code và test luồng...")
        dummy_infos = [{'token': f'sample_{i}', 'scene_token': 'scene_0'} for i in range(20)]
        with open(args.out_path, 'wb') as f:
            pickle.dump(dummy_infos, f)
        print(f"[CreateData] Đã tạo cache giả lập tại: {args.out_path}")
        return

    infos = []
    cam_types = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    for sample in nusc.sample:
        cams = {}
        for cam in cam_types:
            cam_data = nusc.get('sample_data', sample['data'][cam])
            calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            cams[cam] = {
                'data_path': cam_data['filename'],
                'sensor2ego_translation': calib['translation'],
                'sensor2ego_rotation': calib['rotation'],
                'cam_intrinsic': calib['camera_intrinsic']
            }

        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        radar_data = nusc.get('sample_data', sample['data']['RADAR_FRONT'])
        radar_calib = nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])

        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        info = {
            'token': sample['token'],
            'timestamp': sample['timestamp'],
            'scene_token': sample['scene_token'],
            'ego2global_translation': ego_pose['translation'],
            'ego2global_rotation': ego_pose['rotation'],
            'cams': cams,
            'lidar': {
                'data_path': lidar_data['filename'],
                'sensor2ego_translation': lidar_calib['translation'],
                'sensor2ego_rotation': lidar_calib['rotation'],
            },
            'radar': {
                'data_path': radar_data['filename'],
                'sensor2ego_translation': radar_calib['translation'],
                'sensor2ego_rotation': radar_calib['rotation'],
            }
        }
        infos.append(info)

    with open(args.out_path, 'wb') as f:
        pickle.dump(infos, f)
    print(f"\n[CreateData] ĐÃ TẠO THÀNH CÔNG {len(infos)} frames metadata -> {args.out_path}")

if __name__ == '__main__':
    main()
