import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path

data_dir = Path('data/nuscenes/v1.0-mini')
scenes = json.loads((data_dir / 'scene.json').read_text(encoding='utf-8'))
samples = json.loads((data_dir / 'sample.json').read_text(encoding='utf-8'))

print(f"Total scenes: {len(scenes)}")
print(f"Total samples (keyframes): {len(samples)}")

sample_map = {s['token']: s for s in samples}
scene_counts = {}
for s in samples:
    st = s['scene_token']
    scene_counts[st] = scene_counts.get(st, 0) + 1

print("\n--- DANH SÁCH TẤT CẢ SCENE VÀ SỐ KEYFRAME LIÊN TIẾP ---")
for sc in sorted(scenes, key=lambda x: scene_counts.get(x['token'], 0), reverse=True):
    cnt = scene_counts.get(sc['token'], 0)
    # Trace sequence from first_sample_token
    curr = sc['first_sample_token']
    seq_len = 0
    while curr:
        seq_len += 1
        curr = sample_map.get(curr, {}).get('next', '')
    print(f"Scene: {sc['name']:<12} | Keyframes (2Hz): {cnt:<3} | Chuỗi liên tiếp (first->last): {seq_len:<3} | Mô tả: {sc['description'][:60]}")
