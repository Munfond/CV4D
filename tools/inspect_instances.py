import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path

d = Path('data/nuscenes/v1.0-mini')
scenes = json.loads((d / 'scene.json').read_text(encoding='utf-8'))
samples = json.loads((d / 'sample.json').read_text(encoding='utf-8'))
anns = json.loads((d / 'sample_annotation.json').read_text(encoding='utf-8'))
insts = json.loads((d / 'instance.json').read_text(encoding='utf-8'))
cats = json.loads((d / 'category.json').read_text(encoding='utf-8'))

cat_m = {c['token']: c['name'] for c in cats}
inst_m = {i['token']: i for i in insts}
s_m = {s['token']: s for s in samples}

s61 = [s for s in scenes if s['name'] == 'scene-0061'][0]
c = s61['first_sample_token']
toks = set()
while c:
    toks.add(c)
    c = s_m[c].get('next', '')

seen = {}
for a in anns:
    if a['sample_token'] in toks:
        it = a['instance_token']
        if it not in seen:
            seen[it] = {'cat': cat_m[inst_m[it]['category_token']], 'count': 0}
        seen[it]['count'] += 1

print(f"Tổng số đối tượng vật lý duy nhất trong scene-0061: {len(seen)}")
for i, (it, info) in enumerate(sorted(seen.items(), key=lambda x: x[1]['count'], reverse=True)[:15]):
    print(f"  ID #{i+1:02d}: {info['cat']:<32} (xuất hiện trong {info['count']:<2} frames)")
