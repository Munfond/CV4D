"""
Builds a standalone offline HTML dashboard located at:
model_output/visualization/web_dashboard.html
Includes inline CSS and fallback embedded JSON data so it works via file:///
without needing any web server or network access.
"""

import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
VIS_DIR = BASE_DIR / "model_output" / "visualization"
EVAL_DIR = BASE_DIR / "model_output" / "evaluation"
INST_DIR = BASE_DIR / "model_output" / "instance"
WEB_DIR = BASE_DIR / "web"

css = (WEB_DIR / "static" / "css" / "styles.css").read_text(encoding="utf-8")
html = (WEB_DIR / "templates" / "index.html").read_text(encoding="utf-8")
js = (WEB_DIR / "static" / "js" / "app.js").read_text(encoding="utf-8")

# Read metrics
metrics_data = {}
if (EVAL_DIR / "metrics.json").exists():
    metrics_data = json.loads((EVAL_DIR / "metrics.json").read_text(encoding="utf-8"))

# Read detections
detections_list = []
if (INST_DIR / "bounding_box.json").exists():
    boxes_data = json.loads((INST_DIR / "bounding_box.json").read_text(encoding="utf-8"))
    first_frame = next(iter(boxes_data.values()), [])
    for obj in first_frame:
        cx, cy, cz = obj.get("centroid", [0, 0, 0])
        dist = (cx**2 + cy**2)**0.5
        vx, vy, vz = obj.get("velocity", [0, 0, 0])
        speed = (vx**2 + vy**2)**0.5
        threat = "Low"
        if dist < 15.0 and speed > 0.1:
            threat = "High"
        elif dist < 30.0:
            threat = "Medium"
        detections_list.append({
            "id": obj.get("instance_id", "unknown"),
            "class": obj.get("class", "unknown"),
            "class_id": obj.get("class_id", 0),
            "distance_m": round(dist, 2),
            "speed_mps": round(speed, 2),
            "speed_kmh": round(speed * 3.6, 1),
            "position": [round(cx, 2), round(cy, 2), round(cz, 2)],
            "dimensions_lwh": [round(x, 2) for x in obj.get("dimensions_lwh", [0, 0, 0])],
            "threat_level": threat
        })
    detections_list.sort(key=lambda x: x["distance_m"])

# Read per-class metrics
per_class_list = []
if (EVAL_DIR / "per_class_metrics.csv").exists():
    with open(EVAL_DIR / "per_class_metrics.csv", mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            per_class_list.append({
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

# Inline fallback data into JS
fallback_script = f"""
// Fallback data for standalone file:/// mode
window.FALLBACK_DATA = {{
    summary: {{
        metrics: {json.dumps(metrics_data)},
        total_detections: {len(detections_list)},
        metadata: {{
            scene_id: "scene-0061",
            dataset: "nuScenes v1.0-mini",
            spatial_grid: "200x200x16 (0.4m)"
        }}
    }},
    detections: {json.dumps(detections_list)},
    per_class: {json.dumps(per_class_list)}
}};

// Patch fetch for standalone file:/// mode
const origFetch = window.fetch;
window.fetch = async function(url, options) {{
    try {{
        if (location.protocol === 'file:') {{
            if (url === '/api/summary') return {{ ok: true, json: async () => window.FALLBACK_DATA.summary }};
            if (url === '/api/detections') return {{ ok: true, json: async () => ({{ status: 'success', detections: window.FALLBACK_DATA.detections }}) }};
            if (url === '/api/metrics') return {{ ok: true, json: async () => ({{ status: 'success', per_class: window.FALLBACK_DATA.per_class }}) }};
            if (url === '/api/artifacts') return {{ ok: false }};
        }}
        return await origFetch(url, options);
    }} catch(e) {{
        if (url === '/api/summary') return {{ ok: true, json: async () => window.FALLBACK_DATA.summary }};
        if (url === '/api/detections') return {{ ok: true, json: async () => ({{ status: 'success', detections: window.FALLBACK_DATA.detections }}) }};
        if (url === '/api/metrics') return {{ ok: true, json: async () => ({{ status: 'success', per_class: window.FALLBACK_DATA.per_class }}) }};
        return {{ ok: false }};
    }}
}};
"""

# Replace in HTML
html = html.replace('<link rel="stylesheet" href="/static/css/styles.css">', f"<style>\n{css}\n</style>")
html = html.replace('/files/visualization/', '')
html = html.replace('/download/visualization/', '')
html = html.replace('/files/evaluation/confusion_matrix.png', '../evaluation/confusion_matrix.png')
html = html.replace('/download/evaluation/per_class_metrics.csv', '../evaluation/per_class_metrics.csv')

# Replace script
bundled_js = fallback_script + "\n" + js
html = html.replace('<script src="/static/js/app.js"></script>', f"<script>\n{bundled_js}\n</script>")

out_file = VIS_DIR / "web_dashboard.html"
out_file.write_text(html, encoding="utf-8")
print(f"[OK] Standalone HTML generated at: {out_file}")

# Also copy to brain artifact dir
brain_dir = Path(r"C:\Users\nguye\.gemini\antigravity\brain\fb4af746-fefc-4c0b-b904-44f39d39aa9f")
if brain_dir.exists():
    (brain_dir / "web_dashboard.html").write_text(html, encoding="utf-8")
    print(f"[OK] Copied to brain artifacts at: {brain_dir / 'web_dashboard.html'}")
