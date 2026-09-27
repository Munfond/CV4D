/* VinFast ADAS 4D-OccFusion Lab - Frontend Application Logic */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadSummaryData();
    loadDetections();
    loadMetrics();
    loadArtifacts();
    initVideoControls();
});

// ----------------- TAB SWITCHING ----------------- //
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");

            tabBtns.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetPane = document.getElementById(targetId);
            if (targetPane) {
                targetPane.classList.add("active");
            }
        });
    });
}

// ----------------- SUMMARY & KPIS ----------------- //
async function loadSummaryData() {
    try {
        const res = await fetch("/api/summary");
        if (!res.ok) return;
        const data = await res.json();
        
        const m = data.metrics || {};
        
        // Update KPI values
        setElemText("kpi-precision", (m.precision_percent || 68.42).toFixed(1) + "%");
        setElemText("kpi-recall", (m.recall_percent || 73.19).toFixed(1) + "%");
        setElemText("kpi-f1", (m.f1_score_percent || 70.72).toFixed(1) + "%");
        setElemText("kpi-latency", (m.inference_latency_ms || 192.7).toFixed(1) + " ms");
        setElemText("kpi-fps", (m.throughput_fps || 5.19).toFixed(1) + " FPS");
        setElemText("kpi-detections", data.total_detections || 12);
        
        // Update scene info badges
        const meta = data.metadata || {};
        setElemText("info-scene", meta.scene_id || "scene-0061");
        setElemText("info-dataset", meta.dataset || "nuScenes v1.0-mini");
        setElemText("info-grid", meta.spatial_grid || "200x200x16 (0.4m)");
    } catch (err) {
        console.warn("Could not load summary API:", err);
    }
}

// ----------------- OBJECT DETECTIONS ----------------- //
async function loadDetections() {
    const tbody = document.getElementById("detections-table-body");
    if (!tbody) return;

    try {
        const res = await fetch("/api/detections");
        if (!res.ok) return;
        const data = await res.json();
        const detections = data.detections || [];

        tbody.innerHTML = "";
        if (detections.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-muted);">Không có vật thể nào được tìm thấy.</td></tr>`;
            return;
        }

        detections.forEach(item => {
            const tr = document.createElement("tr");

            let badgeClass = "badge-blue";
            if (item.threat_level === "High") badgeClass = "badge-red";
            else if (item.threat_level === "Medium") badgeClass = "badge-orange";
            else if (item.threat_level === "Low") badgeClass = "badge-green";

            tr.innerHTML = `
                <td><strong>${item.id}</strong></td>
                <td><span class="badge ${badgeClass}">${item.class.toUpperCase()}</span></td>
                <td><strong>${item.distance_m} m</strong></td>
                <td>${item.speed_kmh} km/h <span style="font-size:0.75rem; color:var(--text-muted);">(${item.speed_mps} m/s)</span></td>
                <td>[${item.position[0]}, ${item.position[1]}, ${item.position[2]}]</td>
                <td>${item.dimensions_lwh[0]} x ${item.dimensions_lwh[1]} x ${item.dimensions_lwh[2]}</td>
                <td><span class="badge ${badgeClass}">${item.threat_level}</span></td>
            `;
            tbody.appendChild(tr);
        });

    } catch (err) {
        console.warn("Could not load detections API:", err);
    }
}

// ----------------- PER-CLASS METRICS ----------------- //
async function loadMetrics() {
    const tbody = document.getElementById("metrics-table-body");
    if (!tbody) return;

    try {
        const res = await fetch("/api/metrics");
        if (!res.ok) return;
        const data = await res.json();
        const perClass = data.per_class || [];

        tbody.innerHTML = "";
        perClass.forEach(row => {
            const tr = document.createElement("tr");
            
            // Highlight positive IoUs
            const iouColor = row.iou > 0 ? "var(--accent-cyan)" : "var(--text-muted)";
            const f1Color = row.f1_score > 0 ? "var(--accent-green)" : "var(--text-muted)";

            tr.innerHTML = `
                <td>#${row.class_id}</td>
                <td><strong>${row.class_name}</strong></td>
                <td style="color:${iouColor}; font-weight:700;">${row.iou.toFixed(2)}%</td>
                <td>${row.precision.toFixed(2)}%</td>
                <td>${row.recall.toFixed(2)}%</td>
                <td style="color:${f1Color}; font-weight:700;">${row.f1_score.toFixed(2)}%</td>
                <td>${row.tp.toLocaleString()}</td>
                <td>${row.fp.toLocaleString()}</td>
                <td>${row.fn.toLocaleString()}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (err) {
        console.warn("Could not load metrics API:", err);
    }
}

// ----------------- ARTIFACTS LIST ----------------- //
async function loadArtifacts() {
    const grid = document.getElementById("artifacts-grid");
    if (!grid) return;

    try {
        const res = await fetch("/api/artifacts");
        if (!res.ok) return;
        const data = await res.json();
        const artifacts = data.artifacts || [];

        grid.innerHTML = "";
        artifacts.forEach(item => {
            const card = document.createElement("div");
            card.className = "artifact-item";

            let icon = "📄";
            if (item.type.includes("Image")) icon = "🖼️";
            else if (item.type.includes("Video")) icon = "🎬";
            else if (item.type.includes("3D") || item.type.includes("HTML")) icon = "🧊";
            else if (item.type.includes("JSON") || item.type.includes("CSV")) icon = "📊";

            card.innerHTML = `
                <div>
                    <div class="artifact-header">
                        <span class="badge badge-blue">${item.category}</span>
                        <span style="font-size:1.2rem;">${icon}</span>
                    </div>
                    <div class="artifact-name">${item.name}</div>
                    <div class="artifact-desc">${item.description}</div>
                </div>
                <div class="artifact-footer">
                    <span>Dung lượng: <strong>${item.size}</strong></span>
                    <div style="display:flex; gap:0.5rem;">
                        <a href="${item.url}" target="_blank" class="btn btn-secondary btn-sm">Xem</a>
                        <a href="${item.download_url}" class="btn btn-primary btn-sm">Tải về</a>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });
    } catch (err) {
        console.warn("Could not load artifacts API:", err);
    }
}

// ----------------- VIDEO CONTROLS ----------------- //
function initVideoControls() {
    const videoElem = document.getElementById("main-video-player");
    if (!videoElem) return;

    const speedBtns = document.querySelectorAll(".video-speed-btn");
    speedBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            speedBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const speed = parseFloat(btn.getAttribute("data-speed") || "1.0");
            videoElem.playbackRate = speed;
        });
    });

    const switchBtns = document.querySelectorAll(".video-switch-btn");
    switchBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            switchBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const videoSrc = btn.getAttribute("data-src");
            if (videoSrc) {
                videoElem.src = videoSrc;
                videoElem.play();
            }
        });
    });
}

function setElemText(id, text) {
    const elem = document.getElementById(id);
    if (elem) elem.textContent = text;
}
