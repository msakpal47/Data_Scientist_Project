let currentOffset = 0;
let currentLimit = 100;
let currentCluster = "";
let currentSearch = "";
let clusterChart = null;
let SEGMENTS = {
    0: "Low Value",
    1: "Medium Buyers",
    2: "High Value",
    3: "At Risk",
    4: "Churn Risk"
};

async function predict() {
    const status = document.getElementById("status");
    status.textContent = "Running clustering...";
    try {
        if (window.location.protocol !== "http:" && window.location.protocol !== "https:") {
            status.textContent = "Open the app via the URL shown in the server logs.";
            return;
        }
        const limitEl = document.getElementById("limitInput");
        const clusterEl = document.getElementById("clusterFilter");
        const searchEl = document.getElementById("searchInput");
        currentLimit = Number(limitEl?.value || 100);
        currentCluster = clusterEl?.value || "";
        currentSearch = searchEl?.value ? String(searchEl.value).trim() : "";
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 12000);
        const params = new URLSearchParams();
        params.set("limit", String(currentLimit));
        params.set("offset", String(currentOffset));
        if (currentCluster !== "") params.set("cluster", currentCluster);
        if (currentSearch !== "") params.set("user_id", currentSearch);
        const res = await fetch(`/api/predict?${params.toString()}`, {
            headers: {
                "Accept": "application/json"
            },
            signal: controller.signal
        });
        clearTimeout(timeout);
        if (!res.ok) {
            status.textContent = `API error: ${res.status}`;
            return;
        }
        const data = await res.json();

        if (data.error) {
            status.textContent = "Model not trained. Run: python train.py";
            return;
        }

        const tbody = document.querySelector("#resultTable tbody");
        let html = "";
        const displayUsers = data.users;
        const displayClusters = data.clusters;

        displayUsers.forEach((u, i) => {
            const c = displayClusters[i];
            html += `
            <tr>
                <td>${u}</td>
                <td>${c}</td>
                <td>${SEGMENTS[Number(c)] ?? "-"}</td>
            </tr>`;
        });
        tbody.innerHTML = html;
        const fdesc = currentCluster !== "" ? ` | Filter: C${currentCluster}` : "";
        const sdesc = currentSearch !== "" ? ` | Search: ${currentSearch}` : "";
        status.textContent = `Loaded ${displayUsers.length} of ${data.total} users${fdesc}${sdesc}.`;

        updateSummaryAndChart(data);
        ensureRowsVisible(displayUsers.length);
    } catch (e) {
        status.textContent = "Error calling API. Ensure server running and open http://localhost:8000/templates/index.html";
    }
}

async function checkHealth() {
    const status = document.getElementById("status");
    try {
        if (window.location.protocol !== "http:" && window.location.protocol !== "https:") {
            status.textContent = "Open the app via the URL shown in the server logs.";
            return;
        }
        await updateSegmentsFromServer();
        const res = await fetch("/api/health", { headers: { "Accept": "application/json" } });
        if (!res.ok) {
            status.textContent = "Server not responding.";
            return;
        }
        const data = await res.json();
        status.textContent = data.model_loaded ? "Server OK, model loaded." : "Server OK, model missing. Run: python train.py";
    } catch {
        status.textContent = "Server not running. Start: python server.py";
    }
}

window.addEventListener("load", checkHealth);

function updateSummaryAndChart(data) {
    const countsRaw = (currentCluster !== "" || currentSearch !== "")
        ? (data.counts_filtered ?? data.counts)
        : data.counts;
    const counts = Array.isArray(countsRaw)
        ? countsRaw
        : [
            Number((countsRaw && (countsRaw["0"] ?? countsRaw[0])) || 0),
            Number((countsRaw && (countsRaw["1"] ?? countsRaw[1])) || 0),
            Number((countsRaw && (countsRaw["2"] ?? countsRaw[2])) || 0),
            Number((countsRaw && (countsRaw["3"] ?? countsRaw[3])) || 0),
            Number((countsRaw && (countsRaw["4"] ?? countsRaw[4])) || 0),
        ];
    const sumLocal = counts.reduce((a,b)=>a+Number(b||0),0);
    if (sumLocal > 0) {
        applyCountsToUI(counts, sumLocal);
        return;
    }
    // Fallback: fetch global counts if local counts missing or zero
    fetchGlobalCounts().then((globalArr) => {
        let arr = globalArr;
        if (currentCluster !== "") {
            const clusterId = Number(currentCluster);
            const filteredArr = [0, 0, 0, 0, 0];
            filteredArr[clusterId] = arr[clusterId] || 0;
            arr = filteredArr;
        }
        applyCountsToUI(arr);
    });
}


async function fetchGlobalCounts() {
    try {
        const res = await fetch("/api/cluster_counts", { headers: { "Accept": "application/json" } });
        if (!res.ok) return;
        const d = await res.json();
        const arr = Array.isArray(d.counts) ? d.counts : [
            Number((d.counts && (d.counts["0"] ?? d.counts[0])) || 0),
            Number((d.counts && (d.counts["1"] ?? d.counts[1])) || 0),
            Number((d.counts && (d.counts["2"] ?? d.counts[2])) || 0),
            Number((d.counts && (d.counts["3"] ?? d.counts[3])) || 0),
            Number((d.counts && (d.counts["4"] ?? d.counts[4])) || 0),
        ];
        if (d.segments && Object.keys(d.segments).length > 0) {
            SEGMENTS = d.segments;
        }
        populateSegmentOptions(SEGMENTS);
        return arr;
    } catch {}
    return [0,0,0,0,0];
}

async function updateSegmentsFromServer() {
    try {
        const res = await fetch("/api/segments", { headers: { "Accept": "application/json" } });
        if (!res.ok) return;
        const d = await res.json();
        if (d.segments && Object.keys(d.segments).length > 0) {
            SEGMENTS = d.segments;
        }
        populateSegmentOptions(SEGMENTS);
    } catch {}
}

function populateSegmentOptions(map) {
    const select = document.getElementById("clusterFilter");
    if (!select) return;
    const all = document.createElement("option");
    all.value = "";
    all.textContent = "All Segments";
    const entries = Object.entries(map).sort((a, b) => Number(a[0]) - Number(b[0]));
    select.innerHTML = "";
    select.appendChild(all);
    entries.forEach(([id, label]) => {
        const opt = document.createElement("option");
        opt.value = String(id);
        opt.textContent = label;
        select.appendChild(opt);
    });
}

function applyCountsToUI(counts, total, cards) {
    const c = counts || [0,0,0,0,0];
    const cardHigh = cards?.cardHigh ?? document.getElementById("card-high");
    const cardLow = cards?.cardLow ?? document.getElementById("card-low");
    const cardMedium = cards?.cardMedium ?? document.getElementById("card-medium");
    const cardRisk = cards?.cardRisk ?? document.getElementById("card-risk");
    const cardChurn = cards?.cardChurn ?? document.getElementById("card-churn");
    const cardTotal = cards?.cardTotal ?? document.getElementById("card-total");
    const sum = c.reduce((a,b)=>a+Number(b||0),0);
    cardTotal.textContent = sum;
    cardHigh.textContent = c[2] || 0;
    cardLow.textContent = c[0] || 0;
    cardMedium.textContent = c[1] || 0;
    cardRisk.textContent = c[3] || 0;
    cardChurn.textContent = c[4] || 0;
}

document.getElementById("clusterFilter").addEventListener("change", () => {
    currentCluster = document.getElementById("clusterFilter").value;
    currentOffset = 0;
    predict();
});

document.getElementById("searchBtn").addEventListener("click", () => {
    const v = document.getElementById("searchInput").value;
    currentSearch = v ? String(v) : "";
    currentOffset = 0;
    predict();
});

document.getElementById("nextBtn").addEventListener("click", () => {
    currentOffset += currentLimit;
    predict();
});

document.getElementById("prevBtn").addEventListener("click", () => {
    currentOffset = Math.max(0, currentOffset - currentLimit);
    predict();
});

function ensureRowsVisible(usersLen) {
    const tbody = document.querySelector("#resultTable tbody");
    if (usersLen === 0) {
        tbody.innerHTML = `<tr><td colspan="3">No results for current filter/search.</td></tr>`;
    }
}
