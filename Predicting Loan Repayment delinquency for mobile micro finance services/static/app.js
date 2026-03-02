const jsonInput = document.getElementById("jsonInput");
const resultEl = document.getElementById("result");
const liveOutputEl = document.getElementById("liveOutput");

const predictBtn = document.getElementById("predictBtn");
const predictLiveBtn = document.getElementById("predictLiveBtn");
const loadExampleBtn = document.getElementById("loadExampleBtn");
const loadSchemaBtn = document.getElementById("loadSchemaBtn");
const filterEligibleBtn = document.getElementById("filterEligibleBtn");
const filterNotEligibleBtn = document.getElementById("filterNotEligibleBtn");
let lastRows = [];

function riskBand(p) {
  if (p >= 0.8) return "High Eligible";
  if (p >= 0.65) return "Medium Eligible";
  if (p >= 0.45) return "Borderline";
  return "Not Eligible";
}

function setBusy(isBusy) {
  predictBtn.disabled = isBusy;
  predictLiveBtn.disabled = isBusy;
  loadExampleBtn.disabled = isBusy;
  loadSchemaBtn.disabled = isBusy;
}

function showResult(text) {
  resultEl.textContent = text;
}

function renderLiveTable(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    liveOutputEl.textContent = "No customers found.";
    return;
  }
  const ths = ["msisdn", "pcircle", "aon", "daily_decr30", "cnt_ma_rech30", "cnt_loans30", "probability", "band", "prediction"];
  let html = "<div style='overflow:auto'><table style='width:100%; border-collapse:collapse'><thead><tr>";
  for (const h of ths) {
    html += "<th style='text-align:left; padding:6px; border-bottom:1px solid rgba(255,255,255,0.12)'>" + h + "</th>";
  }
  html += "</tr></thead><tbody>";
  for (const row of rows) {
    html += "<tr>";
    for (const h of ths) {
      let v = row[h];
      if (h === "band") {
        const p = typeof row["probability"] === "number" ? row["probability"] : 0;
        v = riskBand(p);
      }
      if (h === "probability" && typeof v === "number") v = v.toFixed(4);
      if (h === "prediction") v = v ? "Eligible" : "Not eligible";
      let color = "";
      if (h === "band") {
        if (v === "High Eligible") color = "color:#2ecc71"; // green
        else if (v === "Medium Eligible") color = "color:#f1c40f"; // yellow
        else if (v === "Borderline") color = "color:#e67e22"; // orange
        else if (v === "Not Eligible") color = "color:#e74c3c"; // red
      }
      html += "<td style='padding:6px; border-bottom:1px solid rgba(255,255,255,0.06);" + color + "'>" + (v ?? "") + "</td>";
    }
    html += "</tr>";
  }
  html += "</tbody></table></div>";
  liveOutputEl.innerHTML = html;
}

async function apiFetch(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const message = data && data.error ? data.error : `HTTP ${res.status}`;
    throw new Error(message);
  }
  return data;
}

loadSchemaBtn.addEventListener("click", async () => {
  setBusy(true);
  showResult("");
  try {
    const schema = await apiFetch("/api/schema");
    const fields = schema.expected_fields || [];
    showResult(`Expected fields loaded: ${fields.length}`);
    renderLiveTable([]);
  } catch (e) {
    showResult(`Error: ${e.message}`);
  } finally {
    setBusy(false);
  }
});

loadExampleBtn.addEventListener("click", async () => {
  setBusy(true);
  showResult("");
  try {
    const data = await apiFetch("/api/predict-live?n=1");
    const row = (data.rows && data.rows[0]) || null;
    if (!row) {
      showResult("No rows returned from live sample.");
      return;
    }
    const { prediction, probability, ...features } = row;
    jsonInput.value = JSON.stringify(features, null, 2);
    showResult("Example loaded into input box.");
  } catch (e) {
    showResult(`Error: ${e.message}`);
  } finally {
    setBusy(false);
  }
});

predictBtn.addEventListener("click", async () => {
  setBusy(true);
  showResult("");
  try {
    const text = jsonInput.value.trim();
    if (!text) {
      showResult("Paste JSON first.");
      return;
    }
    const payload = JSON.parse(text);
    const data = await apiFetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const eligible = data.prediction ? "Eligible" : "Not eligible";
    const band = riskBand(data.probability);
    showResult(`Result: ${eligible} | Probability: ${data.probability.toFixed(4)} | Band: ${band} | Threshold: ${typeof data.threshold === "number" ? data.threshold.toFixed(4) : data.threshold}`);
  } catch (e) {
    showResult(`Error: ${e.message}`);
  } finally {
    setBusy(false);
  }
});

predictLiveBtn.addEventListener("click", async () => {
  setBusy(true);
  showResult("");
  try {
    const data = await apiFetch("/api/predict-live?n=10");
    lastRows = (data.rows || []).slice().sort((a, b) => {
      const pa = typeof a.probability === "number" ? a.probability : -Infinity;
      const pb = typeof b.probability === "number" ? b.probability : -Infinity;
      return pb - pa;
    });
    renderLiveTable(lastRows);
    showResult("Loaded live customers with eligibility.");
  } catch (e) {
    showResult(`Error: ${e.message}`);
  } finally {
    setBusy(false);
  }
});

window.addEventListener("load", async () => {
  try {
    const data = await apiFetch("/api/metrics");
    const thr = typeof data.threshold === "number" ? data.threshold.toFixed(4) : data.threshold;
    const header = document.querySelector(".header");
    if (header) {
      const info = document.createElement("p");
      info.textContent = "Decision Threshold: " + thr;
      header.appendChild(info);
    }
  } catch {}
});

filterEligibleBtn.addEventListener("click", () => {
  if (!Array.isArray(lastRows) || lastRows.length === 0) {
    showResult("Load live customers first.");
    return;
  }
  const filtered = lastRows.filter((r) => r.prediction === 1);
  renderLiveTable(filtered);
  showResult(`Filtered: Eligible (${filtered.length})`);
});

filterNotEligibleBtn.addEventListener("click", () => {
  if (!Array.isArray(lastRows) || lastRows.length === 0) {
    showResult("Load live customers first.");
    return;
  }
  const filtered = lastRows.filter((r) => r.prediction === 0);
  renderLiveTable(filtered);
  showResult(`Filtered: Not Eligible (${filtered.length})`);
});
