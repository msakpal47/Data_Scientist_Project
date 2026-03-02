function animateNumber(element, target, formatter, duration) {
  const start = 0;
  const startTime = performance.now();
  function step(now) {
    const t = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const value = start + (target - start) * eased;
    element.textContent = formatter ? formatter(value) : Math.round(value).toLocaleString();
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

async function loadSummary(txType) {
  const url = txType ? `/api/summary?tx_type=${encodeURIComponent(txType)}` : "/api/summary";
  const res = await fetch(url);
  const data = await res.json();
  const nonFraud = data.non_fraud;
  const fraud = data.fraud;
  const rate = data.fraud_rate;
  const flagged = data.flagged ?? 0;
  const flaggedRate = data.flagged_rate ?? 0;
  const imbalance = data.imbalance_ratio ?? "1:0";
  const elNon = document.getElementById("metric-nonfraud");
  const elFraud = document.getElementById("metric-fraud");
  const elRate = document.getElementById("metric-rate");
  const elFlagged = document.getElementById("metric-flagged");
  const elImb = document.getElementById("metric-imb");
  animateNumber(elNon, nonFraud, v => Math.round(v).toLocaleString(), 700);
  animateNumber(elFraud, fraud, v => Math.round(v).toLocaleString(), 700);
  animateNumber(elRate, rate * 100, v => v.toFixed(3) + "%", 900);
  if (elFlagged) animateNumber(elFlagged, flagged, v => Math.round(v).toLocaleString(), 700);
  if (elImb) elImb.textContent = imbalance;
}

function renderTable(data) {
  const head = document.getElementById("tx-head");
  const body = document.getElementById("tx-body");
  if (!head || !body) return;
  head.innerHTML = "";
  body.innerHTML = "";
  if (!data.rows || data.rows.length === 0) {
    head.innerHTML = "<tr><th>No rows</th></tr>";
    return;
  }
  const columns = data.columns;
  const trHead = document.createElement("tr");
  columns.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col;
    trHead.appendChild(th);
  });
  head.appendChild(trHead);
  data.rows.forEach(row => {
    const tr = document.createElement("tr");
    columns.forEach(col => {
      const td = document.createElement("td");
      const value = row[col];
      if (col === "isFraud") {
        const span = document.createElement("span");
        if (value === 1) {
          span.className = "pill-fraud";
          span.textContent = "Fraud";
        } else {
          span.className = "pill-ok";
          span.textContent = "Non‑fraud";
        }
        td.appendChild(span);
      } else {
        td.textContent = value;
      }
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

async function loadTransactions(filter, txType) {
  const head = document.getElementById("tx-head");
  const body = document.getElementById("tx-body");
  if (!head || !body) return;
  const page = Number(document.getElementById("page")?.value || 1);
  const limit = Number(document.getElementById("limit")?.value || 500);
  try {
    const url = `/api/transactions?filter=${encodeURIComponent(filter)}&page=${page}&limit=${limit}` + (txType ? `&tx_type=${encodeURIComponent(txType)}` : "");
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderTable(data);
    const total = data.total_rows ?? 0;
    const totalEl = document.getElementById("total_rows");
    if (totalEl) totalEl.textContent = `Total rows: ${total.toLocaleString()}`;
  } catch (err) {
    if (head) head.innerHTML = "<tr><th>Error loading rows</th></tr>";
    if (body) body.innerHTML = `<tr><td>${String(err)}</td></tr>`;
  }
}

function wireFilterChips() {
  const container = document.getElementById("filter-chips");
  if (!container) return;
  container.addEventListener("click", e => {
    const chip = e.target.closest(".filter-chip");
    if (!chip) return;
    const filter = chip.dataset.filter;
    container.querySelectorAll(".filter-chip").forEach(c => c.classList.remove("active"));
    chip.classList.add("active");
    loadTransactions(filter);
    updateDownloadLink(filter);
  });
}

function wirePredictForm() {
  const form = document.getElementById("predict-form");
  const output = document.getElementById("prediction-output");
  const thresholdInput = document.getElementById("threshold");
  form.addEventListener("submit", async e => {
    e.preventDefault();
    const formData = new FormData(form);
    const payload = {};
    for (const [key, value] of formData.entries()) {
      if (key === "type") {
        payload[key] = value;
      } else {
        payload[key] = Number(value);
      }
    }
    const threshold = Number(thresholdInput?.value || 0.5);
    payload.threshold = threshold;
    output.innerHTML = "<div class='prediction-label'>Scoring...</div>";
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const prob = data.probability;
      const label = data.label;
      const pct = (prob * 100).toFixed(3);
      const tagClass = label === 1 ? "fraud" : "ok";
      const tagText = label === 1 ? "Model flags this as FRAUD" : "Model considers this NON‑FRAUD";
      const riskLevel = prob >= 0.5 ? "HIGH" : prob >= 0.2 ? "MEDIUM" : prob >= 0.05 ? "LOW" : "VERY LOW";
      let summaryHtml = `
        <div class="prediction-label">Model decision</div>
        <div class="prediction-prob">Fraud probability: <strong>${pct}%</strong></div>
        <div class="prediction-prob">Risk level: <strong>${riskLevel}</strong> (threshold ${Math.round(threshold * 100)}%)</div>
        <div class="prediction-tag ${tagClass}">
          <span class="tiny-dot ${tagClass}"></span>
          <span>${tagText}</span>
        </div>
      `;
      output.innerHTML = summaryHtml;
      const explain = document.getElementById("local-explain");
      const wf = document.getElementById("shap-waterfall");
      if (explain) {
        explain.innerHTML = "<div class='caption'>Computing local explanation…</div>";
        try {
          const er = await fetch("/api/explain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          const ed = await er.json();
          const contribs = ed.contributions || [];
          explain.innerHTML = "";
          const max = Math.max(0.00001, ...contribs.map(c => Math.abs(c.effect)));
      if (contribs.length) {
        const top = contribs[0];
        const header = document.createElement("div");
        header.className = "caption";
        header.textContent = `Top driver: ${top.feature} (${(top.effect*100).toFixed(2)} pp)`;
        explain.appendChild(header);
            output.innerHTML = summaryHtml + `<div class="caption">Top contributing feature: <strong>${top.feature}</strong></div>`;
      }
          contribs.slice(0, 5).forEach(c => {
            const row = document.createElement("div");
            row.className = "feat-row";
            const name = document.createElement("div");
            name.className = "feat-name";
            name.textContent = c.feature;
            const bar = document.createElement("div");
            bar.className = "feat-bar";
            const fill = document.createElement("div");
            fill.className = "feat-fill";
            const width = Math.round((Math.abs(c.effect) / max) * 100);
            fill.style.width = width + "%";
            bar.appendChild(fill);
            row.appendChild(name);
            row.appendChild(bar);
            explain.appendChild(row);
          });
          if (!contribs.length) {
            explain.innerHTML = "<div class='caption'>No local explanation available</div>";
          }
          if (wf && wf.getContext) {
            const ctx = wf.getContext("2d");
            ctx.clearRect(0, 0, wf.width, wf.height);
            const base = Number(ed.base_value ?? ed.final_probability ?? 0);
            const finalP = Number(ed.final_probability ?? prob);
            const items = contribs.slice(0, 6);
            let cum = base;
            const pad = 24, w = wf.width - pad*2, h = wf.height - pad*2;
            ctx.strokeStyle = "#374151";
            ctx.lineWidth = 1;
            ctx.strokeRect(pad, pad, w, h);
            const x0 = pad + 90;
            const x1 = pad + w - 10;
            const yStep = h / (items.length + 2);
            function xFor(p){ return x0 + (Math.max(0, Math.min(1, p)) * (x1 - x0)); }
            ctx.fillStyle = "#9ca3af";
            ctx.font = "10px Segoe UI";
            ctx.fillText("0%", x0 - 26, pad + 10);
            ctx.fillText("100%", x1 - 18, pad + 10);
            ctx.fillText("base", pad + 8, pad + yStep);
            ctx.fillStyle = "#22d3ee";
            ctx.fillRect(xFor(base)-2, pad + yStep - 2, 4, 4);
            items.forEach((it, idx) => {
              const y = pad + (idx + 2) * yStep;
              const next = cum + Number(it.effect);
              const a = xFor(cum), b = xFor(next);
              ctx.fillStyle = (next - cum) >= 0 ? "#34d399" : "#f87171";
              ctx.fillRect(Math.min(a, b), y - 8, Math.abs(b - a), 16);
              ctx.fillStyle = "#9ca3af";
              ctx.fillText(String(it.feature).slice(0, 20), pad + 8, y + 3);
              cum = next;
            });
            ctx.fillStyle = "#9ca3af";
            ctx.fillText("final", pad + 8, pad + (items.length + 2) * yStep);
            ctx.fillStyle = "#f59e0b";
            ctx.fillRect(xFor(finalP)-2, pad + (items.length + 2) * yStep - 2, 4, 4);
          }
        } catch {
          explain.innerHTML = "<div class='caption'>Local explanation failed</div>";
        }
      }
    } catch (err) {
      output.innerHTML = `
        <div class="prediction-label">Prediction failed</div>
        <div class="prediction-prob">Error: ${String(err)}</div>
      `;
    }
  });
}

async function loadMetrics() {
  const elAcc = document.getElementById("metric-acc");
  const elRec = document.getElementById("metric-rec");
  const elPrec = document.getElementById("metric-prec");
  const elAuc = document.getElementById("metric-auc");
  const elF1 = document.getElementById("metric-f1");
  const cmTN = document.getElementById("cm-tn");
  const cmFP = document.getElementById("cm-fp");
  const cmFN = document.getElementById("cm-fn");
  const cmTP = document.getElementById("cm-tp");
  try {
    const res = await fetch(`/api/metrics?_=${Date.now()}`);
    const m = await res.json();
    const acc = Number(m.accuracy || 0);
    const prec = Number(m.precision || 0);
    const rec = Number(m.recall || 0);
    const auc = Number(m.roc_auc || 0);
    const f1 = Number(m.f1 || 0);
    if (elAcc) elAcc.textContent = (acc * 100).toFixed(3) + "%";
    if (elPrec) elPrec.textContent = (prec * 100).toFixed(3) + "%";
    if (elRec) elRec.textContent = (rec * 100).toFixed(3) + "%";
    if (elAuc) elAuc.textContent = auc.toFixed(3);
    if (elF1) elF1.textContent = f1.toFixed(3);
    const tsEl = document.getElementById("metric-ts");
    if (tsEl && m.evaluated_at) tsEl.textContent = m.evaluated_at;
    const cm = Array.isArray(m.confusion_matrix) ? m.confusion_matrix : [[0,0],[0,0]];
    const tn = Number((cm[0]||[])[0] || 0);
    const fp = Number((cm[0]||[])[1] || 0);
    const fn = Number((cm[1]||[])[0] || 0);
    const tp = Number((cm[1]||[])[1] || 0);
    if (cmTN) cmTN.textContent = tn;
    if (cmFP) cmFP.textContent = fp;
    if (cmFN) cmFN.textContent = fn;
    if (cmTP) cmTP.textContent = tp;
  } catch (e) {
    if (elAcc) elAcc.textContent = "0.000%";
    if (elPrec) elPrec.textContent = "0.000%";
    if (elRec) elRec.textContent = "0.000%";
    if (elAuc) elAuc.textContent = "0.000";
    if (elF1) elF1.textContent = "0.000";
    if (cmTN) cmTN.textContent = "0";
    if (cmFP) cmFP.textContent = "0";
    if (cmFN) cmFN.textContent = "0";
    if (cmTP) cmTP.textContent = "0";
  }
}

async function loadFeatureImportances() {
  const container = document.getElementById("feat-imp");
  if (!container) return;
  container.innerHTML = "";
  try {
    const res = await fetch(`/api/feature_importances?_=${Date.now()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const items = Array.isArray(data.importances) ? data.importances.slice(0, 10) : [];
    if (!items.length) {
      container.innerHTML = "<div class='caption'>No feature importance available</div>";
      return;
    }
    const max = Math.max(0.00001, ...items.map(i => Number(i.importance || 0)));
    items.forEach(i => {
      const row = document.createElement("div");
      row.className = "feat-row";
      const name = document.createElement("div");
      name.className = "feat-name";
      const pct = Math.round((Number(i.importance || 0) / items.reduce((s, x) => s + Number(x.importance || 0), 0.00001)) * 1000) / 10;
      name.textContent = `${i.feature} (${pct}%)`;
      const bar = document.createElement("div");
      bar.className = "feat-bar";
      const fill = document.createElement("div");
      fill.className = "feat-fill";
      fill.style.width = Math.round((Number(i.importance || 0) / max) * 100) + "%";
      bar.appendChild(fill);
      row.appendChild(name);
      row.appendChild(bar);
      container.appendChild(row);
    });
  } catch (err) {
    container.innerHTML = `<div class='caption'>Failed to load feature importance: ${String(err)}</div>`;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const typeFilter = document.getElementById("dataset-type-filter");
  const currentType = typeFilter?.value || "";
  loadSummary(currentType);
  loadTransactions("all", currentType);
  wireFilterChips();
  wirePredictForm();
  loadMetrics();
  loadFeatureImportances();
  const reload = document.getElementById("reload");
  if (reload) {
    reload.addEventListener("click", () => {
      const active = document.querySelector(".filter-chip.active");
      const filter = active?.dataset?.filter || "all";
      const txType = typeFilter?.value || "";
      loadTransactions(filter, txType);
    });
  }
  updateDownloadLink("all", currentType);
  if (typeFilter) {
    typeFilter.addEventListener("change", () => {
      const txType = typeFilter.value || "";
      const active = document.querySelector(".filter-chip.active");
      const filter = active?.dataset?.filter || "all";
      loadSummary(txType);
      loadTransactions(filter, txType);
      updateDownloadLink(filter, txType);
    });
  }
});

function updateDownloadLink(filter, txType) {
  const link = document.getElementById("download");
  if (link) {
    const url = `/api/export?filter=${encodeURIComponent(filter)}` + (txType ? `&tx_type=${encodeURIComponent(txType)}` : "");
    link.href = url;
  }
}

async function loadThresholdSuggestions() {
  const res = await fetch("/api/threshold_suggestion");
  const s = await res.json();
  const th = document.getElementById("threshold");
  const best = s.best_f1?.threshold ?? 0.5;
  const p08 = s.precision_0_8?.threshold ?? best;
  const r09 = s.recall_0_9?.threshold ?? best;
  const btnBest = document.getElementById("set-bestf1");
  const btnP = document.getElementById("set-prec08");
  const btnR = document.getElementById("set-rec09");
  function setVal(v){ if (th) { th.value = Number(v).toFixed(2); th.dispatchEvent(new Event("input")); } }
  if (btnBest) btnBest.addEventListener("click", () => setVal(best));
  if (btnP) btnP.addEventListener("click", () => setVal(p08));
  if (btnR) btnR.addEventListener("click", () => setVal(r09));
}

document.addEventListener("DOMContentLoaded", () => {
  loadThresholdSuggestions();
});

document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("metrics-reload");
  if (btn) btn.addEventListener("click", () => loadMetrics());
});

document.addEventListener("DOMContentLoaded", () => {
  const btnSim = document.getElementById("simulate-cm");
  const th = document.getElementById("threshold");
  if (btnSim && th) {
    btnSim.addEventListener("click", async () => {
      const threshold = Number(th.value || 0.5);
      try {
        const res = await fetch(`/api/confusion_sim?threshold=${encodeURIComponent(threshold)}`);
        const data = await res.json();
        const cm = data.confusion_matrix || [[0,0],[0,0]];
        const cmTN = document.getElementById("cm-tn");
        const cmFP = document.getElementById("cm-fp");
        const cmFN = document.getElementById("cm-fn");
        const cmTP = document.getElementById("cm-tp");
        if (cmTN) cmTN.textContent = cm[0][0];
        if (cmFP) cmFP.textContent = cm[0][1];
        if (cmFN) cmFN.textContent = cm[1][0];
        if (cmTP) cmTP.textContent = cm[1][1];
      } catch {}
    });
    th.addEventListener("input", async () => {
      const threshold = Number(th.value || 0.5);
      try {
        const res = await fetch(`/api/confusion_sim?threshold=${encodeURIComponent(threshold)}`);
        const data = await res.json();
        const cm = data.confusion_matrix || [[0,0],[0,0]];
        const cmTN = document.getElementById("cm-tn");
        const cmFP = document.getElementById("cm-fp");
        const cmFN = document.getElementById("cm-fn");
        const cmTP = document.getElementById("cm-tp");
        if (cmTN) cmTN.textContent = cm[0][0];
        if (cmFP) cmFP.textContent = cm[0][1];
        if (cmFN) cmFN.textContent = cm[1][0];
        if (cmTP) cmTP.textContent = cm[1][1];
        const costFP = Number(document.getElementById("cost-fp")?.value || 5);
        const costFN = Number(document.getElementById("cost-fn")?.value || 500);
        const resLoss = await fetch(`/api/cost_sim?threshold=${encodeURIComponent(threshold)}&cost_fp=${encodeURIComponent(costFP)}&cost_fn=${encodeURIComponent(costFN)}`);
        const dataLoss = await resLoss.json();
        const elLoss = document.getElementById("loss-value");
        if (elLoss) elLoss.textContent = (Number(dataLoss.expected_loss || 0)).toLocaleString(undefined, {maximumFractionDigits: 2});
      } catch {}
    });
  }
});

document.addEventListener("DOMContentLoaded", () => {
  const btnLoss = document.getElementById("compute-loss");
  const th = document.getElementById("threshold");
  const elLoss = document.getElementById("loss-value");
  const elFP = document.getElementById("cost-fp");
  const elFN = document.getElementById("cost-fn");
  if (btnLoss && th && elLoss) {
    btnLoss.addEventListener("click", async () => {
      const threshold = Number(th.value || 0.5);
      const costFP = Number(elFP?.value || 5);
      const costFN = Number(elFN?.value || 500);
      try {
        const res = await fetch(`/api/cost_sim?threshold=${encodeURIComponent(threshold)}&cost_fp=${encodeURIComponent(costFP)}&cost_fn=${encodeURIComponent(costFN)}`);
        const data = await res.json();
        elLoss.textContent = (Number(data.expected_loss || 0)).toLocaleString(undefined, {maximumFractionDigits: 2});
        const cm = data.confusion_matrix || [[0,0],[0,0]];
        const cmTN = document.getElementById("cm-tn");
        const cmFP = document.getElementById("cm-fp");
        const cmFN = document.getElementById("cm-fn");
        const cmTP = document.getElementById("cm-tp");
        if (cmTN) cmTN.textContent = cm[0][0];
        if (cmFP) cmFP.textContent = cm[0][1];
        if (cmFN) cmFN.textContent = cm[1][0];
        if (cmTP) cmTP.textContent = cm[1][1];
      } catch {}
    });
    [elFP, elFN].forEach(inp => inp && inp.addEventListener("change", async () => {
      const threshold = Number(th.value || 0.5);
      const costFP = Number(elFP?.value || 5);
      const costFN = Number(elFN?.value || 500);
      try {
        const res = await fetch(`/api/cost_sim?threshold=${encodeURIComponent(threshold)}&cost_fp=${encodeURIComponent(costFP)}&cost_fn=${encodeURIComponent(costFN)}`);
        const data = await res.json();
        elLoss.textContent = (Number(data.expected_loss || 0)).toLocaleString(undefined, {maximumFractionDigits: 2});
      } catch {}
    }));
  }
});

document.addEventListener("DOMContentLoaded", () => {
  const btnSuggest = document.getElementById("suggest-th");
  const th = document.getElementById("threshold");
  const elFP = document.getElementById("cost-fp");
  const elFN = document.getElementById("cost-fn");
  const elTh = document.getElementById("optimal-th");
  const elLoss = document.getElementById("optimal-loss");
  const biz = document.getElementById("biz-reco");
  if (btnSuggest && th) {
    btnSuggest.addEventListener("click", async () => {
      const costFP = Number(elFP?.value || 5);
      const costFN = Number(elFN?.value || 500);
      try {
        const res = await fetch(`/api/optimal_threshold?cost_fp=${encodeURIComponent(costFP)}&cost_fn=${encodeURIComponent(costFN)}`);
        const data = await res.json();
        const optTh = Number(data.optimal_threshold || 0.5);
        const minLoss = Number(data.min_expected_loss || 0);
        if (elTh) elTh.textContent = optTh.toFixed(2);
        if (elLoss) elLoss.textContent = minLoss.toLocaleString(undefined, {maximumFractionDigits: 2});
        th.value = optTh.toFixed(2);
        th.dispatchEvent(new Event("input"));
        try {
          const r = await fetch(`/api/confusion_sim?threshold=${encodeURIComponent(optTh)}`);
          const d = await r.json();
          const cm = d.confusion_matrix || [[0,0],[0,0]];
          const tp = Number((cm[1]||[])[1] || 0);
          const fp = Number((cm[0]||[])[1] || 0);
          if (biz) biz.textContent = `Prevents ${tp.toLocaleString()} fraud with ${fp.toLocaleString()} false alerts`;
          const thEl = document.getElementById("impact-th");
          const detEl = document.getElementById("impact-detected");
          const missEl = document.getElementById("impact-missed");
          const falEl = document.getElementById("impact-false");
          const lossEl = document.getElementById("impact-loss");
          const fn = Number((cm[1]||[])[0] || 0);
          if (thEl) thEl.textContent = optTh.toFixed(2);
          if (detEl) detEl.textContent = tp.toLocaleString();
          if (missEl) missEl.textContent = fn.toLocaleString();
          if (falEl) falEl.textContent = fp.toLocaleString();
          if (lossEl) lossEl.textContent = minLoss.toLocaleString(undefined, {maximumFractionDigits: 2});
        } catch {}
      } catch {}
    });
  }
});

async function loadPrCurve() {
  const canvas = document.getElementById("pr-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  try {
    const res = await fetch(`/api/pr_curve?_=${Date.now()}`);
    const data = await res.json();
    const curve = Array.isArray(data.curve) ? data.curve : [];
    if (!curve.length) {
      ctx.fillStyle = "#9ca3af";
      ctx.fillText("No PR data", 10, 20);
      return;
    }
    const w = canvas.width, h = canvas.height, pad = 24;
    function xFor(t) { return pad + (t * (w - 2*pad)); }
    function yFor(v) { return h - pad - (v * (h - 2*pad)); }
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2*pad, h - 2*pad);
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px Segoe UI";
    ctx.fillText("threshold", w/2 - 20, h - 6);
    ctx.save();
    ctx.translate(8, h/2 + 20);
    ctx.rotate(-Math.PI/2);
    ctx.fillText("value", 0, 0);
    ctx.restore();
    ctx.beginPath();
    ctx.strokeStyle = "#22d3ee";
    curve.forEach((p, i) => {
      const x = xFor(Number(p.threshold || 0));
      const y = yFor(Number(p.precision || 0));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.beginPath();
    ctx.strokeStyle = "#f97316";
    curve.forEach((p, i) => {
      const x = xFor(Number(p.threshold || 0));
      const y = yFor(Number(p.recall || 0));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#22d3ee";
    ctx.fillRect(w - pad - 100, pad + 6, 10, 3);
    ctx.fillStyle = "#9ca3af";
    ctx.fillText("precision", w - pad - 84, pad + 10);
    ctx.fillStyle = "#f97316";
    ctx.fillRect(w - pad - 100, pad + 22, 10, 3);
    ctx.fillStyle = "#9ca3af";
    ctx.fillText("recall", w - pad - 84, pad + 26);
  } catch (e) {}
}

document.addEventListener("DOMContentLoaded", () => {
  loadPrCurve();
});

async function loadCostCurve() {
  const canvas = document.getElementById("cost-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  try {
    const costFP = Number(document.getElementById("cost-fp")?.value || 5);
    const costFN = Number(document.getElementById("cost-fn")?.value || 500);
    const res = await fetch(`/api/cost_curve?cost_fp=${encodeURIComponent(costFP)}&cost_fn=${encodeURIComponent(costFN)}&_=${Date.now()}`);
    const data = await res.json();
    const curve = Array.isArray(data.curve) ? data.curve : [];
    if (!curve.length) {
      ctx.fillStyle = "#9ca3af";
      ctx.fillText("No cost data", 10, 20);
      return;
    }
    const w = canvas.width, h = canvas.height, pad = 24;
    function xFor(t) { return pad + (t * (w - 2*pad)); }
    const losses = curve.map(p => Number(p.loss || 0));
    const maxL = Math.max(...losses, 1);
    function yFor(l) { return h - pad - ((l / maxL) * (h - 2*pad)); }
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2*pad, h - 2*pad);
    ctx.beginPath();
    ctx.strokeStyle = "#22c55e";
    curve.forEach((p, i) => {
      const x = xFor(Number(p.threshold || 0));
      const y = yFor(Number(p.loss || 0));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px Segoe UI";
    ctx.fillText("threshold", w/2 - 20, h - 6);
    ctx.save();
    ctx.translate(8, h/2 + 20);
    ctx.rotate(-Math.PI/2);
    ctx.fillText("expected loss", 0, 0);
    ctx.restore();
  } catch (e) {}
}

document.addEventListener("DOMContentLoaded", () => {
  loadCostCurve();
  const fp = document.getElementById("cost-fp");
  const fn = document.getElementById("cost-fn");
  [fp, fn].forEach(inp => inp && inp.addEventListener("change", () => loadCostCurve()));
});

async function loadProbHistogram() {
  const canvas = document.getElementById("prob-hist");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  try {
    const res = await fetch(`/api/prob_histogram?_=${Date.now()}`);
    const data = await res.json();
    const neg = Array.isArray(data.neg) ? data.neg : [];
    const pos = Array.isArray(data.pos) ? data.pos : [];
    const w = canvas.width, h = canvas.height, pad = 24;
    const maxCount = Math.max(1, ...neg, ...pos);
    const bw = (w - 2*pad) / Math.max(neg.length, 1);
    function yFor(c){ return h - pad - ((c / maxCount) * (h - 2*pad)); }
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, w - 2*pad, h - 2*pad);
    for (let i = 0; i < neg.length; i++) {
      const x = pad + i * bw;
      const nh = (h - 2*pad) * (neg[i] / maxCount);
      const ph = (h - 2*pad) * (pos[i] / maxCount);
      ctx.fillStyle = "rgba(148,163,184,0.45)";
      ctx.fillRect(x + 1, h - pad - nh, bw/2 - 2, nh);
      ctx.fillStyle = "rgba(248, 113, 113, 0.6)";
      ctx.fillRect(x + bw/2 + 1, h - pad - ph, bw/2 - 2, ph);
    }
    ctx.fillStyle = "#9ca3af";
    ctx.font = "10px Segoe UI";
    ctx.fillText("non‑fraud", w - pad - 120, pad + 10);
    ctx.fillStyle = "rgba(148,163,184,0.45)";
    ctx.fillRect(w - pad - 140, pad + 6, 10, 6);
    ctx.fillStyle = "#9ca3af";
    ctx.fillText("fraud", w - pad - 120, pad + 26);
    ctx.fillStyle = "rgba(248, 113, 113, 0.6)";
    ctx.fillRect(w - pad - 140, pad + 22, 10, 6);
  } catch {}
}

document.addEventListener("DOMContentLoaded", () => {
  loadProbHistogram();
});

async function loadModelCompare() {
  const body = document.getElementById("mc-body");
  if (!body) return;
  body.innerHTML = "<tr><td>Loading…</td><td></td><td></td><td></td><td></td><td></td></tr>";
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 20000);
    const res = await fetch(`/api/model_compare?_=${Date.now()}`, { signal: ctrl.signal });
    clearTimeout(timer);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt.slice(0, 200)}`);
    }
    const data = await res.json();
    const names = ["HGB", "LogisticRegression", "RandomForest"];
    const metrics = names.map(n => ({ name: n, ...(data[n] || {}) }));
    const bestF1 = Math.max(...metrics.map(m => Number(m.f1 || 0)));
    const bestAUC = Math.max(...metrics.map(m => Number(m.roc_auc || 0)));
    const rows = metrics.map(m => {
      const acc = ((m.accuracy || 0) * 100).toFixed(3) + "%";
      const prec = ((m.precision || 0) * 100).toFixed(3) + "%";
      const rec = ((m.recall || 0) * 100).toFixed(3) + "%";
      const f1v = Number(m.f1 || 0);
      const aucv = Number(m.roc_auc || 0);
      const f1 = f1v.toFixed(3) + (f1v === bestF1 ? ` <span class="badge-best">Best F1</span>` : "");
      const auc = aucv.toFixed(3) + (aucv === bestAUC ? ` <span class="badge-best">Best ROC‑AUC</span>` : "");
      const cls = (f1v === bestF1 || aucv === bestAUC) ? ' class="mc-highlight"' : "";
      return `<tr${cls}><td>${m.name}</td><td>${acc}</td><td>${prec}</td><td>${rec}</td><td>${f1}</td><td>${auc}</td></tr>`;
    }).join("");
    body.innerHTML = rows;
    const recModel = metrics.reduce((best, cur) => (Number(cur.roc_auc || 0) > Number(best.roc_auc || 0) ? cur : best), metrics[0] || {name:"—"});
    const reco = document.getElementById("mc-reco");
    if (reco && recModel) {
      const auc = Number(recModel.roc_auc || 0).toFixed(3);
      const recall = Number(recModel.recall || 0).toFixed(3);
      reco.textContent = `Recommended production model: ${recModel.name} — Reason: Highest ROC‑AUC (${auc}) and balanced recall (${recall})`;
    }
  } catch (e) {
    body.innerHTML = `<tr><td>Error</td><td colspan="5">${String(e)}</td></tr>`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadModelCompare();
});
