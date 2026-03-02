/* ==============================
   Electric Motor Dashboard JS
   Clean – Stable – Interview Ready
================================= */

let importanceChart = null;
let shapChart = null;
let residualChart = null;

/* ==============================
   Utility Helpers
================================= */

function fmt(n, d = 2) {
  return (typeof n === "number" && isFinite(n)) ? n.toFixed(d) : "--";
}

function getNum(id) {
  const v = parseFloat(document.getElementById(id)?.value);
  return Number.isFinite(v) ? v : null;
}

function showError(id, msg) {
  const el = document.getElementById(id);
  if (el) el.textContent = msg || "";
}

// Fallback drawings when Chart.js is unavailable
function drawHBar(canvas, labels, values) {
  if (!canvas) return;
  const dpr = 1;
  let w = canvas.clientWidth || 600, h = canvas.clientHeight || 320;
  if (!w || w <= 0) {
    const rect = canvas.parentElement ? canvas.parentElement.getBoundingClientRect() : { width: 600, height: 320 };
    w = Math.max(300, rect.width || 600);
    h = Math.max(200, rect.height || 320);
    canvas.style.width = "100%";
    canvas.style.height = h + "px";
  }
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const n = Math.min(labels.length, 6);
  const order = [...Array(labels.length).keys()].sort((a, b) => (values[b] || 0) - (values[a] || 0)).slice(0, n);
  const max = Math.max(...order.map(i => values[i] || 0), 1e-9);
  const pad = 12, rowH = (h - 2 * pad) / n, barMaxW = w * 0.55, xLabel = 10, xBar = w * 0.35;
  ctx.font = "12px Arial";
  for (let r = 0; r < n; r++) {
    const i = order[r];
    const y = pad + r * rowH + 6;
    const pct = Math.max(2, (values[i] / max) * barMaxW);
    ctx.fillStyle = "#e6f7ff";
    ctx.fillText(`${r + 1}. ${labels[i]}`, xLabel, y + 8);
    ctx.globalAlpha = 0.25;
    ctx.fillStyle = "#065f46";
    ctx.fillRect(xBar, y, barMaxW, 10);
    ctx.globalAlpha = 1;
    const grad = ctx.createLinearGradient(xBar, 0, xBar + pct, 0);
    grad.addColorStop(0, "#60a5fa"); grad.addColorStop(1, "#06b6d4");
    ctx.fillStyle = grad;
    ctx.fillRect(xBar, y, pct, 10);
    ctx.fillStyle = "#cfefff";
    const val = values[i];
    ctx.fillText((typeof val === "number" ? val.toFixed(4) : String(val)), xBar + barMaxW + 8, y + 8);
  }
}
function drawScatter(canvas, points) {
  if (!canvas || !points || !points.length) return;
  const dpr = 1;
  let w = canvas.clientWidth || 600, h = canvas.clientHeight || 320;
  if (!w || w <= 0) {
    const rect = canvas.parentElement ? canvas.parentElement.getBoundingClientRect() : { width: 600, height: 320 };
    w = Math.max(300, rect.width || 600);
    h = Math.max(200, rect.height || 320);
    canvas.style.width = "100%";
    canvas.style.height = h + "px";
  }
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const xs = points.map(p => p.x), ys = points.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 28;
  const sx = x => pad + ((x - minX) / (maxX - minX || 1)) * (w - 2 * pad);
  const sy = y => (h - pad) - ((y - minY) / (maxY - minY || 1)) * (h - 2 * pad);
  const zy = sy(0);
  ctx.strokeStyle = "rgba(255,99,132,0.6)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, zy); ctx.lineTo(w - pad, zy); ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.7)";
  for (const p of points) {
    const x = sx(p.x), y = sy(p.y);
    ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2); ctx.fill();
  }
}

/* ==============================
   Prediction
================================= */

async function predict() {
  showError("predictError", "");
  const payload = {
    u_q: getNum("u_q"),
    u_d: getNum("u_d"),
    i_d: getNum("i_d"),
    i_q: getNum("i_q"),
    coolant: getNum("coolant"),
    motor_speed: getNum("motor_speed"),
    ambient: getNum("ambient")
  };
  if (Object.values(payload).some(v => v === null)) {
    showError("predictError", "Please fill all numeric inputs.");
    return;
  }
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) {
      showError("predictError", data?.error || "Prediction failed.");
      return;
    }
    document.getElementById("result").innerText = fmt(data.predicted_pm) + " °C";
    if (typeof data.rmse === "number") {
      document.getElementById("uncertainty").innerText =
        `±RMSE: ${fmt(data.rmse)} °C | Down: ${fmt(data.lower)} °C | Up: ${fmt(data.upper)} °C`;
      updateConfidenceMeter(data.predicted_pm, data.rmse);
    }
    await Promise.all([loadImportance(), loadShap(), loadMetrics()]);
    await explainForPayload(payload);
  } catch {
    showError("predictError", "Server error.");
  }
}

/* ==============================
   Feature Importance
================================= */

async function loadImportance() {
  try {
    const res = await fetch("/importance");
    if (!res.ok) throw new Error();
    const data = await res.json();
    window._importanceData = Object.entries(data).sort((a,b)=>b[1]-a[1]);
    renderImportanceTable();
  } catch {
    showError("importanceError", "Importance unavailable.");
  }
}

function renderImportanceTable(){
  const tbl=document.getElementById("importanceTable");
  if(!tbl || !window._importanceData) return;
  const filter=(document.getElementById("importanceFilter")?.value||"").toLowerCase();
  const topSel=document.getElementById("importanceTopN")?.value||"10";
  let rows=window._importanceData.filter(([k])=>!filter || k.toLowerCase().includes(filter));
  if(topSel!=="all"){ rows=rows.slice(0,parseInt(topSel,10)||10); }
  const header = "<thead><tr><th>#</th><th>Feature</th><th>Importance</th></tr></thead>";
  const body = "<tbody>" + rows.map(([k,v],i)=>`<tr><td>${i+1}</td><td>${k}</td><td>${fmt(v,4)}</td></tr>`).join("") + "</tbody>";
  tbl.innerHTML = header + body;
  updateEngineeringInsight();
}
/* ==============================
   SHAP
================================= */

async function loadShap() {
  try {
    const res = await fetch("/shap");
    if (!res.ok) throw new Error();
    const data = await res.json();
    const entries = Object.entries(data);
    if(!entries.length){
      const btn=document.getElementById("shapGenerateBtn");
      if(btn){ btn.style.display="inline-block"; btn.onclick=generateShap; }
      const tbl=document.getElementById("shapTable");
      if(tbl){ tbl.innerHTML="<thead><tr><th>#</th><th>Feature</th><th>SHAP |mean|</th></tr></thead><tbody></tbody>"; }
      return;
    }
    window._shapData = entries.sort((a,b)=>b[1]-a[1]);
    renderShapTable();
  } catch {
    showError("shapError", "SHAP unavailable.");
  }
}

async function generateShap(){
  try{
    const btn=document.getElementById("shapGenerateBtn");
    if(btn){ btn.disabled=true; btn.textContent="Generating…"; }
    const res=await fetch("/generate-shap");
    if(!res.ok) throw new Error();
    await loadShap();
  }catch(e){
    showError("shapError","Failed to generate SHAP.");
  }finally{
    const btn=document.getElementById("shapGenerateBtn");
    if(btn){ btn.disabled=false; btn.textContent="Generate SHAP"; }
  }
}
function renderShapTable(){
  const tbl=document.getElementById("shapTable");
  if(!tbl || !window._shapData) return;
  const filter=(document.getElementById("shapFilter")?.value||"").toLowerCase();
  const topSel=document.getElementById("shapTopN")?.value||"10";
  let rows=window._shapData.filter(([k])=>!filter || k.toLowerCase().includes(filter));
  if(topSel!=="all"){ rows=rows.slice(0,parseInt(topSel,10)||10); }
  const header = "<thead><tr><th>#</th><th>Feature</th><th>SHAP |mean|</th></tr></thead>";
  const body = "<tbody>" + rows.map(([k,v],i)=>`<tr><td>${i+1}</td><td>${k}</td><td>${fmt(v,4)}</td></tr>`).join("") + "</tbody>";
  tbl.innerHTML = header + body;
  const driversEl=document.getElementById("shapDrivers");
  if(driversEl && rows.length){ const top=rows.slice(0,3).map(r=>r[0]); driversEl.textContent=`Top Drivers: ${top.join(", ")}`; }
  updateEngineeringInsight();
}
/* ==============================
   Residual Plot
================================= */

async function loadResiduals() {
  try {
    const res = await fetch("/residuals");
    if (!res.ok) throw new Error();
    const data = await res.json();
    if (!data.points || !data.points.length) return;
    const table=document.getElementById("residualTable");
    const sample=data.points.slice(0,50);
    if(table){
      table.innerHTML = "<tr><th>Pred</th><th>Resid</th></tr>" +
        sample.map(p=>`<tr><td>${fmt(p.pred,2)}</td><td>${fmt(p.resid,2)}</td></tr>`).join("");
    }
    const n = sample.length;
    let sum = 0, sumsq = 0, sumabs = 0;
    for (const p of sample) { const y=p.resid; sum += y; sumsq += y*y; sumabs += Math.abs(y); }
    const mean = sum / n;
    const std = Math.sqrt(Math.max(0, (sumsq / n) - (mean * mean)));
    const mae = sumabs / n;
    const stats = document.getElementById("residualStats");
    if (stats) stats.textContent = `Residual Mean: ${fmt(mean)}°C | Std Dev: ${fmt(std)}°C | MAE: ${fmt(mae)}°C`;
  } catch {
    showError("residualsError", "Residuals unavailable.");
  }
}

/* ==============================
   Metrics
================================= */

async function loadMetrics() {
  try {
    const res = await fetch("/metrics");
    if (!res.ok) throw new Error();
    const m = await res.json();
    let r2, mae, rmse;
    if (m && m.models) {
      const best = m.primary || Object.keys(m.models)[0];
      const b = m.models[best] || {};
      r2 = b.r2; mae = b.mae; rmse = b.rmse;
    } else {
      r2 = m.r2; mae = m.mae; rmse = m.rmse;
    }
    window._metrics = { r2, mae, rmse };
    const tbl = document.getElementById("metricsTable");
    if (tbl) {
      if (m && m.models) {
        const primary = m.primary || Object.keys(m.models)[0];
        const vals = m.models[primary] || {};
        const label = `${String(primary).toUpperCase()} (Primary)`;
        const row = `<tr><td>1</td><td>${label} ★</td><td>${fmt(vals.r2,3)}</td><td>${fmt(vals.mae)}</td><td>${fmt(vals.rmse)}</td></tr>`;
        tbl.innerHTML = `<thead><tr><th>#</th><th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>${row}</tbody>`;
      } else {
        const row = `<tr><td>1</td><td>XGB (Primary) ★</td><td>${fmt(r2,3)}</td><td>${fmt(mae)}</td><td>${fmt(rmse)}</td></tr>`;
        tbl.innerHTML = `<thead><tr><th>#</th><th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>${row}</tbody>`;
      }
    }
    updateEngineeringInsight();
  } catch {
    showError("metricsError", "Metrics unavailable.");
  }
}

async function loadComparison() {
  try {
    const res = await fetch("/comparison");
    if (!res.ok) throw new Error();
    const rows = await res.json();
    const tbl = document.getElementById("comparisonTable");
    if (tbl && Array.isArray(rows)) {
      if (rows.length <= 1) {
        tbl.innerHTML = `<thead><tr><th>#</th><th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th></tr></thead>
          <tbody>
            ${rows.map((r,i)=>`<tr><td>${i+1}</td><td>${r.model}</td><td>${fmt(r.r2,3)}</td><td>${fmt(r.mae)}</td><td>${fmt(r.rmse)}</td></tr>`).join("")}
            <tr><td colspan="5" style="text-align:left;color:#cfefff">Only one model available — train additional models to compare.</td></tr>
          </tbody>`;
      } else {
        const body = rows.map((r, i) =>
          `<tr><td>${i + 1}</td><td>${r.model}</td><td>${fmt(r.r2,3)}</td><td>${fmt(r.mae)}</td><td>${fmt(r.rmse)}</td></tr>`
        ).join("");
        tbl.innerHTML = `<thead><tr><th>#</th><th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th></tr></thead><tbody>${body}</tbody>`;
      }
    }
  } catch {
    showError("comparisonError", "Comparison unavailable.");
  }
}

/* ==============================
   Initialize
================================= */

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("predictBtn")?.addEventListener("click", (e) => {
    e.preventDefault();
    predict();
  });
  loadImportance();
  loadShap();
  loadMetrics();
  loadComparison();
  document.getElementById("importanceFilter")?.addEventListener("input",renderImportanceTable);
  document.getElementById("importanceTopN")?.addEventListener("change",renderImportanceTable);
  document.getElementById("shapFilter")?.addEventListener("input",renderShapTable);
  document.getElementById("shapTopN")?.addEventListener("change",renderShapTable);
  document.getElementById("residualSampleBtn")?.addEventListener("click",loadResiduals);
});

// expose for onclick fallback in markup
window.predict = predict;

function updateEngineeringInsight(){
  const pnl=document.getElementById("insightPanel");
  if(!pnl) return;
  const impTop=(window._importanceData||[]).slice(0,3).map(r=>r[0]);
  const shapTop=(window._shapData||[]).slice(0,3).map(r=>r[0]);
  const overlap=impTop.filter(k=>shapTop.includes(k));
  const rmse = window._metrics?.rmse;
  let parts=[];
  if(impTop.length){ parts.push(`Top Drivers (Importance): ${impTop.join(", ")}`); }
  if(shapTop.length){ parts.push(`Top Drivers (SHAP): ${shapTop.join(", ")}`); }
  if(overlap.length){ parts.push(`Consensus Drivers: ${overlap.join(", ")}`); }
  if(typeof rmse==="number"){ parts.push(`Typical ±RMSE: ${fmt(rmse)} °C`); }
  if(!parts.length){ parts.push("Insights will appear after data loads."); }
  pnl.innerHTML = parts.map(t=>`<div class="insight">${t}</div>`).join("");
}

async function explainForPayload(payload){
  try{
    const res=await fetch("/explain",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
    if(!res.ok) return;
    const ex=await res.json();
    if(ex && ex.abs){
      window._shapData = Object.entries(ex.abs).sort((a,b)=>b[1]-a[1]);
      renderShapTable();
    }
  }catch{}
}

function updateConfidenceMeter(pred, rmse){
  const fill=document.getElementById("confidenceFill");
  const text=document.getElementById("confidenceLabel");
  if(!fill || !text) return;
  if(typeof rmse!=="number"){ fill.style.width="0%"; text.textContent=""; return; }
  const conf = Math.max(5, Math.min(95, 100 - rmse*20));
  fill.style.width = conf + "%";
  text.textContent = `Confidence: ${conf.toFixed(0)}%`;
}
