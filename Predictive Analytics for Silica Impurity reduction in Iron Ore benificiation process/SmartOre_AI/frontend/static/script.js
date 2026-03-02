const form = document.getElementById('predict-form');
const result = document.getElementById('result');
const silicaEl = document.getElementById('silica');
const riskEl = document.getElementById('risk');
const recsEl = document.getElementById('recs');
const presetEl = document.getElementById('preset');
const statusBox = document.getElementById('model-status');
const verEl = document.getElementById('model-version');
const trainedEl = document.getElementById('model-trained');
const r2El = document.getElementById('model-r2');
const r2TrainEl = document.getElementById('model-r2-train');
const r2TestEl = document.getElementById('model-r2-test');
const maeEl = document.getElementById('model-mae');
const rmseEl = document.getElementById('model-rmse');
const overlay = document.getElementById('overlay');
let inflight = null;
const fiBox = document.getElementById('fi');
const predictBtn = document.getElementById('predict-btn');
const trendCanvas = document.getElementById('trend');
const trendCtx = trendCanvas ? trendCanvas.getContext('2d') : null;
const trendData = [];

function serializeForm(f) {
  const data = {};
  const fd = new FormData(f);
  for (const [k, v] of fd.entries()) {
    if (v === '') continue;
    data[k] = Number(v);
  }
  return data;
}

const PRESETS = {
  typical: {
    "% Silica Feed": 4.2,
    "Starch Flow": 110,
    "Amina Flow": 22,
    "Ore Pulp pH": 9.6,
    "Ore Pulp Density": 1.24,
    "Ore Pulp Flow": 780,
    "Avg Air Flow": 95
  },
  high_silica: {
    "% Silica Feed": 6.5,
    "Starch Flow": 130,
    "Amina Flow": 28,
    "Ore Pulp pH": 9.4,
    "Ore Pulp Density": 1.26,
    "Ore Pulp Flow": 800,
    "Avg Air Flow": 100
  },
  low_air: {
    "% Silica Feed": 4.0,
    "Starch Flow": 115,
    "Amina Flow": 24,
    "Ore Pulp pH": 9.7,
    "Ore Pulp Density": 1.25,
    "Ore Pulp Flow": 770,
    "Avg Air Flow": 70
  },
  high_ph: {
    "% Silica Feed": 4.0,
    "Starch Flow": 100,
    "Amina Flow": 20,
    "Ore Pulp pH": 10.4,
    "Ore Pulp Density": 1.23,
    "Ore Pulp Flow": 760,
    "Avg Air Flow": 90
  }
};

function fillForm(values) {
  if (!values) return;
  Object.entries(values).forEach(([k, v]) => {
    const el = form.querySelector(`[name="${CSS.escape(k)}"]`);
    if (el) el.value = v;
  });
}

presetEl?.addEventListener('change', (e) => {
  const v = presetEl.value;
  if (v === 'custom') return;
  fillForm(PRESETS[v]);
});

async function loadStatus() {
  try {
    const r = await fetch('/api/status');
    const j = await r.json();
    verEl.textContent = j.version ? `v${j.version}` : 'v–';
    trainedEl.textContent = j.trained_at ? new Date(j.trained_at).toLocaleString() : '–';
    r2El.textContent = j.r2 != null ? `R² ${Number(j.r2).toFixed(3)}` : 'R² –';
    r2TrainEl.textContent = j.r2_train != null ? `R²tr ${Number(j.r2_train).toFixed(3)}` : 'R²tr –';
    r2TestEl.textContent = j.r2_test != null ? `R²te ${Number(j.r2_test).toFixed(3)}` : 'R²te –';
    maeEl.textContent = j.mae_test != null ? `MAE ${Number(j.mae_test).toFixed(3)}` : 'MAE –';
    rmseEl.textContent = j.rmse_test != null ? `RMSE ${Number(j.rmse_test).toFixed(3)}` : 'RMSE –';
    fiBox.innerHTML = '';
    const src = (j.shap_top && j.shap_top.length) ? j.shap_top : (j.importances || []);
    const imps = src.slice().sort((a,b)=>b.importance-a.importance).slice(0,10);
    imps.forEach(it => {
      const row = document.createElement('div');
      row.className = 'row';
      const name = document.createElement('div');
      name.className = 'name';
      name.textContent = it.feature;
      const bar = document.createElement('div');
      bar.className = 'bar';
      const span = document.createElement('span');
      span.style.width = `${Math.max(2, it.importance*100)}%`;
      bar.appendChild(span);
      row.appendChild(name);
      row.appendChild(bar);
      fiBox.appendChild(row);
    });
  } catch (e) {
    verEl.textContent = 'v–';
    trainedEl.textContent = '–';
    r2El.textContent = 'R² –';
    if (r2TrainEl) r2TrainEl.textContent = 'R²tr –';
    if (r2TestEl) r2TestEl.textContent = 'R²te –';
    if (maeEl) maeEl.textContent = 'MAE –';
    if (rmseEl) rmseEl.textContent = 'RMSE –';
  }
}
loadStatus();

window.addEventListener('DOMContentLoaded', () => {
  // Ensure overlay is a direct child of body to avoid layout side-effects
  if (overlay && overlay.parentElement !== document.body) {
    try { document.body.appendChild(overlay); } catch {}
  }
  if (presetEl) {
    presetEl.value = 'typical';
    fillForm(PRESETS['typical']);
  }
});

async function doPredict() {
  const payload = serializeForm(form);
  try {
    console.log('[predict] payload ->', payload);
    if (predictBtn) predictBtn.disabled = true;
    // Abort any in-flight request
    if (inflight && typeof inflight.abort === 'function') {
      try { inflight.abort(); } catch {}
    }
    inflight = new AbortController();
    const { signal } = inflight;
    const start = Date.now();
    if (overlay) overlay.style.display = 'none';
    // Client-side timeout to prevent indefinite spinner
    const timeout = setTimeout(() => {
      try { inflight.abort(); } catch {}
    }, 5000);
    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
      signal
    });
    clearTimeout(timeout);
    console.log('[predict] status ->', r.status);
    const j = await r.json();
    console.log('[predict] response ->', j);
    const elapsed = Date.now() - start;
    if (elapsed < 0) {}
    if (overlay) overlay.style.display = 'none';
    if (j.error) {
      alert(j.error);
      return;
    }
    silicaEl.textContent = j.silica_concentrate?.toFixed(3);
    riskEl.textContent = j.risk;
    if (typeof j.silica_concentrate === 'number') {
      const t = Date.now();
      trendData.push({ t, y: j.silica_concentrate });
      if (trendData.length > 50) trendData.shift();
      drawTrend();
    }
    recsEl.innerHTML = '';
    (j.recommendations || []).forEach(tip => {
      const li = document.createElement('li');
      li.textContent = tip;
      recsEl.appendChild(li);
    });
    result.classList.remove('hidden');
    if (typeof j.elapsed_ms === 'number') {
      console.log(`Predict time: ${j.elapsed_ms} ms`);
    }
  } catch (err) {
    console.error('[predict] error ->', err);
    if (overlay) overlay.style.display = 'none';
    const msg = err?.name === 'AbortError' ? 'Request timed out. Please try again.' : (err?.message || String(err));
    alert(msg);
  } finally {
    if (predictBtn) predictBtn.disabled = false;
  }
}

predictBtn?.addEventListener('click', (e) => {
  doPredict();
});

function drawTrend() {
  if (!trendCtx || !trendCanvas) return;
  const w = trendCanvas.width, h = trendCanvas.height;
  trendCtx.clearRect(0,0,w,h);
  if (trendData.length < 2) return;
  const xs = trendData.map(p=>p.t);
  const ys = trendData.map(p=>p.y);
  const minx = Math.min(...xs), maxx = Math.max(...xs);
  const miny = Math.min(...ys), maxy = Math.max(...ys);
  const scaleX = (x)=> (w-40) * (x - minx) / Math.max(1, maxx - minx) + 20;
  const scaleY = (y)=> h - ((h-20) * (y - miny) / Math.max(1e-6, maxy - miny)) - 10;
  trendCtx.strokeStyle = '#34d399';
  trendCtx.lineWidth = 2;
  trendCtx.beginPath();
  trendCtx.moveTo(scaleX(trendData[0].t), scaleY(trendData[0].y));
  for (let i=1;i<trendData.length;i++) {
    trendCtx.lineTo(scaleX(trendData[i].t), scaleY(trendData[i].y));
  }
  trendCtx.stroke();
}
