const state = {
  connected: false,
  tableName: null,
  rowCount: 0,
  splits: null,
  columns: [],
  numericOnly: true,
  includeAdmin: false,
  fastMode: true,
  modelReady: false,
  oversample: false,
  evalOffset: 0,
  evalCount: 0,
};

function now() {
  const d = new Date();
  return d.toLocaleTimeString();
}

function addMsg(who, txt) {
  const log = document.getElementById('chatlog');
  const el = document.createElement('div');
  el.className = 'msg';
  el.innerHTML = `
    <div class="meta"><span>${who}</span><span>${now()}</span></div>
    <div class="txt"></div>
  `;
  el.querySelector('.txt').textContent = txt;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

function clearChat() {
  document.getElementById('chatlog').innerHTML = '';
}

function setConnected(isConnected) {
  state.connected = isConnected;
  const dot = document.getElementById('connDot');
  const txt = document.getElementById('connText');
  dot.className = isConnected ? 'dot ok' : 'dot';
  txt.textContent = isConnected ? 'Connected' : 'Not connected';
}

function showPanel(which) {
  document.getElementById('panelConnect').style.display = which === 'connect' ? '' : 'none';
  document.getElementById('panelTrain').style.display = which === 'train' ? '' : 'none';
  document.getElementById('panelInfer').style.display = which === 'infer' ? '' : 'none';
  document.getElementById('tabConnect').classList.toggle('active', which === 'connect');
  document.getElementById('tabTrain').classList.toggle('active', which === 'train');
  document.getElementById('tabInfer').classList.toggle('active', which === 'infer');
}

function showWorkspace(which) {
  document.getElementById('wsData').style.display = which === 'data' ? '' : 'none';
  document.getElementById('wsMetrics').style.display = which === 'metrics' ? '' : 'none';
  document.getElementById('wsChat').style.display = which === 'chat' ? '' : 'none';
  document.getElementById('tabData').classList.toggle('active', which === 'data');
  document.getElementById('tabMetrics').classList.toggle('active', which === 'metrics');
  document.getElementById('tabChat').classList.toggle('active', which === 'chat');
}

function setConfig(cfg) {
  document.getElementById('dbPath').value = cfg.db_path || '';
  document.getElementById('ddPath').value = cfg.dd_path || '';
  document.getElementById('tableName').textContent = cfg.table_name || '—';
}

function formatPct(v) {
  if (v === null || v === undefined) return '—';
  return (Math.round(v * 1000) / 10).toFixed(1) + '%';
}

function renderPreview(columns, rows) {
  const wrap = document.getElementById('previewWrap');
  if (!columns || columns.length === 0) {
    wrap.innerHTML = '<div class="small">No preview loaded.</div>';
    return;
  }
  let html = '<table><thead><tr>';
  for (const c of columns) html += `<th>${c}</th>`;
  html += '</tr></thead><tbody>';
  for (const r of rows) {
    html += '<tr>';
    for (const c of columns) {
      const val = r[c];
      html += `<td title="${String(val ?? '')}">${val ?? ''}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

function renderMetrics(m) {
  const cards = document.getElementById('metricCards');
  cards.innerHTML = '';
  const items = [
    ['Accuracy', formatPct(m.accuracy)],
    ['Precision', formatPct(m.precision)],
    ['Recall', formatPct(m.recall)],
    ['F1', formatPct(m.f1)],
    ['ROC AUC', formatPct(m.roc_auc)],
  ];
  for (const [k, v] of items) {
    const el = document.createElement('div');
    el.className = 'card';
    el.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    cards.appendChild(el);
  }

  const cm = m.confusion_matrix || [[0, 0], [0, 0]];
  const cmWrap = document.getElementById('cmWrap');
  if (cmWrap) cmWrap.innerHTML = `
    <table style="min-width: 420px">
      <thead>
        <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
      </thead>
      <tbody>
        <tr><td>Actual 0</td><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr>
        <tr><td>Actual 1</td><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr>
      </tbody>
    </table>
  `;

  const dist = m.class_dist_eval || { 0: 0, 1: 0 };
  const distWrap = document.getElementById('distWrap');
  if (distWrap) distWrap.innerHTML = `
    <table style="min-width: 300px">
      <thead><tr><th>Label</th><th>Count</th></tr></thead>
      <tbody>
        <tr><td>0</td><td>${dist['0'] ?? 0}</td></tr>
        <tr><td>1</td><td>${dist['1'] ?? 0}</td></tr>
      </tbody>
    </table>
  `;

  const pr = m.pr_curve;
  const prWrap = document.getElementById('prWrap');
  if (prWrap) {
    if (!pr || !pr.precision || !pr.recall || pr.precision.length < 2) {
      prWrap.innerHTML = '<div class="small">PR curve unavailable (no positives in eval).</div>';
    } else {
      const W = 480, H = 300, pad = 24;
      const ps = pr.precision, rs = pr.recall;
      const toX = r => pad + r * (W - 2 * pad);
      const toY = p => H - pad - p * (H - 2 * pad);
      let path = '';
      for (let i = 0; i < ps.length; i++) {
        const x = toX(rs[i]);
        const y = toY(ps[i]);
        path += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
      }
      prWrap.innerHTML = `
        <svg width="${W}" height="${H}">
          <rect x="0" y="0" width="${W}" height="${H}" fill="transparent" />
          <path d="${path}" stroke="url(#grad)" stroke-width="2" fill="none"/>
          <defs>
            <linearGradient id="grad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stop-color="#7c3aed"/>
              <stop offset="100%" stop-color="#22c55e"/>
            </linearGradient>
          </defs>
          <line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="rgba(255,255,255,0.2)"/>
          <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}" stroke="rgba(255,255,255,0.2)"/>
          <text x="${W/2}" y="${H-6}" fill="rgba(255,255,255,0.7)" font-size="12" text-anchor="middle">Recall</text>
          <text x="10" y="${H/2}" fill="rgba(255,255,255,0.7)" font-size="12" transform="rotate(-90 10 ${H/2})">Precision</text>
        </svg>
      `;
    }
  }

  const featWrap = document.getElementById('featWrap');
  const feats = m.feature_ranking || [];
  if (!feats || feats.length === 0) {
    featWrap.innerHTML = '<div class="small">No feature ranking available for this model.</div>';
    return;
  }
  const maxImp = Math.max(...feats.map(x => x.importance || 0), 1e-9);
  let html = '';
  for (const f of feats.slice(0, 10)) {
    const pct = Math.max(0, Math.min(100, (f.importance / maxImp) * 100));
    html += `
      <div class="card" style="margin-bottom:10px">
        <div class="k">${f.feature}</div>
        <div style="display:flex; justify-content:space-between; gap:10px; align-items:center; margin-top:6px">
          <div class="bar" style="flex:1"><div style="width:${pct}%"></div></div>
          <div style="font-family: var(--mono); font-size: 12px; color: rgba(255,255,255,0.80)">${(Math.round(f.importance * 10000) / 10000).toFixed(4)}</div>
        </div>
      </div>
    `;
  }
  featWrap.innerHTML = html;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const ct = res.headers.get('content-type') || '';
  const data = ct.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok) {
    const msg = (data && data.error) ? data.error : (typeof data === 'string' ? data : 'Request failed');
    throw new Error(msg);
  }
  return data;
}

async function loadConfig() {
  const cfg = await api('/api/config');
  setConfig(cfg);
  document.getElementById('tableName').textContent = cfg.table_name || '—';
  addMsg('system', 'Loaded defaults. Click Connect to start.');
}

async function connect() {
  const btn = document.getElementById('btnConnect');
  btn.disabled = true;
  try {
    addMsg('user', 'Connect');
    const payload = {
      db_path: document.getElementById('dbPath').value,
      dd_path: document.getElementById('ddPath').value,
    };
    const out = await api('/api/connect', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
    setConnected(true);
    document.getElementById('tableName').textContent = out.table_name;
    document.getElementById('rowCount').textContent = String(out.total_rows);
    state.tableName = out.table_name;
    state.rowCount = out.total_rows;
    state.splits = out.splits;
    addMsg('system', `Connected. Table=${out.table_name}, total_rows=${out.total_rows}`);
    await refreshPreview();
    await refreshDictionary();
    await refreshColumns();
    document.getElementById('inferHint').textContent = `Live rows: ${out.splits.live_count} (offset ${out.splits.live_offset})`;
    state.evalOffset = out.splits.train_count || 0;
    state.evalCount = out.splits.eval_count || 0;
    const eo = document.getElementById('evalOffset');
    const ec = document.getElementById('evalCount');
    if (eo) eo.value = String(state.evalOffset);
    if (ec) ec.value = String(state.evalCount);
  } catch (e) {
    setConnected(false);
    addMsg('system', `Connect failed: ${e.message}`);
    alert(e.message);
  } finally {
    btn.disabled = false;
  }
}

async function refreshDictionary() {
  try {
    const txt = await api('/api/data_dictionary');
    document.getElementById('ddText').value = txt || '';
  } catch (e) {
    document.getElementById('ddText').value = '';
  }
}

async function refreshPreview() {
  try {
    const out = await api('/api/preview?limit=50');
    state.columns = out.columns || [];
    renderPreview(out.columns || [], out.rows || []);
    addMsg('system', 'Loaded data preview.');
  } catch (e) {
    renderPreview([], []);
    addMsg('system', `Preview failed: ${e.message}`);
  }
}

function selectedValues(selectEl) {
  return Array.from(selectEl.selectedOptions).map(o => o.value);
}

function fillSelect(selectEl, options, defaultValue) {
  selectEl.innerHTML = '';
  for (const opt of options) {
    const o = document.createElement('option');
    o.value = opt;
    o.textContent = opt;
    if (defaultValue && opt === defaultValue) o.selected = true;
    selectEl.appendChild(o);
  }
}

async function refreshColumns() {
  const out = await api('/api/columns');
  const cols = out.columns || [];
  fillSelect(document.getElementById('targetSelect'), cols, out.default_target || cols[0]);
  await refreshFeatures();
}

async function refreshFeatures() {
  const target = document.getElementById('targetSelect').value;
  const out = await api(`/api/features?target=${encodeURIComponent(target)}&numeric_only=${state.numericOnly ? 1 : 0}&include_admin=${state.includeAdmin ? 1 : 0}`);
  const feats = out.features || [];
  const sel = document.getElementById('featureSelect');
  sel.innerHTML = '';
  const defaults = out.default_features || [];
  for (const f of feats) {
    const o = document.createElement('option');
    o.value = f;
    o.textContent = f;
    if (defaults.includes(f)) o.selected = true;
    sel.appendChild(o);
  }
  addMsg('system', `Loaded feature list (${feats.length}).`);
}

function toggleNumeric() {
  state.numericOnly = !state.numericOnly;
  document.getElementById('toggleNumeric').textContent = `Numeric-only: ${state.numericOnly ? 'ON' : 'OFF'}`;
  refreshFeatures();
}

function toggleAdmin() {
  state.includeAdmin = !state.includeAdmin;
  document.getElementById('toggleAdmin').textContent = `Admin fields: ${state.includeAdmin ? 'ON' : 'OFF'}`;
  refreshFeatures();
}

function toggleFast() {
  state.fastMode = !state.fastMode;
  document.getElementById('toggleFast').textContent = `Fast mode: ${state.fastMode ? 'ON' : 'OFF'}`;
  document.getElementById('toggleFast').classList.toggle('good', state.fastMode);
}

function toggleOversample() {
  state.oversample = !state.oversample;
  document.getElementById('toggleOversample').textContent = `Oversample: ${state.oversample ? 'ON' : 'OFF'}`;
}

async function train() {
  if (!state.connected) {
    alert('Connect first.');
    return;
  }
  const btn = document.getElementById('btnTrain');
  btn.disabled = true;
  try {
    showWorkspace('metrics');
    addMsg('user', 'Train model');
    const payload = {
      target: document.getElementById('targetSelect').value,
      features: selectedValues(document.getElementById('featureSelect')),
      fast_mode: state.fastMode,
      chunk_size: parseInt(document.getElementById('chunkSelect').value, 10),
      eval_offset: parseInt(document.getElementById('evalOffset').value || String(state.evalOffset), 10),
      eval_count: parseInt(document.getElementById('evalCount').value || String(state.evalCount), 10),
      oversample: state.oversample,
    };
    if (!payload.features || payload.features.length === 0) {
      alert('Select at least one feature.');
      return;
    }
    addMsg('system', `Training started (fast_mode=${payload.fast_mode}, chunk=${payload.chunk_size})...`);
    const out = await api('/api/train', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
    renderMetrics(out.metrics);
    state.modelReady = true;
    document.getElementById('btnDownload').disabled = false;
    addMsg('system', 'Training completed. Metrics updated.');
  } catch (e) {
    addMsg('system', `Training failed: ${e.message}`);
    alert(e.message);
  } finally {
    btn.disabled = false;
  }
}

function downloadModel() {
  window.location.href = '/download/model';
}

async function loadModel() {
  const btn = document.getElementById('btnLoadModel');
  if (btn) btn.disabled = true;
  try {
    addMsg('user', 'Load saved model');
    const out = await api('/api/load_model', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({}),
    });
    state.modelReady = true;
    addMsg('system', `Model loaded. Features=${out.feature_count}`);
  } catch (e) {
    addMsg('system', `Load failed: ${e.message}`);
    alert(e.message);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function predict() {
  if (!state.modelReady) {
    alert('Train a model first.');
    return;
  }
  const btn = document.getElementById('btnPredict');
  btn.disabled = true;
  try {
    addMsg('user', 'Predict');
    const idx = parseInt(document.getElementById('rowIndex').value || '0', 10);
    const out = await api(`/api/predict?index=${idx}`);
    showWorkspace('chat');
    addMsg('system', `predicted_fault=${out.predicted_fault}, probability=${out.probability}`);
    addMsg('system', JSON.stringify(out.row, null, 2));
  } catch (e) {
    addMsg('system', `Predict failed: ${e.message}`);
    alert(e.message);
  } finally {
    btn.disabled = false;
  }
}

async function simulateLive() {
  if (!state.modelReady) {
    alert('Train a model first or load a model.');
    return;
  }
  const btn = document.getElementById('btnSimulate');
  if (btn) btn.disabled = true;
  try {
    addMsg('user', 'Simulate live');
    const chunkSel = document.getElementById('chunkSelect');
    const chunk = chunkSel ? parseInt(chunkSel.value || '50000', 10) : 50000;
    const out = await api(`/api/infer_summary?chunk_size=${chunk}`);
    showWorkspace('chat');
    addMsg('system', `Live summary: total_rows=${out.total_rows}, predicted_faults=${out.predicted_faults}, fault_rate=${formatPct(out.fault_rate)}`);
    for (const s of out.samples || []) {
      addMsg('system', `sample idx=${s.index} predicted_fault=${s.predicted_fault} prob=${s.probability}`);
    }
  } catch (e) {
    addMsg('system', `Simulation failed: ${e.message}`);
    alert(e.message);
  } finally {
    if (btn) btn.disabled = false;
  }
}

window.addMsg = addMsg;
window.clearChat = clearChat;
window.showPanel = showPanel;
window.showWorkspace = showWorkspace;
window.connect = connect;
window.refreshPreview = refreshPreview;
window.toggleNumeric = toggleNumeric;
window.toggleAdmin = toggleAdmin;
window.toggleFast = toggleFast;
window.train = train;
window.downloadModel = downloadModel;
window.loadModel = loadModel;
window.predict = predict;
window.simulateLive = simulateLive;

document.getElementById('targetSelect').addEventListener('change', refreshFeatures);
loadConfig();
