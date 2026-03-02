const $ = (id) => document.getElementById(id);

function serializeForm(f) {
  const data = {};
  const fd = new FormData(f);
  for (const [k, v] of fd.entries()) {
    if (v === '') continue;
    if (!isNaN(v) && v.trim() !== '') {
      data[k] = Number(v);
    } else {
      data[k] = v;
    }
  }
  return data;
}

async function loadMetrics() {
  try {
    const r = await fetch('/api/metrics?refresh=1');
    const j = await r.json();
    const isUntrained = (j.r2 === null);
    $('metric-r2').textContent = isUntrained ? 'Not Trained' : j.r2.toFixed(2);
    $('metric-mae').textContent = isUntrained ? 'Not Trained' : j.mae.toFixed(1);
    $('metric-rmse').textContent = isUntrained ? 'Not Trained' : j.rmse.toFixed(1);
    if ($('metric-rows')) $('metric-rows').textContent = j.rows_trained ?? '—';
    if ($('metric-version')) $('metric-version').textContent = j.model_version ?? '—';
    if ($('metric-model')) $('metric-model').textContent = j.model_type ?? '—';
    if ($('metric-last')) $('metric-last').textContent = j.last_trained_iso ? new Date(j.last_trained_iso).toLocaleString() : '—';
  } catch (_) {
    $('metric-r2').textContent = 'Not Trained';
    $('metric-mae').textContent = 'Not Trained';
    $('metric-rmse').textContent = 'Not Trained';
    if ($('metric-rows')) $('metric-rows').textContent = '—';
    if ($('metric-version')) $('metric-version').textContent = '—';
    if ($('metric-model')) $('metric-model').textContent = '—';
    if ($('metric-last')) $('metric-last').textContent = '—';
  }
}

async function loadImportance() {
  try {
    const r = await fetch('/api/importance');
    const j = await r.json();
    let feats = j.features || [];
    if (!feats.length) {
      feats = [
        { feature: 'latency_ms', importance: 0.4 },
        { feature: 'throughput_mbps', importance: 0.3 },
        { feature: 'network_type', importance: 0.2 },
        { feature: 'latitude', importance: 0.1 },
        { feature: 'longitude', importance: 0.1 },
      ];
    }
    const labels = feats.map(f => f.feature);
    const data = feats.map(f => f.importance);
    const ctx = document.getElementById('importanceChart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Importance',
          data,
          borderWidth: 0,
          backgroundColor: 'rgba(0, 245, 212, 0.6)'
        }]
      },
      options: {
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#7B92B2' }, grid: { color: '#11385f' } },
          y: { ticks: { color: '#E6F1FF' }, grid: { color: '#11385f' } }
        }
      }
    });
    const explain = document.getElementById('explain-text');
    if (explain && feats.length) {
      const top = feats[0];
      explain.textContent = `${top.feature} contributes ${Math.round(top.importance * 100)}% to prediction influence in this sample.`;
    }
  } catch (_) {}
}

function badgeClass(label) {
  if (label === 'Excellent') return 'badge excellent';
  if (label === 'Good') return 'badge good';
  if (label === 'Weak') return 'badge weak';
  return 'badge poor';
}

let trendChart;
function drawGauge(score, label) {
  const canvas = $('signalGauge');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  const center = { x: w/2, y: h/2 };
  const radius = Math.min(w,h)/2 - 10;
  // background circle
  ctx.lineWidth = 14;
  ctx.strokeStyle = '#11385f';
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, Math.PI, 0);
  ctx.stroke();
  // value arc
  const pct = Math.max(0, Math.min(100, score)) / 100;
  const end = Math.PI + (Math.PI * pct);
  ctx.strokeStyle = (score < 40) ? '#FF3B30' : (score < 70 ? '#FF8C00' : '#00D084');
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, Math.PI, end);
  ctx.stroke();
  // text
  ctx.fillStyle = '#E6F1FF';
  ctx.font = '16px Segoe UI, Arial';
  ctx.textAlign = 'center';
  ctx.fillText(`${score}%`, center.x, center.y + 8);
}

let lastPred = 0;
function animateDbm(el, target) {
  const start = performance.now();
  const duration = 500;
  const from = lastPred || target;
  function step(t) {
    const p = Math.min(1, (t - start) / duration);
    const val = from + (target - from) * p;
    el.textContent = `${val.toFixed(1)} dBm`;
    if (p < 1) requestAnimationFrame(step);
  }
  lastPred = target;
  requestAnimationFrame(step);
}

function bindPreset(selectId, inputId) {
  const sel = $(selectId);
  const inp = $(inputId);
  if (!sel || !inp) return;
  const apply = () => {
    const v = sel.value;
    if (v && v !== 'custom') {
      inp.value = v;
      inp.disabled = true;
      inp.classList.add('hidden');
    } else {
      inp.disabled = false;
      inp.classList.remove('hidden');
      if (v !== 'custom') inp.value = '';
    }
  };
  sel.addEventListener('change', apply);
  // Initialize on load
  apply();
}

async function doPredict() {
  const form = document.getElementById('predictForm');
  const data = serializeForm(form);
  const payload = {
    locality: data.locality ?? null,
    latitude: data.latitude ?? null,
    longitude: data.longitude ?? null,
    network_type: data['Network Type'] ?? null,
    throughput_mbps: data.throughput_mbps ?? null,
    latency_ms: data.latency_ms ?? null,
    signal_quality_pct: data.signal_quality_pct ?? null,
    bb60c_dbm: data.bb60c_dbm ?? null,
    srsran_dbm: data.srsran_dbm ?? null,
    bladerf_dbm: data.bladerf_dbm ?? null,
  };
  const btn = $('predictBtn');
  btn.disabled = true;
  try {
    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const t = await r.text();
    let j;
    try {
      j = t ? JSON.parse(t) : {};
    } catch {
      throw new Error((t || 'Non-JSON response from server').slice(0, 200));
    }
    if (!r.ok) throw new Error(j.error || t || 'Prediction failed');
    animateDbm($('pred-dbms'), j.predicted_dbm);
    if ($('pred-ci')) $('pred-ci').textContent = j.ci_dbm ? `± ${j.ci_dbm.toFixed(1)} dBm` : '';
    $('coverage-level').textContent = j.coverage;
    $('coverage-level').className = badgeClass(j.coverage);
    $('health-score').textContent = `${j.health_score}%`;
    drawGauge(j.health_score, j.coverage);
    if ($('inference-time')) $('inference-time').textContent = j.inference_ms ? `Inference: ${j.inference_ms} ms` : '';
    const ul = $('suggestions-list');
    ul.innerHTML = '';
    (j.suggestions || []).forEach(s => {
      const li = document.createElement('li');
      li.textContent = s;
      ul.appendChild(li);
    });
    // trend simulation
    const pred = j.predicted_dbm;
    const trend = [pred - 2, pred - 1, pred, pred + 1, pred - 0.5];
    const tctx = document.getElementById('trendChart')?.getContext('2d');
    if (tctx) {
      if (trendChart) trendChart.destroy();
      trendChart = new Chart(tctx, {
        type: 'line',
        data: {
          labels: ['t-2', 't-1', 't', 't+1', 't+2'],
          datasets: [{
            data: trend,
            borderColor: '#00B4FF',
            backgroundColor: 'rgba(0,180,255,0.2)',
            tension: 0.3,
            fill: true
          }]
        },
        options: {
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { color: '#7B92B2' }, grid: { color: '#11385f' } },
            y: { ticks: { color: '#E6F1FF' }, grid: { color: '#11385f' } }
          }
        }
      });
    }
  } catch (e) {
    $('pred-dbms').textContent = '—';
    $('coverage-level').textContent = '—';
    $('coverage-level').className = 'badge';
    $('health-score').textContent = '—';
    drawGauge(0, 'Weak');
    const ul = $('suggestions-list');
    ul.innerHTML = '';
    const li = document.createElement('li');
    li.textContent = e.message;
    ul.appendChild(li);
  } finally {
    btn.disabled = false;
  }
}

function resetForm() {
  const form = document.getElementById('predictForm');
  form.reset();
  ['locality','latitude','longitude','throughput_mbps','latency_ms','signal_quality_pct','bb60c_dbm','srsran_dbm','bladerf_dbm'].forEach(id=>{
    const el = document.getElementById(id);
    if (el) { el.disabled = false; el.classList.remove('hidden'); el.value = ''; }
  });
  $('pred-dbms').textContent = '—';
  $('coverage-level').textContent = '—';
  $('coverage-level').className = 'badge';
  $('health-score').textContent = '—';
  drawGauge(0, 'Weak');
  const ul = $('suggestions-list');
  ul.innerHTML = '';
}

async function fetchDistinct(column, limit=50) {
  try {
    const r = await fetch(`/api/table/distinct?table=signal_metrics&column=${encodeURIComponent(column)}&limit=${limit}`);
    const j = await r.json();
    return j.values || [];
  } catch { return []; }
}

function setOptions(sel, placeholder, values, extra=[]) {
  const opts = [];
  if (placeholder) opts.push({label: placeholder, value: '', disabled: true, selected: true});
  extra.forEach(v => opts.push({label: v.label ?? v, value: v.value ?? v}));
  values.forEach(v => opts.push({label: String(v), value: String(v)}));
  opts.push({label: 'Custom…', value: 'custom'});
  sel.innerHTML = '';
  opts.forEach(o => {
    const opt = document.createElement('option');
    opt.textContent = o.label;
    opt.value = o.value;
    if (o.disabled) opt.disabled = true;
    if (o.selected) opt.selected = true;
    sel.appendChild(opt);
  });
}

async function loadDropdowns() {
  const localityVals = await fetchDistinct('Locality', 100);
  const ntVals = await fetchDistinct('Network Type', 20);
  const latVals = await fetchDistinct('Latitude', 100);
  const lonVals = await fetchDistinct('Longitude', 100);
  const thrVals = await fetchDistinct('Data Throughput (Mbps)', 50);
  const latMsVals = await fetchDistinct('Latency (ms)', 50);
  const qualVals = await fetchDistinct('Signal Quality (%)', 50);
  const bbVals = await fetchDistinct('BB60C Measurement (dBm)', 50);
  const srVals = await fetchDistinct('srsRAN Measurement (dBm)', 50);
  const blVals = await fetchDistinct('BladeRFxA9 Measurement (dBm)', 50);
  const ls = $('locality_select'); if (ls && localityVals.length) setOptions(ls, 'Select Locality', localityVals);
  const ns = $('network_type'); if (ns && ntVals.length) setOptions(ns, 'Select Network', ntVals, [{label:'All', value:'All'}]);
  const las = $('latitude_select'); if (las && latVals.length) setOptions(las, 'Select Latitude', latVals);
  const los = $('longitude_select'); if (los && lonVals.length) setOptions(los, 'Select Longitude', lonVals);
  const ts = $('throughput_select'); if (ts && thrVals.length) setOptions(ts, 'Select Throughput', thrVals);
  const lsms = $('latency_select'); if (lsms && latMsVals.length) setOptions(lsms, 'Select Latency', latMsVals);
  const qs = $('quality_select'); if (qs && qualVals.length) setOptions(qs, 'Select Signal Quality', qualVals);
  const bbs = $('bb60c_select'); if (bbs && bbVals.length) setOptions(bbs, 'Select BB60C dBm', bbVals);
  const srs = $('srsran_select'); if (srs && srVals.length) setOptions(srs, 'Select srsRAN dBm', srVals);
  const bls = $('bladerf_select'); if (bls && blVals.length) setOptions(bls, 'Select BladeRFxA9 dBm', blVals);
}

document.addEventListener('DOMContentLoaded', () => {
  loadMetrics();
  loadImportance();
  loadDropdowns();
  $('predictBtn').addEventListener('click', doPredict);
  const resetBtn = document.getElementById('resetBtn');
  if (resetBtn) resetBtn.addEventListener('click', resetForm);
  const trainBtn = document.getElementById('trainBtn');
  if (trainBtn) {
    trainBtn.addEventListener('click', async () => {
      trainBtn.disabled = true; trainBtn.textContent = 'Training…';
      try {
        const r = await fetch('/api/train', { method: 'POST' });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || 'Training failed');
        await loadMetrics();
      } catch (e) {
        alert(e.message);
      } finally {
        trainBtn.disabled = false; trainBtn.textContent = 'Train Model';
      }
    });
  }
  const refreshBtn = document.getElementById('refreshKpiBtn');
  if (refreshBtn) refreshBtn.addEventListener('click', loadMetrics);
  bindPreset('locality_select', 'locality');
  bindPreset('latitude_select', 'latitude');
  bindPreset('longitude_select', 'longitude');
  bindPreset('throughput_select', 'throughput_mbps');
  bindPreset('latency_select', 'latency_ms');
  bindPreset('quality_select', 'signal_quality_pct');
  bindPreset('bb60c_select', 'bb60c_dbm');
  bindPreset('srsran_select', 'srsran_dbm');
  bindPreset('bladerf_select', 'bladerf_dbm');
});
