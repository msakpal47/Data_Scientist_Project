async function fetchJSON(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || 'Request failed');
  return j;
}

function renderReport(target, rep) {
  if (!rep) {
    document.getElementById(target).textContent = 'No data';
    return;
  }
  const keys = ['precision','recall','f1-score','support'];
  const labels = ['0','1','macro avg','weighted avg','accuracy'];
  let html = '<table class="metrics"><thead><tr><th>Label</th>';
  keys.forEach(k=>{ html += `<th>${k}</th>`; });
  html += '</tr></thead><tbody>';
  labels.forEach(l=>{
    if (!rep[l] && l !== 'accuracy') return;
    if (l === 'accuracy') {
      html += `<tr><td>accuracy</td><td colspan="3">${(rep.accuracy*100).toFixed(2)}%</td><td>${rep['macro avg']?rep['macro avg'].support:''}</td></tr>`;
      return;
    }
    const row = rep[l];
    html += `<tr><td>${l}</td><td>${(row.precision*100).toFixed(2)}%</td><td>${(row.recall*100).toFixed(2)}%</td><td>${(row['f1-score']*100).toFixed(2)}%</td><td>${row.support}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById(target).innerHTML = html;
}

function renderCM(target, cm) {
  if (!cm) {
    document.getElementById(target).textContent = 'No data';
    return;
  }
  let html = '<table class="cm"><tbody>';
  cm.forEach(row=>{
    html += '<tr>' + row.map(v=>`<td>${v}</td>`).join('') + '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById(target).innerHTML = html;
}

function renderList(target, arr) {
  const el = document.getElementById(target);
  el.innerHTML = (arr||[]).map(w=>`<li>${w}</li>`).join('');
}

function renderBalance(target, bal) {
  const el = document.getElementById(target);
  if (!bal) { el.textContent = 'No data'; return; }
  const total = (bal.pos||0)+(bal.neg||0);
  const posPct = total? ((bal.pos/total)*100).toFixed(1): '0.0';
  const negPct = total? ((bal.neg/total)*100).toFixed(1): '0.0';
  el.innerHTML = `<div>Positive: ${bal.pos} (${posPct}%) &nbsp; Negative: ${bal.neg} (${negPct}%) &nbsp; Total: ${total}</div>`;
}

function drawROC(target, roc) {
  const el = document.getElementById(target);
  if (!roc || !roc.fpr || !roc.tpr) { el.textContent = 'No data'; return; }
  const w = 220, h = 220, p = 20;
  const pts = roc.fpr.map((x,i)=>[x, roc.tpr[i]]);
  const path = pts.map(([x,y])=>`${p + x*(w-2*p)},${h - (p + y*(h-2*p))}`).join(' ');
  el.innerHTML = `<svg width="${w}" height="${h}">
    <rect x="${p}" y="${p}" width="${w-2*p}" height="${h-2*p}" fill="none" stroke="#1c2230"/>
    <polyline points="${p},${h-p} ${w-p},${p}" stroke="#555" fill="none"/>
    <polyline points="${path}" stroke="#4f8cff" fill="none"/>
    <text x="${p}" y="14" fill="#9aa0a6">AUC ${(roc.auc||0).toFixed(3)}</text>
  </svg>`;
}

function drawCAL(target, cal) {
  const el = document.getElementById(target);
  if (!cal || !cal.prob_pred_mean || !cal.frac_pos) { el.textContent = 'No data'; return; }
  const w = 220, h = 220, p = 20;
  const pts = cal.prob_pred_mean.map((x,i)=>[x, cal.frac_pos[i]]);
  const circles = pts.map(([x,y])=>`<circle cx="${p + x*(w-2*p)}" cy="${h - (p + y*(h-2*p))}" r="3" fill="#1db954"/>`).join('');
  el.innerHTML = `<svg width="${w}" height="${h}">
    <rect x="${p}" y="${p}" width="${w-2*p}" height="${h-2*p}" fill="none" stroke="#1c2230"/>
    <polyline points="${p},${h-p} ${w-p},${p}" stroke="#555" fill="none"/>
    ${circles}
  </svg>`;
}

function renderCondDist(target, dist) {
  const el = document.getElementById(target);
  if (!dist || !dist.length) { el.textContent = 'No data'; return; }
  let html = '<table class="metrics"><thead><tr><th>Condition</th><th>Pos</th><th>Neg</th><th>Total</th><th>Pos Ratio</th></tr></thead><tbody>';
  dist.forEach(d=>{
    html += `<tr><td>${d.condition}</td><td>${d.pos}</td><td>${d.neg}</td><td>${d.total}</td><td>${(d.pos_ratio*100).toFixed(1)}%</td></tr>`;
  });
  html += '</tbody></table>';
  el.innerHTML = html;
}

(async function init() {
  try {
    let data;
    try {
      data = await fetchJSON('/metrics_data');
    } catch (e) {
      data = await fetchJSON('/static/metrics.json');
    }
    const s = data.meta;
    document.getElementById('summary').innerHTML = `<div>Model: ${s.model}</div><div>Vectorizer: ${s.vectorizer} ngram ${s.ngram_range.join('-')} max_features ${s.max_features}</div><div>Splits: train ${data.splits.train} | val ${data.splits.val} | prod ${data.splits.prod}</div><div>Version: ${s.model_version || ''}</div><div>Training Date: ${s.training_date || ''}</div>`;
    renderReport('valMetrics', data.validation?data.validation.report:null);
    renderCM('valCM', data.validation?data.validation.confusion_matrix:null);
    renderBalance('valBalance', data.validation?data.validation.class_balance:null);
    drawROC('valROC', data.validation?data.validation.roc:null);
    drawCAL('valCAL', data.validation?data.validation.calibration:null);
    renderReport('prodMetrics', data.production?data.production.report:null);
    renderCM('prodCM', data.production?data.production.confusion_matrix:null);
    renderBalance('prodBalance', data.production?data.production.class_balance:null);
    drawROC('prodROC', data.production?data.production.roc:null);
    drawCAL('prodCAL', data.production?data.production.calibration:null);
    renderList('topPos', data.feature_importance.top_positive);
    renderList('topNeg', data.feature_importance.top_negative);
    renderCondDist('condDist', data.sentiment_distribution);
  } catch (e) {
    document.getElementById('summary').textContent = e.message;
  }
})(); 
