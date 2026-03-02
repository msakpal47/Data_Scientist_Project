const reviewEl = document.getElementById('review');
const predictBtn = document.getElementById('predictBtn');
const resultEl = document.getElementById('result');
const recBtn = document.getElementById('recBtn');
const condEl = document.getElementById('condition');
const topkEl = document.getElementById('topk');
const minrevEl = document.getElementById('minreviews');
const recResultEl = document.getElementById('recResult');

async function loadConditions() {
  if (!condEl || condEl.tagName !== 'SELECT') return;
  try {
    async function load(url) {
      const r = await fetch(url);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || `Failed ${url}`);
      return j;
    }
    let data;
    try {
      data = await load('/static/conditions.json');
    } catch (_) {
      data = await load('/conditions');
    }
    condEl.innerHTML = '<option value="">Select a condition...</option>';
    (data.results || []).forEach(item => {
      const opt = document.createElement('option');
      opt.value = item.condition;
      opt.textContent = `${item.condition} (${item.count})`;
      condEl.appendChild(opt);
    });
  } catch (e) {
    condEl.innerHTML = '<option value="">Failed to load conditions</option>';
  }
}

async function predict() {
  const text = reviewEl.value.trim();
  if (!text) {
    resultEl.textContent = 'Please enter a review.';
    return;
  }
  predictBtn.disabled = true;
  resultEl.textContent = 'Predicting...';
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');
    const badgeClass = data.label === 1 ? 'pos' : 'neg';
    const labelText = data.label === 1 ? 'Positive' : 'Negative';
    const confText = typeof data.confidence === 'number' ? `confidence ${(data.confidence*100).toFixed(1)}%` : '';
    resultEl.innerHTML = `<span class="badge ${badgeClass}">${labelText}</span><span class="conf">${confText}</span>`;
  } catch (err) {
    resultEl.textContent = err.message;
  } finally {
    predictBtn.disabled = false;
  }
}

async function recommend() {
  const condition = (condEl?.value || '').trim();
  const top_k = parseInt(topkEl?.value || '3', 10);
  const min_reviews = parseInt(minrevEl?.value || '5', 10);
  if (!condition) {
    recResultEl.textContent = 'Please enter a condition.';
    return;
  }
  if (recBtn) recBtn.disabled = true;
  recResultEl.textContent = 'Recommending...';
  try {
    const res = await fetch('/recommend_drug', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ condition, top_k, min_reviews })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Recommendation failed');
    if (!data.results || data.results.length === 0) {
      recResultEl.textContent = 'No recommendations found for this condition.';
      return;
    }
    const items = data.results.map((r, idx) => {
      const pct = (r.score * 100).toFixed(1);
      return `<div>${idx+1}. ${r.drugName} — score ${pct} (avg rating ${r.avg_rating.toFixed(2)}, confidence ${ (r.avg_sentiment_prob*100).toFixed(1)}%, useful ${r.avg_useful_norm.toFixed(2)})</div>`;
    }).join('');
    recResultEl.innerHTML = items;
  } catch (err) {
    recResultEl.textContent = err.message;
  } finally {
    if (recBtn) recBtn.disabled = false;
  }
}

predictBtn?.addEventListener('click', predict);
recBtn?.addEventListener('click', recommend);
loadConditions();
