const reviewEl = document.getElementById('review');
const predictBtn = document.getElementById('predictBtn');
const resultEl = document.getElementById('result');

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

predictBtn.addEventListener('click', predict);
