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

async function loadSummary() {
  const res = await fetch("/api/summary");
  const data = await res.json();
  const nonFraud = data.non_fraud;
  const fraud = data.fraud;
  const rate = data.fraud_rate;
  const flagged = data.flagged ?? 0;
  const flaggedRate = data.flagged_rate ?? 0;
  const elNon = document.getElementById("metric-nonfraud");
  const elFraud = document.getElementById("metric-fraud");
  const elRate = document.getElementById("metric-rate");
  const elFlagged = document.getElementById("metric-flagged");
  animateNumber(elNon, nonFraud, v => Math.round(v).toLocaleString(), 700);
  animateNumber(elFraud, fraud, v => Math.round(v).toLocaleString(), 700);
  animateNumber(elRate, rate * 100, v => v.toFixed(3) + "%", 900);
  if (elFlagged) animateNumber(elFlagged, flagged, v => Math.round(v).toLocaleString(), 700);
}

function renderTable(data) {
  const head = document.getElementById("tx-head");
  const body = document.getElementById("tx-body");
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

async function loadTransactions(filter) {
  const page = Number(document.getElementById("page")?.value || 1);
  const limit = Number(document.getElementById("limit")?.value || 500);
  try {
    const res = await fetch(`/api/transactions?filter=${encodeURIComponent(filter)}&page=${page}&limit=${limit}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderTable(data);
    const total = data.total_rows ?? 0;
    const totalEl = document.getElementById("total_rows");
    if (totalEl) totalEl.textContent = `Total rows: ${total.toLocaleString()}`;
  } catch (err) {
    const head = document.getElementById("tx-head");
    const body = document.getElementById("tx-body");
    head.innerHTML = "<tr><th>Error loading rows</th></tr>";
    body.innerHTML = `<tr><td>${String(err)}</td></tr>`;
  }
}

function wireFilterChips() {
  const container = document.getElementById("filter-chips");
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
      output.innerHTML = `
        <div class="prediction-label">Model decision</div>
        <div class="prediction-prob">Fraud probability: <strong>${pct}%</strong></div>
        <div class="prediction-tag ${tagClass}">
          <span class="tiny-dot ${tagClass}"></span>
          <span>${tagText}</span>
        </div>
      `;
    } catch (err) {
      output.innerHTML = `
        <div class="prediction-label">Prediction failed</div>
        <div class="prediction-prob">Error: ${String(err)}</div>
      `;
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  loadSummary();
  loadTransactions("all");
  wireFilterChips();
  wirePredictForm();
  const reload = document.getElementById("reload");
  if (reload) {
    reload.addEventListener("click", () => {
      const active = document.querySelector(".filter-chip.active");
      const filter = active?.dataset?.filter || "all";
      loadTransactions(filter);
    });
  }
  updateDownloadLink("all");
});

function updateDownloadLink(filter) {
  const link = document.getElementById("download");
  if (link) {
    link.href = `/api/export?filter=${encodeURIComponent(filter)}`;
  }
}
