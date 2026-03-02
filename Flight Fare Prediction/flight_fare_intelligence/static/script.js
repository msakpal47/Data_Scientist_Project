const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const trendCanvas = document.getElementById("trendChart");
const fiCanvas = document.getElementById("fiChart");
const fiNote = document.getElementById("fiNote");
const topFactors = document.getElementById("topFactors");
const recommendBox = document.getElementById("recommendBox");
const elasticitySel = document.getElementById("elasticity");
const adjRange = document.getElementById("adjPct");
const adjVal = document.getElementById("adjVal");
const revCanvas = document.getElementById("revChart");
const revSummary = document.getElementById("revSummary");
const modelReport = document.getElementById("modelReport");
const positioningNote = document.getElementById("positioningNote");
const modelMeta = document.getElementById("modelMeta");

let trendChart, fiChart, revChart;
let baseFare = null;
let modelMetrics = null;

function showResult(text) {
  resultBox.textContent = text;
  resultBox.classList.remove("hidden");
}

function payloadFromUI() {
  const route = document.getElementById("route").value;
  const [srcCode, dstCode] = route.split("-");
  const codeToCity = {
    DEL: "Delhi",
    BOM: "Mumbai",
    BLR: "Bangalore",
    HYD: "Hyderabad",
    MAA: "Chennai",
    CCU: "Kolkata",
    GOI: "Goa",
  };
  const source_city = codeToCity[srcCode] || srcCode;
  const destination_city = codeToCity[dstCode] || dstCode;
  const travel_class = document.getElementById("travel_class").value;
  const booking_lead_days = Number(document.getElementById("booking_lead_days").value);
  const carrier = document.getElementById("carrier").value;
  const stops = Number(document.getElementById("stops").value);
  const duration_minutes = Number(document.getElementById("duration_minutes").value);
  const departure_time = document.getElementById("departure_time").value;
  const arrival_time = document.getElementById("arrival_time").value;
  return {
    travel_class,
    booking_lead_days,
    route,
    carrier,
    stops,
    duration_minutes,
    airline: carrier,
    source_city,
    destination_city,
    departure_time,
    arrival_time,
    duration: duration_minutes,
    days_left: booking_lead_days,
    class_type: travel_class,
  };
}

async function fetchJSON(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return resp.json();
}

function renderTrend(days, prices) {
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(trendCanvas, {
    type: "line",
    data: {
      labels: days,
      datasets: [
        {
          label: "Fare",
          data: prices,
          borderColor: "#36bffa",
          backgroundColor: "rgba(54,191,250,0.2)",
          tension: 0.2,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Lead Days" } },
        y: { title: { display: true, text: "Fare (₹)" } },
      },
    },
  });
}

function renderFeatureImportance(items) {
  if (fiChart) fiChart.destroy();
  const labels = items.map((d) => d.feature);
  const values = items.map((d) => d.value);
  fiChart = new Chart(fiCanvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Contribution (₹)",
          data: values,
          backgroundColor: values.map((v) => (v >= 0 ? "#22c55e" : "#ef4444")),
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        y: { title: { display: true, text: "Contribution (₹)" } },
      },
    },
  });
}

function renderTopFactors(items) {
  topFactors.innerHTML = "";
  const top3 = items.slice(0, 3);
  top3.forEach((it) => {
    const li = document.createElement("li");
    const sign = it.value >= 0 ? "+" : "−";
    li.textContent = `${it.feature}: ${sign}${Math.abs(it.value).toFixed(2)}`;
    topFactors.appendChild(li);
  });
}

function renderRevenueSimulator(basePrice) {
  // Build curve for -30%..+30% with selected elasticity
  const elasticity = Number(elasticitySel.value);
  const pctRange = Array.from({ length: 61 }, (_, i) => i - 30);
  const baseDemand = 100;
  const revenues = pctRange.map((p) => {
    const pMult = 1 + p / 100;
    const newPrice = basePrice * pMult;
    const demand = baseDemand * (1 + elasticity * (p / 100));
    return Math.max(0, newPrice * Math.max(0, demand));
  });
  if (revChart) revChart.destroy();
  revChart = new Chart(revCanvas, {
    type: "line",
    data: {
      labels: pctRange,
      datasets: [
        {
          label: "Revenue",
          data: revenues,
          borderColor: "#a78bfa",
          backgroundColor: "rgba(167,139,250,0.2)",
          tension: 0.2,
        },
      ],
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Price Adjustment (%)" } },
        y: { title: { display: true, text: "Projected Revenue" } },
      },
    },
  });

  function updateSummary(adjustPct) {
    const pMult = 1 + adjustPct / 100;
    const newPrice = basePrice * pMult;
    const demand = 100 * (1 + elasticity * (adjustPct / 100));
    const revenue = Math.max(0, newPrice * Math.max(0, demand));
    revSummary.textContent = `Base Fare ₹${basePrice.toFixed(2)} → Adjust ${adjustPct}% → Price ₹${newPrice.toFixed(
      2
    )}, Demand ${Math.max(0, demand).toFixed(0)}, Revenue ${revenue.toFixed(0)}`;
  }
  updateSummary(Number(adjRange.value));
  adjRange.oninput = () => {
    adjVal.textContent = `${adjRange.value}%`;
    updateSummary(Number(adjRange.value));
  };
  elasticitySel.onchange = () => {
    renderRevenueSimulator(basePrice);
  };
}

async function loadModelMetrics() {
  try {
    const resp = await fetch("/model_metrics");
    const data = await resp.json();
    if (!data.models_loaded) {
      modelReport.textContent = "Models not loaded";
      return;
    }
    modelMetrics = data;
    const eR2 = data.economy?.r2 ?? null;
    const eMAE = data.economy?.mae ?? null;
    const bR2 = data.business?.r2 ?? null;
    const bMAE = data.business?.mae ?? null;
    modelReport.textContent = `Economy R²: ${eR2?.toFixed ? eR2.toFixed(2) : eR2}  |  MAE: ₹${eMAE?.toFixed ? eMAE.toFixed(0) : eMAE}    •    Business R²: ${bR2?.toFixed ? bR2.toFixed(2) : bR2}  |  MAE: ₹${bMAE?.toFixed ? bMAE.toFixed(0) : bMAE}`;
    positioningNote.textContent = "Portfolio demo: low MAE indicates stable absolute error; moderate R² reflects higher target variance.";
    const em = data.economy || {};
    const bm = data.business || {};
    const ver = em.model_version || bm.model_version || "v1";
    const trained = em.trained_at || bm.trained_at || "";
    const cv = em.cv_r2_mean ?? bm.cv_r2_mean;
    const cvTxt = cv !== undefined && cv !== null ? ` | CV R²: ${cv.toFixed ? cv.toFixed(2) : cv}` : "";
    const rows = em.dataset_rows ?? bm.dataset_rows ?? "";
    modelMeta.textContent = `Model ${ver} | Trained: ${trained}${cvTxt} | Dataset rows: ${rows}`;
  } catch (_) {
    modelReport.textContent = "Model metrics unavailable";
  }
}

loadModelMetrics();

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const payload = payloadFromUI();
  try {
    const pred = await fetchJSON("/predict", payload);
    if (typeof pred.predicted_fare !== "number") throw new Error("Bad response");
    baseFare = pred.predicted_fare;
    const low = pred.ci_low ?? baseFare * 0.9;
    const high = pred.ci_high ?? baseFare * 1.1;
    const conf = pred.confidence ?? 90;
    let approxPct = null;
    if (modelMetrics) {
      const cls = payload.class_type?.toLowerCase?.() || "economy";
      const mae = cls === "economy" ? modelMetrics.economy?.mae : modelMetrics.business?.mae;
      if (mae && baseFare) approxPct = (100 * mae / baseFare);
    }
    const approxText = approxPct ? `  |  Expected error ≈ ±${approxPct.toFixed(1)}%` : "";
    showResult(`Estimated Fare: ₹ ${baseFare.toFixed(2)}  |  Range: ₹${low.toFixed(2)} – ₹${high.toFixed(2)}  |  Confidence: ${conf}%${approxText}`);

    // Trend
    const trend = await fetchJSON("/price_trend", { ...payload, max_days: 120, step: 5 });
    renderTrend(trend.days, trend.prices);

    // Explain
    const exp = await fetchJSON("/explain", payload);
    renderFeatureImportance(exp.contributions.slice(0, 8));
    if (exp.method === "shap") {
      fiNote.textContent = "SHAP local explanation";
      renderTopFactors(exp.contributions);
    } else if (exp.method === "model_importances") {
      fiNote.textContent = "Global feature importances from model";
      renderTopFactors(exp.contributions);
    } else {
      fiNote.textContent = "Approximate contributions shown (model not loaded)";
      renderTopFactors(exp.contributions);
    }

    // Recommendation
    const rec = await fetchJSON("/recommend", { ...payload, horizon: 90, step: 3 });
    recommendBox.textContent = `${rec.action}: current ₹${rec.current_price.toFixed(
      2
    )}. Best around lead day ${rec.recommended_lead_days} at ₹${rec.recommended_price.toFixed(
      2
    )}. Potential savings ₹${rec.potential_savings.toFixed(2)}.`;

    // Revenue sim
    renderRevenueSimulator(baseFare);
  } catch (err) {
    showResult("Prediction failed");
  }
});
