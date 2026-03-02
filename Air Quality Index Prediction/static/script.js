function setCategory(aqi) {
  if (aqi <= 50) return ["Good", "Air quality is satisfactory."];
  if (aqi <= 100) return ["Moderate", "Acceptable air quality."];
  if (aqi <= 150) return ["Unhealthy (Sensitive)", "Sensitive people should limit exposure."];
  if (aqi <= 200) return ["Unhealthy", "Health effects possible."];
  if (aqi <= 300) return ["Very Unhealthy", "Health alert issued."];
  return ["Hazardous", "Serious health risk."];
}

(function setupChartDefaults() {
  if (window.Chart) {
    Chart.defaults.color = "rgba(255,255,255,0.9)";
    Chart.defaults.borderColor = "rgba(255,255,255,0.15)";
    Chart.defaults.font.family = "Segoe UI, Arial, sans-serif";
  }
  if (window.ChartDataLabels) {
    Chart.register(window.ChartDataLabels);
  }
})();

function applyCategoryClass(el, category) {
  if (!el) return;
  const classes = ["cat-good","cat-moderate","cat-sensitive","cat-unhealthy","cat-very","cat-hazardous","cat-neutral"];
  el.classList.remove(...classes);
  const map = {
    "Good": "cat-good",
    "Moderate": "cat-moderate",
    "Unhealthy (Sensitive)": "cat-sensitive",
    "Unhealthy": "cat-unhealthy",
    "Very Unhealthy": "cat-very",
    "Hazardous": "cat-hazardous"
  };
  el.classList.add(map[category] || "cat-neutral");
}

fetch("/trend")
  .then((res) => res.json())
  .then((data) => {
    const canvas = document.getElementById("trendChart");
    if (canvas) {
      const ctx = canvas.getContext("2d");
      const grad = ctx.createLinearGradient(0, 0, 0, canvas.height || 280);
      grad.addColorStop(0, "rgba(255,255,0,0.35)");
      grad.addColorStop(1, "rgba(255,255,0,0.02)");
      const labels = Array.isArray(data.dates) ? data.dates.map(d => String(d).replace("T", " ").slice(0, 16)) : [];
      const series = Array.isArray(data.aqi) ? data.aqi : [];

      new Chart(ctx, {
        type: "line",
        data: {
          labels,
          datasets: [{
            label: "AQI Last 24 Hours",
            data: series,
            borderColor: "rgba(255,255,0,0.95)",
            backgroundColor: grad,
            pointRadius: 2,
            borderWidth: 3,
            fill: true,
            tension: 0.3
          }]
        },
        options: {
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: "rgba(255,255,255,0.95)" } },
            tooltip: { backgroundColor: "rgba(0,0,0,0.7)" }
          },
          scales: {
            x: {
              ticks: { color: "rgba(255,255,255,0.85)", maxRotation: 45, minRotation: 45, autoSkip: true },
              grid: { color: "rgba(255,255,255,0.08)" }
            },
            y: {
              beginAtZero: true,
              ticks: { color: "rgba(255,255,255,0.85)" },
              grid: { color: "rgba(255,255,255,0.08)" }
            }
          }
        }
      });
    }

    if (data.aqi && data.aqi.length > 0) {
      const latest = data.aqi[data.aqi.length - 1];
      const [cat, advice] = setCategory(latest);
      const aqiValue = document.getElementById("aqiValue");
      const aqiCategory = document.getElementById("aqiCategory");
      const healthAdvice = document.getElementById("healthAdvice");
      if (aqiValue) aqiValue.innerText = "AQI: " + Number(latest).toFixed(0);
      if (aqiCategory) {
        aqiCategory.innerText = cat;
        applyCategoryClass(aqiCategory, cat);
      }
      if (healthAdvice) healthAdvice.innerText = advice;
    }
  })
  .catch(() => {
    const aqiValue = document.getElementById("aqiValue");
    if (aqiValue) aqiValue.innerText = "AQI: --";
  });

fetch("/feature_importance")
  .then((res) => res.json())
  .then((data) => {
    const labels = Object.keys(data || {});
    const raw = Object.values(data || {});
    const sum = raw.reduce((a, b) => a + b, 0) || 1;
    const valuesPct = raw.map((v) => (v / sum) * 100);

    const canvas = document.getElementById("importanceChart");
    if (canvas) {
      const ctx = canvas.getContext("2d");
      new Chart(ctx, {
        type: "bar",
        data: {
          labels,
          datasets: [{
            label: "Feature Importance (%)",
            data: valuesPct,
            backgroundColor: "rgba(0,255,255,0.6)",
            borderColor: "rgba(0,255,255,0.95)",
            borderWidth: 1.5
          }]
        },
        options: {
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: "rgba(255,255,255,0.95)" } },
            tooltip: { backgroundColor: "rgba(0,0,0,0.7)", callbacks: { label: (ctx) => ctx.raw.toFixed(2) + "%" } },
            datalabels: {
              color: "#ffffff",
              anchor: "end",
              align: "top",
              formatter: (v) => v.toFixed(1) + "%"
            }
          },
          scales: {
            x: { ticks: { color: "rgba(255,255,255,0.85)" }, grid: { color: "rgba(255,255,255,0.08)" } },
            y: { beginAtZero: true, ticks: { color: "rgba(255,255,255,0.85)", callback: (v) => v.toFixed ? v.toFixed(0) + "%" : v + "%" }, grid: { color: "rgba(255,255,255,0.08)" } }
          }
        }
      });
    }
  })
  .catch(() => {
    const canvas = document.getElementById("importanceChart");
    if (canvas) canvas.parentElement.innerHTML = "<p>Feature importance unavailable.</p>";
  });

fetch("/model_comparison")
  .then((res) => res.json())
  .then((data) => {
    const tbody = document.querySelector("#modelTable tbody");
    if (!tbody || !data) return;
    const order = ["Linear","RF","XGB"];
    const rows = order.filter((k) => k in data).map((k) => {
      const m = data[k];
      const r2 = typeof m.R2 === "number" ? m.R2.toFixed(3) : "--";
      const mae = typeof m.MAE === "number" ? m.MAE.toFixed(3) : "--";
      const rmse = typeof m.RMSE === "number" ? m.RMSE.toFixed(3) : "--";
      return `<tr><td>${k}</td><td>${mae}</td><td>${rmse}</td><td>${r2}</td></tr>`;
    }).join("");
    tbody.innerHTML = rows || "<tr><td colspan='4'>No comparison available</td></tr>";
  })
  .catch(() => {
    const tbody = document.querySelector("#modelTable tbody");
    if (tbody) tbody.innerHTML = "<tr><td colspan='4'>No comparison available</td></tr>";
  });

fetch("/model_confidence")
  .then((res) => res.json())
  .then((data) => {
    const el = document.getElementById("confidenceScore");
    const desc = document.getElementById("confidenceDesc");
    if (el && data && typeof data.r2_score === "number") {
      el.innerText = "R² Score: " + data.r2_score.toFixed(3);
      if (desc) {
        const pct = (data.r2_score * 100);
        desc.innerText = "Model explains " + pct.toFixed(1) + "% of AQI variation on unseen data.";
      }
    } else if (el) {
      el.innerText = "R² Score: --";
      if (desc) desc.innerText = "";
    }
  })
  .catch(() => {
    const el = document.getElementById("confidenceScore");
    const desc = document.getElementById("confidenceDesc");
    if (el) el.innerText = "R² Score: --";
    if (desc) desc.innerText = "";
  });
