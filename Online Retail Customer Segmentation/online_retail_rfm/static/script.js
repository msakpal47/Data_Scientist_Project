function loadData() {
  fetch("/api/segments")
    .then((res) => res.json())
    .then((data) => {
      const totalEl = document.getElementById("totalCount");
      const highEl = document.getElementById("highValueCount");
      const loyalEl = document.getElementById("loyalCount");
      const riskEl = document.getElementById("riskCount");
      const chartCanvas = document.getElementById("segmentChart");

      const tbody = document.querySelector("#table tbody");
      tbody.innerHTML = "";
      data.slice(0, 100).forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${row.CustomerID}</td>
            <td>${row.Recency}</td>
            <td>${row.Frequency}</td>
            <td>${Number(row.Monetary).toFixed(2)}</td>
            <td>${row.Cluster}</td>
            <td>${row.Segment || ""}</td>
          `;
        tbody.appendChild(tr);
      });

      const total = data.length;
      const counts = data.reduce((acc, row) => {
        const seg = row.Segment || "Unknown";
        acc[seg] = (acc[seg] || 0) + 1;
        return acc;
      }, {});

      totalEl.textContent = total;
      highEl.textContent = counts["High Value"] || 0;
      loyalEl.textContent = counts["Loyal"] || 0;
      riskEl.textContent = counts["At Risk"] || 0;

      if (chartCanvas && chartCanvas.getContext) {
        const ctx = chartCanvas.getContext("2d");
        const segments = ["Low Value", "Medium", "High Value", "Loyal", "At Risk"];
        const values = segments.map((s) => counts[s] || 0);
        const maxVal = Math.max(...values, 1);
        const width = chartCanvas.width;
        const height = chartCanvas.height;
        ctx.clearRect(0, 0, width, height);

        const padding = 30;
        const chartWidth = width - padding * 2;
        const chartHeight = height - padding * 2;
        const barWidth = chartWidth / segments.length - 20;
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px Arial";
        ctx.textAlign = "center";

        segments.forEach((seg, i) => {
          const x = padding + i * (barWidth + 20);
          const barHeight = Math.round((values[i] / maxVal) * chartHeight);
          const y = height - padding - barHeight;
          ctx.fillStyle = "#00c853";
          ctx.fillRect(x, y, barWidth, barHeight);
          ctx.fillStyle = "#ffffff";
          ctx.fillText(seg, x + barWidth / 2, height - padding + 15);
          ctx.fillText(values[i], x + barWidth / 2, y - 5);
        });
      }
    })
    .catch((err) => {
      console.error("Failed to load segments", err);
    });
}
