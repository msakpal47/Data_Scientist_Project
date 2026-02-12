function safeId(name) {
  return "f_" + name.replace(/[^a-zA-Z0-9_]+/g, "_");
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data && data.error ? data.error : `Request failed: ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function setStatus(el, text, kind) {
  el.textContent = text;
  el.classList.remove("ok", "bad");
  if (kind) el.classList.add(kind);
}

function buildForm(schema) {
  const form = document.getElementById("form");
  form.innerHTML = "";

  for (const f of schema.features) {
    const wrap = document.createElement("div");
    wrap.className = "field";

    const label = document.createElement("label");
    label.setAttribute("for", safeId(f.name));
    label.textContent = f.name;

    const input = document.createElement("input");
    input.id = safeId(f.name);
    input.type = "number";
    input.step = "any";
    input.value = Number.isFinite(f.median) ? String(f.median) : "0";
    if (Number.isFinite(f.min)) input.min = String(f.min);
    if (Number.isFinite(f.max)) input.max = String(f.max);

    wrap.appendChild(label);
    wrap.appendChild(input);
    form.appendChild(wrap);
  }
}

function readForm(schema) {
  const payload = {};
  for (const f of schema.features) {
    const el = document.getElementById(safeId(f.name));
    const v = el.value === "" ? null : Number(el.value);
    payload[f.name] = Number.isFinite(v) ? v : null;
  }
  return payload;
}

async function main() {
  const trainStatus = document.getElementById("train-status");
  const predictStatus = document.getElementById("predict-result");
  const btnTrain = document.getElementById("btn-train");
  const btnPredict = document.getElementById("btn-predict");

  setStatus(trainStatus, "Loading schema…");
  setStatus(predictStatus, "");

  let schema = null;
  try {
    schema = await fetchJson("/api/schema");
    buildForm(schema);
    setStatus(trainStatus, `Schema ready. Features: ${schema.features.length}`, "ok");
  } catch (e) {
    setStatus(trainStatus, `Schema load failed: ${e.message}`, "bad");
    return;
  }

  btnTrain.addEventListener("click", async () => {
    btnTrain.disabled = true;
    setStatus(trainStatus, "Training model… (this may take a bit)");
    try {
      const out = await fetchJson("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_negative_rows: 200000 })
      });
      setStatus(trainStatus, `Training complete. Accuracy: ${out.accuracy.toFixed(3)}`, "ok");
    } catch (e) {
      setStatus(trainStatus, `Training failed: ${e.message}`, "bad");
    } finally {
      btnTrain.disabled = false;
    }
  });

  btnPredict.addEventListener("click", async () => {
    btnPredict.disabled = true;
    setStatus(predictStatus, "Predicting…");
    try {
      const payload = readForm(schema);
      const out = await fetchJson("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const pred = out.prediction;
      const prob = out.probability;
      const probTxt = prob === null || prob === undefined ? "" : ` (probability: ${prob.toFixed(3)})`;
      setStatus(predictStatus, `Predicted fault_occurred: ${pred}${probTxt}`, pred === 1 ? "bad" : "ok");
    } catch (e) {
      setStatus(predictStatus, `Prediction failed: ${e.message}`, "bad");
    } finally {
      btnPredict.disabled = false;
    }
  });
}

main();
