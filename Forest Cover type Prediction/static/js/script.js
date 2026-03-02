const BASE_COLUMNS = Array.isArray(window.__BASE_COLUMNS__) ? window.__BASE_COLUMNS__ : [];
let WILDERNESS_OPTIONS = [];
let SOIL_OPTIONS = [];
let WILDERNESS_KEYS = [];
let SOIL_KEYS = [];
let CONTINUOUS_VALUES = {};

function el(id) {
  return document.getElementById(id);
}

function setBadge(state, text) {
  const badge = el("statusBadge");
  badge.textContent = text;
  badge.className = `badge ${state}`;
}

function groupColumns(cols) {
  const continuous = [];
  const wilderness = [];
  const soil = [];
  for (const c of cols) {
    if (c.startsWith("Wilderness_Area_")) wilderness.push(c);
    else if (c.startsWith("Soil_Type_")) soil.push(c);
    else continuous.push(c);
  }
  return { continuous, wilderness, soil };
}

function makeNumberField(name, placeholder = "") {
  const wrap = document.createElement("div");
  wrap.className = "field";

  const label = document.createElement("div");
  label.className = "field__label";
  label.textContent = name;

  const inputWrap = document.createElement("div");
  inputWrap.className = "field__input";

  const input = document.createElement("input");
  input.className = "input";
  input.type = "number";
  input.step = "any";
  input.placeholder = placeholder;
  input.setAttribute("data-key", name);

  inputWrap.appendChild(input);
  wrap.appendChild(label);
  wrap.appendChild(inputWrap);
  return wrap;
}

function makeCheckboxField(name) {
  const wrap = document.createElement("label");
  wrap.className = "field check";

  const input = document.createElement("input");
  input.type = "checkbox";
  input.setAttribute("data-key", name);

  const text = document.createElement("div");
  text.className = "field__label";
  text.textContent = name;

  wrap.appendChild(input);
  wrap.appendChild(text);
  return wrap;
}

function makeSelectField(labelText, options, groupName) {
  const wrap = document.createElement("div");
  wrap.className = "field";

  const label = document.createElement("div");
  label.className = "field__label";
  label.textContent = labelText;

  const inputWrap = document.createElement("div");
  inputWrap.className = "field__input";

  const select = document.createElement("select");
  select.className = "input";
  select.setAttribute("data-group", groupName);

  const noneOpt = document.createElement("option");
  noneOpt.value = "";
  noneOpt.textContent = "— Select —";
  select.appendChild(noneOpt);

  options.forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt.key;
    o.textContent = opt.name;
    select.appendChild(o);
  });

  inputWrap.appendChild(select);
  wrap.appendChild(label);
  wrap.appendChild(inputWrap);
  return wrap;
}

function makeSelectFieldForColumn(name, values) {
  const wrap = document.createElement("div");
  wrap.className = "field";
  const label = document.createElement("div");
  label.className = "field__label";
  label.textContent = name;
  const inputWrap = document.createElement("div");
  inputWrap.className = "field__input";
  const select = document.createElement("select");
  select.className = "input";
  select.setAttribute("data-key", name);
  const noneOpt = document.createElement("option");
  noneOpt.value = "";
  noneOpt.textContent = "— Select —";
  select.appendChild(noneOpt);
  (values || []).forEach((v) => {
    const o = document.createElement("option");
    o.value = String(v);
    o.textContent = String(v);
    select.appendChild(o);
  });
  inputWrap.appendChild(select);
  wrap.appendChild(label);
  wrap.appendChild(inputWrap);
  return wrap;
}

function setActiveTab(tabName) {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");
  tabs.forEach((t) => t.classList.toggle("tab--active", t.getAttribute("data-tab") === tabName));
  panels.forEach((p) => p.classList.toggle("panel--active", p.getAttribute("data-panel") === tabName));
}

function attachTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((t) => {
    t.addEventListener("click", () => setActiveTab(t.getAttribute("data-tab")));
  });
}

function fillExample() {
  const defaults = {
    Elevation: 2800,
    Aspect: 160,
    Slope: 12,
    Horizontal_Distance_To_Hydrology: 240,
    Vertical_Distance_To_Hydrology: 30,
    Horizontal_Distance_To_Roadways: 800,
    Hillshade_9am: 220,
    Hillshade_Noon: 230,
    Hillshade_3pm: 180,
    Horizontal_Distance_To_Fire_Points: 1200,
  };

  for (const k of Object.keys(defaults)) {
    const input = document.querySelector(`[data-key="${CSS.escape(k)}"]`);
    if (input) {
      if (input.tagName === "SELECT") input.value = String(defaults[k]);
      else if (input.type === "number") input.value = String(defaults[k]);
    }
  }

  const wildSelect = document.querySelector('select[data-group="wilderness"]');
  if (wildSelect) {
    const opt = WILDERNESS_OPTIONS.find((o) => o.key === "Wilderness_Area_1");
    wildSelect.value = opt ? opt.key : "";
  }
  const soilSelect = document.querySelector('select[data-group="soil"]');
  if (soilSelect) {
    const opt = SOIL_OPTIONS.find((o) => o.key === "Soil_Type_10");
    soilSelect.value = opt ? opt.key : "";
  }
}

function resetForm() {
  el("predictForm").reset();
  el("resultValue").textContent = "—";
  el("resultMeta").textContent = "";
}

function collectRow() {
  const row = {};
  const inputs = document.querySelectorAll("[data-key]");
  inputs.forEach((inp) => {
    const key = inp.getAttribute("data-key");
    if (inp.type === "checkbox") row[key] = inp.checked ? 1 : 0;
    else {
      const val = inp.value;
      row[key] = val === "" ? 0 : Number(val);
    }
  });
  const wildSelect = document.querySelector('select[data-group="wilderness"]');
  if (wildSelect) {
    WILDERNESS_KEYS.forEach((k) => (row[k] = 0));
    if (wildSelect.value) row[wildSelect.value] = 1;
  }
  const soilSelect = document.querySelector('select[data-group="soil"]');
  if (soilSelect) {
    SOIL_KEYS.forEach((k) => (row[k] = 0));
    if (soilSelect.value) row[soilSelect.value] = 1;
  }
  return row;
}

async function fetchMetadata() {
  const res = await fetch("/api/metadata");
  if (!res.ok) throw new Error("Failed to load metadata");
  return await res.json();
}

async function predict(row) {
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(row),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data && data.error ? data.error : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

async function fetchOptions() {
  const res = await fetch("/api/options");
  if (!res.ok) return { wilderness: [], soils: [] };
  return await res.json();
}

async function fetchContinuousValues(cols) {
  const q = encodeURIComponent(cols.join(","));
  const res = await fetch(`/api/column_values?columns=${q}`);
  if (!res.ok) return {};
  return await res.json();
}

async function fetchInsights() {
  const res = await fetch("/api/insights");
  if (!res.ok) return { items: [] };
  return await res.json();
}

function renderInsights(ins) {
  const root = el("insightsRoot");
  if (!root) return;
  root.innerHTML = "";
  const items = Array.isArray(ins.items) ? ins.items : [];
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "kv__row";
    empty.textContent = "No insights available";
    root.appendChild(empty);
    return;
  }
  items.slice(0, 10).forEach((m) => {
    const row = document.createElement("div");
    row.className = "kv__row";
    const k = document.createElement("div");
    k.className = "kv__k";
    k.textContent = (m.model_name || "Model") + " — " + (m.trained_at || "");
    const v = document.createElement("div");
    v.className = "kv__v";
    const acc = typeof m.accuracy === "number" ? m.accuracy.toFixed(4) : "—";
    const f1 = typeof m.macro_f1 === "number" ? m.macro_f1.toFixed(4) : "—";
    v.textContent = `Acc: ${acc} • Macro F1: ${f1} • n=${m.sample_n ?? "?"}`;
    row.appendChild(k);
    row.appendChild(v);
    root.appendChild(row);
  });
}

function renderFields(cols, opts, contVals) {
  const { continuous, wilderness, soil } = groupColumns(cols);

  const continuousRoot = el("continuousFields");
  const wildernessRoot = el("wildernessFields");
  const soilRoot = el("soilFields");
  continuousRoot.innerHTML = "";
  wildernessRoot.innerHTML = "";
  soilRoot.innerHTML = "";

  continuous.forEach((c) => {
    const values = contVals && Array.isArray(contVals[c]) ? contVals[c] : null;
    continuousRoot.appendChild(makeSelectFieldForColumn(c, values || []));
  });
  WILDERNESS_KEYS = wilderness.slice();
  SOIL_KEYS = soil.slice();
  WILDERNESS_OPTIONS = Array.isArray(opts?.wilderness) && opts.wilderness.length
    ? opts.wilderness
    : wilderness.map((k) => ({ key: k, name: k }));
  SOIL_OPTIONS = Array.isArray(opts?.soils) && opts.soils.length
    ? opts.soils
    : soil.map((k) => ({ key: k, name: k }));
  wildernessRoot.appendChild(makeSelectField("Wilderness Area", WILDERNESS_OPTIONS, "wilderness"));
  soilRoot.appendChild(makeSelectField("Soil Type", SOIL_OPTIONS, "soil"));
}

function renderMeta(meta) {
  el("acc").textContent = typeof meta.accuracy === "number" ? meta.accuracy.toFixed(4) : "—";
  el("f1").textContent = typeof meta.macro_f1 === "number" ? meta.macro_f1.toFixed(4) : "—";

  el("modelFile").textContent = meta.model_available ? "models/model.pkl" : "missing (placeholder used)";
  el("metadataFile").textContent = meta.metadata_available ? "models/metadata.json" : "missing";

  if (meta.model_available) setBadge("badge--ok", "Model loaded");
  else setBadge("badge--warn", "Model missing (placeholder)");
}

async function main() {
  attachTabs();

  el("fillExample").addEventListener("click", fillExample);
  el("resetForm").addEventListener("click", resetForm);
  const refreshBtn = el("refreshInsights");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", async () => {
      const ins = await fetchInsights().catch(() => ({ items: [] }));
      renderInsights(ins);
    });
  }

  setBadge("badge--loading", "Loading…");

  let meta;
  try {
    meta = await fetchMetadata();
  } catch (e) {
    setBadge("badge--warn", "Metadata error");
    el("resultMeta").textContent = String(e && e.message ? e.message : e);
    const opts = await fetchOptions().catch(() => ({}));
    const contVals = await fetchContinuousValues(groupColumns(BASE_COLUMNS).continuous).catch(() => ({}));
    renderFields(BASE_COLUMNS, opts, contVals);
    return;
  }

  const opts = await fetchOptions().catch(() => ({}));
  const colsToUse = Array.isArray(meta.base_columns) && meta.base_columns.length ? meta.base_columns : BASE_COLUMNS;
  const contVals = await fetchContinuousValues(groupColumns(colsToUse).continuous).catch(() => ({}));
  renderFields(colsToUse, opts, contVals);
  renderMeta(meta);
  const ins = await fetchInsights().catch(() => ({ items: [] }));
  renderInsights(ins);

  el("predictForm").addEventListener("submit", async (evt) => {
    evt.preventDefault();
    const btn = el("predictBtn");
    btn.disabled = true;
    el("resultMeta").textContent = "Predicting…";
    el("resultValue").textContent = "—";

    try {
      const row = collectRow();
      const out = await predict(row);
      const pred = Array.isArray(out.predictions) ? out.predictions[0] : null;
      el("resultValue").textContent = pred == null ? "—" : String(pred);
      if (Array.isArray(out.probabilities) && Array.isArray(out.class_labels) && out.probabilities.length) {
        const maxIdx = out.probabilities.reduce((mi, v, i, arr) => (v > arr[mi] ? i : mi), 0);
        const topProb = out.probabilities[maxIdx];
        const topLabel = out.class_labels[maxIdx];
        el("resultMeta").textContent = `Top: Cover ${topLabel} (${(topProb * 100).toFixed(1)}%)`;
      } else {
        el("resultMeta").textContent = out.model_available ? "Prediction from model.pkl" : "Prediction from placeholder (add model.pkl)";
      }
    } catch (e) {
      el("resultMeta").textContent = String(e && e.message ? e.message : e);
    } finally {
      btn.disabled = false;
    }
  });
}

main();
