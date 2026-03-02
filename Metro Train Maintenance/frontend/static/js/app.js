function safeId(name) {
  return "f_" + name.replace(/[^a-zA-Z0-9_]+/g, "_");
}

async function fetchJson(url, options, retries = 3) {
  let lastErr = null;
  for (let i = 0; i <= retries; i++) {
    try {
      const res = await fetch(url, options);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg = data && data.error ? data.error : `Request failed: ${res.status}`;
        throw new Error(msg);
      }
      return data;
    } catch (e) {
      lastErr = e;
      if (i < retries) {
        const delay = 300 * Math.pow(2, i);
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
    }
  }
  throw lastErr || new Error("Request failed");
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

    // Attach datalist for dropdown suggestions
    const dlId = "dl_" + safeId(f.name);
    input.setAttribute("list", dlId);
    const dl = document.createElement("datalist");
    dl.id = dlId;
    // placeholder option while loading
    const opt = document.createElement("option");
    opt.value = "";
    opt.label = "Loading…";
    dl.appendChild(opt);

    wrap.appendChild(label);
    wrap.appendChild(input);
    wrap.appendChild(dl);
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

function applyPrefill(schema, mode) {
  for (const f of schema.features) {
    const el = document.getElementById(safeId(f.name));
    let val = "";
    if (mode === "median") {
      val = Number.isFinite(f.median) ? f.median : "";
    } else if (mode === "min") {
      val = Number.isFinite(f.min) ? f.min : "";
    } else if (mode === "max") {
      val = Number.isFinite(f.max) ? f.max : "";
    } else if (mode === "random") {
      const lo = Number.isFinite(f.min) ? f.min : 0;
      const hi = Number.isFinite(f.max) ? f.max : 1;
      val = lo + Math.random() * (hi - lo);
    } else if (mode === "zeros") {
      val = 0;
    }
    if (el) el.value = val === "" ? "" : String(val);
  }
}

async function main() {
  const trainStatus = document.getElementById("train-status");
  const predictStatus = document.getElementById("predict-result");
  const btnTrain = document.getElementById("btn-train");
  const btnPredict = document.getElementById("btn-predict");
  const prefillSelect = document.getElementById("prefill-select");
  const modelSelect = document.getElementById("model-select");
  const thresholdEl = document.getElementById("threshold");
  const healthBadge = document.getElementById("health-badge");
  const calibrateEl = document.getElementById("calibrate");
  const speedModeEl = document.getElementById("speed-mode");
  const trainBgEl = document.getElementById("train-bg");
  const btnRefreshSchema = document.getElementById("btn-refresh-schema");
  const timestampColEl = document.getElementById("timestamp-col");
  const splitSelect = document.getElementById("split-select");
  const metricsPanel = document.getElementById("metrics-panel");
  const importanceList = document.getElementById("importance-list");
  const rocCanvas = document.getElementById("roc-chart");
  const confusionPanel = document.getElementById("confusion-panel");
  const prCanvas = document.getElementById("pr-chart");
  const cmCanvas = document.getElementById("cm-chart");
  const btnUseBest = document.getElementById("btn-use-best");
  const btnUseBestRecall = document.getElementById("btn-use-best-recall");
  const quickThresholds = document.getElementById("quick-thresholds");
  const thresholdEvalPanel = document.getElementById("threshold-eval");
  const autoBestEl = document.getElementById("auto-best");
  const autoBestRecallEl = document.getElementById("auto-best-recall");
  const recallTargetEl = document.getElementById("recall-target");
  const smoteEl = document.getElementById("smote");
  const healthTable = document.getElementById("model-health");
  const btnPromote = document.getElementById("btn-promote");
  const promoteStatus = document.getElementById("promote-status");
  const recBadge = document.getElementById("recommended-badge");

  setStatus(trainStatus, "Loading schema…");
  setStatus(predictStatus, "");

  let schema = null;
  try {
    schema = await fetchJson("/api/schema", undefined, 2);
    buildForm(schema);
    setStatus(trainStatus, `Schema ready. Features: ${schema.features.length}`, "ok");
    // Lazy-load distinct values for each feature on first focus to reduce startup time
    const loaded = new Set();
    for (const f of schema.features) {
      const input = document.getElementById(safeId(f.name));
      const dl = document.getElementById("dl_" + safeId(f.name));
      if (!input || !dl) continue;
      input.addEventListener("focus", async () => {
        if (loaded.has(f.name)) return;
        try {
          const resp = await fetchJson(`/api/feature_values?name=${encodeURIComponent(f.name)}&limit=200`, undefined, 1);
          dl.innerHTML = "";
          for (const v of resp.values) {
            const o = document.createElement("option");
            o.value = String(v);
            dl.appendChild(o);
          }
        } catch {
          dl.innerHTML = "";
        } finally {
          loaded.add(f.name);
        }
      }, { once: false });
    }
    if (prefillSelect) {
      applyPrefill(schema, prefillSelect.value || "median");
      prefillSelect.addEventListener("change", () => {
        applyPrefill(schema, prefillSelect.value || "median");
      });
    }
  } catch (e) {
    setStatus(trainStatus, `Schema load failed: ${e.message}`, "bad");
    return;
  }

  async function pollHealth() {
    try {
      const h = await fetchJson("/api/health", undefined, 1);
      if (healthBadge) {
        healthBadge.textContent = "Backend: Online";
        healthBadge.classList.add("primary");
      }
    } catch {
      if (healthBadge) {
        healthBadge.textContent = "Backend: Offline";
        healthBadge.classList.remove("primary");
      }
    }
  }
  pollHealth();
  setInterval(pollHealth, 10000);

  // Populate timestamp column dropdown (datalist)
  async function initTimestampOptions() {
    const dl = document.getElementById("timestamp-options");
    if (!dl) return;
    try {
      const res = await fetchJson("/api/time_columns", undefined, 1);
      dl.innerHTML = "";
      const cols = Array.isArray(res.columns) ? res.columns : [];
      for (const c of cols) {
        const o = document.createElement("option");
        o.value = c;
        dl.appendChild(o);
      }
      // If user selects or types a timestamp, auto switch to Time-Based
      if (timestampColEl) {
        timestampColEl.addEventListener("change", () => {
          if (timestampColEl.value && splitSelect) splitSelect.value = "temporal";
        });
      }
    } catch {
      // ignore
    }
  }
  initTimestampOptions();

  function buildTrainPayload() {
    const model_type = modelSelect ? modelSelect.value : "logreg";
    const speed = speedModeEl ? speedModeEl.checked : false;
    const calibrate = calibrateEl ? calibrateEl.checked : false;
    const payload = {
      model_type,
      calibrate: speed ? false : calibrate,
      calibration_method: "sigmoid",
      speed_mode: speed,
      recall_target: (() => {
        const v = recallTargetEl ? Number(recallTargetEl.value) : NaN;
        return Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0.7;
      })(),
      smote: smoteEl ? !!smoteEl.checked : false,
      cv_folds: speed ? 3 : 5
    };
    if (splitSelect) {
      payload.split_type = splitSelect.value;
    }
    if (model_type === "logreg") {
      payload.solver = speed ? "liblinear" : "lbfgs";
    }
    if (speed && !payload.max_negative_rows) {
      payload.max_negative_rows = 100000;
    }
    if (timestampColEl && timestampColEl.value.trim() !== "") {
      payload.timestamp_column = timestampColEl.value.trim();
    }
    return payload;
  }

  if (speedModeEl && calibrateEl) {
    const syncCal = () => {
      if (speedModeEl.checked) {
        calibrateEl.checked = false;
        calibrateEl.disabled = true;
      } else {
        calibrateEl.disabled = false;
      }
    };
    syncCal();
    speedModeEl.addEventListener("change", syncCal);
  }

  function setRecommendedDefault() {
    if (!recBadge) return;
    recBadge.classList.remove("warning", "info");
    recBadge.textContent = "Recommended: XGBoost • Recall≥0.85 • Time‑Based Split • SMOTE Off";
  }
  setRecommendedDefault();

  function updateRecommendedBadge(payload, out) {
    if (!recBadge) return;
    const msgs = [];
    if (out && out.model_type && out.model_type !== "xgb") msgs.push("Switch to XGBoost");
    if (out && out.split && out.split !== "temporal") msgs.push("Use Time‑Based Split");
    if (payload && payload.smote) msgs.push("Turn SMOTE Off");
    const rt = payload && typeof payload.recall_target === "number" ? payload.recall_target : 0.7;
    if (rt < 0.85) msgs.push("Use Recall≥0.85");
    if (out && out.cv) {
      if (typeof out.cv.avg_precision === "number" && out.cv.avg_precision < 0.5) msgs.push("CV PR AUC low");
      if (typeof out.cv.f1_mean === "number" && out.cv.f1_mean < 0.5) msgs.push("CV F1 low");
    }
    if (msgs.length === 0) {
      recBadge.classList.remove("warning");
      recBadge.classList.add("info");
      recBadge.textContent = "Using recommended settings";
    } else {
      recBadge.classList.remove("info");
      recBadge.classList.add("warning");
      recBadge.textContent = "Recommended: XGBoost • Recall≥0.85 • Time‑Based Split • SMOTE Off — " + msgs.join(" • ");
    }
  }

  if (btnRefreshSchema) {
    btnRefreshSchema.addEventListener("click", async () => {
      setStatus(trainStatus, "Refreshing schema…");
      try {
        const out = await fetchJson("/api/schema_refresh", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ limit: 50000 })
        }, 0);
        schema = out.schema;
        buildForm(schema);
        setStatus(trainStatus, "Schema refreshed.", "ok");
      } catch (e) {
        setStatus(trainStatus, `Schema refresh failed: ${e.message}`, "bad");
      }
    });
  }

  btnTrain.addEventListener("click", async () => {
    btnTrain.disabled = true;
    setStatus(trainStatus, "Training model… (this may take a bit)");
    try {
      const payload = buildTrainPayload();
      let out = null;
      if (trainBgEl && trainBgEl.checked) {
        const started = await fetchJson("/api/train_async", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }, 0);
        const jobId = started.job_id;
        setStatus(trainStatus, "Training in background…");
        for (;;) {
          await new Promise(r => setTimeout(r, 1500));
          const st = await fetchJson(`/api/train_status?job_id=${encodeURIComponent(jobId)}`, undefined, 0);
          if (st.status === "completed") { out = st.result; break; }
          if (st.status === "failed") { throw new Error(st.error || "Training failed"); }
        }
      } else {
        out = await fetchJson("/api/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        }, 0);
      }
      updateRecommendedBadge(payload, out);
      // Keep last training output for promotion checks
      window.__lastTrainOut = out;
      const parts = [];
      if (typeof out.accuracy === "number") parts.push(`Accuracy: ${out.accuracy.toFixed(3)}`);
      if (typeof out.precision === "number") parts.push(`Precision: ${out.precision.toFixed(3)}`);
      if (typeof out.recall === "number") parts.push(`Recall: ${out.recall.toFixed(3)}`);
      if (typeof out.f1 === "number") parts.push(`F1: ${out.f1.toFixed(3)}`);
      if (typeof out.roc_auc === "number") parts.push(`ROC AUC: ${out.roc_auc.toFixed(3)}`);
      if (typeof out.avg_precision === "number") parts.push(`PR AUC: ${out.avg_precision.toFixed(3)}`);
      if (typeof out.best_threshold === "number") parts.push(`Best T: ${out.best_threshold.toFixed(2)}`);
      const mt = out.model_type ? ` (${out.model_type})` : "";
      const splitTxt = out.split ? ` • Evaluation Mode: ${out.split === "temporal" ? "Time‑Based" : "Random"}` : "";
      const warnTxt = out.split_warning ? ` • Note: ${out.split_warning}` : "";
      let healthTxt = "";
      if (out.dataset_info) {
        const di = out.dataset_info;
        healthTxt = ` • Data — Train: ${di.n_train} (pos ${di.pos_train}) • Test: ${di.n_test} (pos ${di.pos_test})`;
      }
      if (out.cv && typeof out.cv.f1_mean === "number") {
        healthTxt += ` • CV F1: ${out.cv.f1_mean.toFixed(3)} ± ${out.cv.f1_std?.toFixed ? out.cv.f1_std.toFixed(3) : "0.000"}`;
      }
      setStatus(trainStatus, `Training complete${mt}. ${parts.join(" • ")}${splitTxt}${healthTxt}${warnTxt}`, "ok");
      if (metricsPanel) {
        metricsPanel.textContent = parts.join(" • ");
      }
      let selectedConf = null;
      let selectedThr = null;
      if (Array.isArray(out.thresholds_eval) && typeof out.best_threshold === "number") {
        let best = null;
        let dmin = Infinity;
        for (const e of out.thresholds_eval) {
          const d = Math.abs(e.threshold - out.best_threshold);
          if (d < dmin) { dmin = d; best = e; }
        }
        if (best && best.confusion) {
          selectedConf = best.confusion;
          selectedThr = best.threshold;
        }
      }
      if (!selectedConf && out.confusion) {
        selectedConf = out.confusion;
      }
      if (confusionPanel && selectedConf) {
        const c = selectedConf;
        const tTxt = selectedThr != null ? ` at T=${Number(selectedThr).toFixed(2)}` : "";
        confusionPanel.textContent = `Confusion${tTxt} — TN: ${c.tn}, FP: ${c.fp}, FN: ${c.fn}, TP: ${c.tp}`;
      }
      // Show suggested best threshold (by F1)
      if (btnUseBest && typeof out.best_threshold === "number" && thresholdEl) {
        btnUseBest.onclick = () => {
          thresholdEl.value = out.best_threshold.toFixed(2);
        };
        btnUseBest.title = `Best threshold (F1): ${out.best_threshold.toFixed(3)}`;
      }
      // Show best-at-recall threshold
      if (btnUseBestRecall && typeof out.best_threshold_at_recall === "number" && thresholdEl) {
        btnUseBestRecall.onclick = () => {
          thresholdEl.value = out.best_threshold_at_recall.toFixed(2);
        };
        const p = typeof out.precision_at_recall_target === "number" ? out.precision_at_recall_target.toFixed(2) : "N/A";
        const r = typeof out.recall_at_recall_target === "number" ? out.recall_at_recall_target.toFixed(2) : "N/A";
        const rt = typeof out.recall_target === "number" ? out.recall_target.toFixed(2) : "0.70";
        btnUseBestRecall.title = `Best@Recall≥${rt} • P=${p} R=${r}`;
      }
      if (autoBestRecallEl && autoBestRecallEl.checked && typeof out.best_threshold_at_recall === "number" && thresholdEl) {
        thresholdEl.value = out.best_threshold_at_recall.toFixed(2);
      } else if (autoBestEl && autoBestEl.checked && typeof out.best_threshold === "number" && thresholdEl) {
        thresholdEl.value = out.best_threshold.toFixed(2);
      } else if (typeof out.default_threshold === "number" && thresholdEl) {
        thresholdEl.value = out.default_threshold.toFixed(2);
      }
      if (importanceList) {
        const imp = Array.isArray(out.importance) ? out.importance : [];
        const top = imp.slice(0, 5).map(x => {
          const dir = x.direction === "up" ? "↑" : (x.direction === "down" ? "↓" : "");
          return `${x.name} ${dir} ${x.abs.toFixed(3)}`;
        });
        importanceList.textContent = top.length ? `Top features: ${top.join(" • ")}` : "Top features: N/A";
      }
      // Threshold evaluations
      if (thresholdEvalPanel && Array.isArray(out.thresholds_eval)) {
        const lines = out.thresholds_eval.map(e => {
          const t = e.threshold.toFixed(2);
          const pr = e.precision.toFixed(2);
          const rc = e.recall.toFixed(2);
          return `T=${t} • P=${pr} • R=${rc}`;
        });
        const extra = (typeof out.best_threshold_at_recall === "number" && typeof out.precision_at_recall_target === "number" && typeof out.recall_at_recall_target === "number")
          ? ` • Best@Recall: T=${out.best_threshold_at_recall.toFixed(2)} • P=${out.precision_at_recall_target.toFixed(2)} • R=${out.recall_at_recall_target.toFixed(2)}`
          : "";
        thresholdEvalPanel.textContent = `Threshold evals: ${lines.join(" • ")}${extra}`;
      }
      // Model health table
      if (healthTable) {
        const rows = [];
        if (out.split) {
          const splitLabel = out.split === "temporal" ? "time-based" : "random";
          rows.push(`<tr><td>Split</td><td>${splitLabel}</td></tr>`);
        }
        if (out.dataset_info) {
          const di = out.dataset_info;
          const trainNeg = (di.n_train - di.pos_train);
          const testNeg = (di.n_test - di.pos_test);
          const totalPos = di.pos_train + di.pos_test;
          const totalNeg = trainNeg + testNeg;
          const ratio = totalPos > 0 ? `1:${Math.round(totalNeg / totalPos)}` : "N/A";
          rows.push(`<tr><td>Train size</td><td>${di.n_train} (pos ${di.pos_train}, neg ${trainNeg})</td></tr>`);
          rows.push(`<tr><td>Test size</td><td>${di.n_test} (pos ${di.pos_test}, neg ${testNeg})</td></tr>`);
          rows.push(`<tr><td>Total positives</td><td>${totalPos}</td></tr>`);
          rows.push(`<tr><td>Total negatives</td><td>${totalNeg}</td></tr>`);
          rows.push(`<tr><td>Imbalance ratio</td><td>${ratio}</td></tr>`);
        }
        if (out.cv && typeof out.cv.f1_mean === "number") {
          rows.push(`<tr><td>CV F1</td><td>${out.cv.f1_mean.toFixed(3)} ± ${out.cv.f1_std?.toFixed ? out.cv.f1_std.toFixed(3) : "0.000"}</td></tr>`);
          if (typeof out.cv.avg_precision === "number") rows.push(`<tr><td>CV PR AUC</td><td>${out.cv.avg_precision.toFixed(3)}</td></tr>`);
          if (typeof out.cv.roc_auc === "number") rows.push(`<tr><td>CV ROC AUC</td><td>${out.cv.roc_auc.toFixed(3)}</td></tr>`);
        }
        if (rows.length) {
          healthTable.innerHTML = `<table style="width:100%;border-collapse:collapse;">
            <tbody>
              ${rows.map(r => r).join("")}
            </tbody>
          </table>`;
        } else {
          healthTable.textContent = "Model health: N/A";
        }
      }
      // Draw ROC curve
      if (rocCanvas && out.roc && Array.isArray(out.roc.fpr) && Array.isArray(out.roc.tpr)) {
        const ctx = rocCanvas.getContext("2d");
        const W = rocCanvas.width, H = rocCanvas.height;
        ctx.clearRect(0,0,W,H);
        ctx.strokeStyle = "rgba(255,255,255,0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(40, H-30); ctx.lineTo(W-10, H-30);
        ctx.moveTo(40, H-30); ctx.lineTo(40, 10);
        ctx.stroke();
        ctx.strokeStyle = "rgba(255,255,255,0.2)";
        ctx.beginPath();
        ctx.moveTo(40, H-30); ctx.lineTo(W-10, 10);
        ctx.stroke();
        const fpr = out.roc.fpr, tpr = out.roc.tpr;
        ctx.strokeStyle = "rgba(14,165,233,0.9)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i=0;i<fpr.length;i++){
          const x = 40 + (W-50) * fpr[i];
          const y = (H-30) - (H-40) * tpr[i];
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.stroke();
      }
      // Draw PR curve
      if (prCanvas && out.pr && Array.isArray(out.pr.precision) && Array.isArray(out.pr.recall)) {
        const ctx = prCanvas.getContext("2d");
        const W = prCanvas.width, H = prCanvas.height;
        ctx.clearRect(0,0,W,H);
        ctx.strokeStyle = "rgba(255,255,255,0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(40, H-30); ctx.lineTo(W-10, H-30);
        ctx.moveTo(40, H-30); ctx.lineTo(40, 10);
        ctx.stroke();
        const prec = out.pr.precision, rec = out.pr.recall;
        ctx.strokeStyle = "rgba(34,197,94,0.9)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i=0;i<prec.length;i++){
          const x = 40 + (W-50) * rec[i];
          const y = (H-30) - (H-40) * prec[i];
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }
        ctx.stroke();
      }
      if (cmCanvas && selectedConf) {
        const { tn, fp, fn, tp } = selectedConf;
        const ctx = cmCanvas.getContext("2d");
        const W = cmCanvas.width, H = cmCanvas.height;
        ctx.clearRect(0,0,W,H);
        const cells = [[tn, fp],[fn, tp]];
        const maxVal = Math.max(tn, fp, fn, tp, 1);
        const pad = 40, cw = (W - pad - 10) / 2, ch = (H - pad - 10) / 2;
        // axes labels
        ctx.fillStyle = "rgba(255,255,255,0.8)";
        ctx.fillText("Pred 0", pad + cw/2 - 15, 15);
        ctx.fillText("Pred 1", pad + cw + cw/2 - 15, 15);
        ctx.save();
        ctx.translate(10, pad + ch);
        ctx.rotate(-Math.PI/2);
        ctx.fillText("True 0 / True 1", 0, 0);
        ctx.restore();
        for (let r=0;r<2;r++){
          for (let c=0;c<2;c++){
            const val = cells[r][c];
            const x = pad + c*cw;
            const y = pad + r*ch;
            const intensity = val / maxVal;
            ctx.fillStyle = `rgba(56,189,248,${0.2 + 0.6*intensity})`;
            ctx.fillRect(x, y, cw-4, ch-4);
            ctx.fillStyle = "white";
            ctx.fillText(String(val), x + (cw/2) - 8, y + (ch/2));
          }
        }
      }
      // Promote button enable
      if (btnPromote) {
        btnPromote.disabled = false;
      }
    } catch (e) {
      setStatus(trainStatus, `Training failed: ${e.message}`, "bad");
    } finally {
      btnTrain.disabled = false;
    }
  });

  // Promote to production
  if (btnPromote) {
    btnPromote.addEventListener("click", async () => {
      const last = window.__lastTrainOut || null;
      // Client-side guardrails for clear UX
      if (last) {
        if (last.split === "temporal" && last.invalid_temporal_test === true) {
          if (promoteStatus) setStatus(promoteStatus, "Invalid time‑based split: No positive events in test window.", "bad");
          return;
        }
        if (last.split !== "temporal") {
          if (promoteStatus) setStatus(promoteStatus, "Retrain with Time‑Based Split before promoting.", "bad");
          return;
        }
        if (last.smote === true) {
          if (promoteStatus) setStatus(promoteStatus, "Turn SMOTE off before promoting.", "bad");
          return;
        }
      }
      if (promoteStatus) setStatus(promoteStatus, "Promoting model to production…");
      try {
        const out = await fetchJson("/api/model_promote", { method: "POST" }, 0);
        if (promoteStatus) setStatus(promoteStatus, `Saved as production • Default T=${Number(out.default_threshold ?? 0.5).toFixed(2)}`, "ok");
      } catch (e) {
        if (promoteStatus) setStatus(promoteStatus, `Promote failed: ${e.message}`, "bad");
      }
    });
  }

  // Quick thresholds buttons
  if (quickThresholds && thresholdEl) {
    quickThresholds.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-th]");
      if (!btn) return;
      const v = Number(btn.getAttribute("data-th"));
      if (Number.isFinite(v)) thresholdEl.value = v.toFixed(2);
    });
  }

  btnPredict.addEventListener("click", async () => {
    btnPredict.disabled = true;
    setStatus(predictStatus, "Predicting…");
    try {
      const payload = readForm(schema);
      const thr = thresholdEl ? Number(thresholdEl.value) : NaN;
      if (Number.isFinite(thr)) payload.threshold = thr;
      const out = await fetchJson("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      }, 1);

      const pred = out.prediction;
      const prob = out.probability;
      const pct = prob === null || prob === undefined ? NaN : (prob * 100);
      const risk = Number.isFinite(pct) ? `${pct.toFixed(1)}%` : "N/A";
      if (!Number.isFinite(pct)) {
        setStatus(predictStatus, `Prediction complete • Fault Risk Probability: ${risk}`, pred === 1 ? "bad" : "ok");
      } else if (pct >= 60) {
        setStatus(predictStatus, `🔴 High Risk • Fault Risk Probability: ${risk}. Immediate shutdown recommended.`, "bad");
      } else if (pct >= 30) {
        setStatus(predictStatus, `🟠 Moderate Risk • Fault Risk Probability: ${risk}. Preventive inspection advised.`, "bad");
      } else {
        setStatus(predictStatus, `System operating normally • Fault Risk Probability: ${risk}.`, "ok");
      }
    } catch (e) {
      setStatus(predictStatus, `Prediction failed: ${e.message}`, "bad");
    } finally {
      btnPredict.disabled = false;
    }
  });
}

main();
