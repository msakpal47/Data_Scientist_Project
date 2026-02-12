import io
import json
import os
import pickle
import sqlite3
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from werkzeug.exceptions import InternalServerError


BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Fault Detection in Manufacturing Processes using Sensor Data"
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "classification.db")
DEFAULT_DD_PATH = os.path.join(BASE_DIR, "Data Dictionary.txt")
DEFAULT_TABLE = "fault_detection_manufacturing"


def read_text_file(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def normalize_name(s: str) -> str:
    return "".join(s.lower().replace("_", "").split())


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    rows = cur.fetchall()
    return [r[0] for r in rows]


def resolve_required_table(conn: sqlite3.Connection, preferred: str) -> Optional[str]:
    tables = list_tables(conn)
    target = normalize_name(preferred)
    for t in tables:
        if normalize_name(t) == target:
            return t
    return None


def load_sql_range(conn: sqlite3.Connection, table: str, limit: int, offset: int) -> pd.DataFrame:
    return pd.read_sql_query(
        f"SELECT * FROM [{table}] ORDER BY rowid LIMIT {int(limit)} OFFSET {int(offset)}",
        conn,
    )


def load_sql_features(conn: sqlite3.Connection, table: str, cols: List[str], limit: int, offset: int) -> pd.DataFrame:
    sel = ", ".join([f"[{c}]" for c in cols])
    return pd.read_sql_query(
        f"SELECT {sel} FROM [{table}] ORDER BY rowid LIMIT {int(limit)} OFFSET {int(offset)}",
        conn,
    )


def transform_columns(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    dfc = df.copy()
    if "time" in features and "time" in dfc.columns:
        try:
            if not np.issubdtype(dfc["time"].dtype, np.number):
                dt = pd.to_datetime(dfc["time"], errors="coerce")
                dfc["time"] = (dt.view("int64") // 10**9).astype("float")
        except Exception:
            dfc["time"] = pd.Series(np.nan, index=dfc.index)
    return dfc


def impute_median_fit(df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    med = df[features].median(numeric_only=True)
    return {str(k): float(v) for k, v in med.to_dict().items()}


def impute_median_apply(df: pd.DataFrame, features: List[str], medians: Dict[str, float]) -> np.ndarray:
    x = df[features].copy()
    for c in features:
        fallback = float(x[c].median()) if c in x.columns else np.nan
        x[c] = x[c].fillna(medians.get(c, fallback))
    return x.values


def infer_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    excluded = {"time", "stage", "Lot", "runnum", "recipe", "recipe_step", target}
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    return [c for c in num_cols if c not in excluded]


def candidate_features(df: pd.DataFrame, target: str, numeric_only: bool, include_admin: bool) -> List[str]:
    admin = {"time", "stage", "Lot", "runnum", "recipe", "recipe_step", target}
    cols = list(df.columns)
    if not include_admin:
        cols = [c for c in cols if c not in admin]
    if numeric_only:
        nums = set(df.select_dtypes(include=[np.number]).columns)
        cols = [c for c in cols if c in nums or c == "time"]
    return cols


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": (None if np.isnan(roc) else float(roc)),
        "confusion_matrix": cm.tolist(),
    }


def model_feature_ranking(model, features: List[str]) -> Optional[List[Dict[str, float]]]:
    if hasattr(model, "feature_importances_"):
        imp = getattr(model, "feature_importances_")
        rows = [{"feature": features[i], "importance": float(imp[i])} for i in range(len(features))]
        rows.sort(key=lambda r: r["importance"], reverse=True)
        return rows[:40]
    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"))
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        coef = np.abs(coef).astype(float)
        rows = [{"feature": features[i], "importance": float(coef[i])} for i in range(len(features))]
        rows.sort(key=lambda r: r["importance"], reverse=True)
        return rows[:40]
    return None


def get_splits(total_rows: int, train_count: int = 500000, eval_count: int = 200000) -> Dict[str, int]:
    train_n = max(0, min(int(train_count), int(total_rows)))
    eval_n = max(0, min(int(eval_count), max(0, int(total_rows) - train_n)))
    live_offset = train_n + eval_n
    live_n = max(0, int(total_rows) - live_offset)
    return {"train_count": train_n, "eval_count": eval_n, "live_offset": live_offset, "live_count": live_n}


def serialize_model_payload(model, feature_names: List[str], medians: Dict[str, float]) -> bytes:
    payload = {"model": model, "features": feature_names, "medians": medians}
    return pickle.dumps(payload)


def predict_row(model, feature_names: List[str], row: pd.Series) -> Tuple[int, float]:
    x = row[feature_names].values.reshape(1, -1)
    try:
        proba = float(model.predict_proba(x)[0][1])
    except Exception:
        proba = float(model.predict(x)[0])
    pred = int(model.predict(x)[0])
    return pred, proba


def train_streaming_sgd(
    conn: sqlite3.Connection,
    table: str,
    features: List[str],
    target: str,
    train_count: int,
    chunk: int = 50000,
) -> Tuple[Optional[SGDClassifier], Optional[Dict[str, float]]]:
    if train_count <= 0:
        return None, None

    cols = features + [target]
    sample = load_sql_features(conn, table, cols, min(chunk, max(1, min(train_count, 100000))), 0)
    sample = transform_columns(sample, features)
    medians = impute_median_fit(sample, features)
    y_sample = pd.to_numeric(sample[target], errors="coerce").fillna(0)
    y_sample = (y_sample > 0).astype(int).values
    classes = np.array([0, 1], dtype=int)

    if (y_sample == 1).any() and (y_sample == 0).any():
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_sample)
    else:
        cw = np.array([1.0, 1.0], dtype=float)
    class_weight_map = {int(classes[i]): float(cw[i]) for i in range(len(classes))}

    clf = SGDClassifier(loss="log_loss", random_state=42)

    boot_X = None
    boot_y = None
    pos_found = (y_sample == 1).any()
    neg_found = (y_sample == 0).any()
    scan_offset = 0

    while not (pos_found and neg_found) and scan_offset < train_count:
        size = min(chunk, train_count - scan_offset)
        df_scan = load_sql_features(conn, table, cols, size, scan_offset)
        df_scan = transform_columns(df_scan, features)
        y_scan = pd.to_numeric(df_scan[target], errors="coerce").fillna(0)
        y_scan = (y_scan > 0).astype(int).values
        pos_found = pos_found or (y_scan == 1).any()
        neg_found = neg_found or (y_scan == 0).any()

        if pos_found and neg_found and boot_X is None:
            X_scan = impute_median_apply(df_scan, features, medians)
            idx_pos = int(np.where(y_scan == 1)[0][0]) if (y_scan == 1).any() else None
            idx_neg = int(np.where(y_scan == 0)[0][0]) if (y_scan == 0).any() else None
            rows = []
            labels = []
            if idx_pos is not None:
                rows.append(X_scan[idx_pos])
                labels.append(1)
            if idx_neg is not None:
                rows.append(X_scan[idx_neg])
                labels.append(0)
            if rows:
                boot_X = np.vstack(rows)
                boot_y = np.array(labels, dtype=int)

        scan_offset += size

    if boot_X is None:
        return None, None

    done = 0
    offset = 0
    while done < train_count:
        size = min(chunk, train_count - done)
        df = load_sql_features(conn, table, cols, size, offset)
        df = transform_columns(df, features)
        X = impute_median_apply(df, features, medians)
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        y = (y > 0).astype(int).values
        sw = np.array([class_weight_map.get(int(v), 1.0) for v in y], dtype=float)

        if done == 0:
            first_X = np.vstack([boot_X, X])
            first_y = np.concatenate([boot_y, y])
            sw_first = np.array([class_weight_map.get(int(v), 1.0) for v in first_y], dtype=float)
            clf.partial_fit(first_X, first_y, classes=classes, sample_weight=sw_first)
        else:
            clf.partial_fit(X, y, sample_weight=sw)

        done += size
        offset += size

    return clf, medians


def evaluate_streaming(
    conn: sqlite3.Connection,
    table: str,
    model,
    features: List[str],
    target: str,
    medians: Dict[str, float],
    eval_count: int,
    eval_offset: int,
    chunk: int = 50000,
) -> Dict:
    preds_all = []
    prob_all = []
    y_all = []

    done = 0
    cols = features + [target]
    offset = int(eval_offset)
    while done < eval_count:
        size = min(chunk, eval_count - done)
        df = load_sql_features(conn, table, cols, size, offset)
        df = transform_columns(df, features)
        X = impute_median_apply(df, features, medians)
        y = pd.to_numeric(df[target], errors="coerce").fillna(0)
        y = (y > 0).astype(int).values

        try:
            prob = model.predict_proba(X)[:, 1]
        except Exception:
            prob = model.predict(X).astype(float)

        preds = model.predict(X)
        preds_all.append(preds)
        prob_all.append(prob)
        y_all.append(y)
        done += size
        offset += size

    y_true = np.concatenate(y_all) if y_all else np.array([], dtype=int)
    y_pred = np.concatenate(preds_all) if preds_all else np.array([], dtype=int)
    y_prob = np.concatenate(prob_all) if prob_all else np.array([], dtype=float)
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    return evaluate_predictions(y_true, y_pred, y_prob)


@dataclass
class AppState:
    connected: bool = False
    db_path: str = DEFAULT_DB_PATH
    dd_path: str = DEFAULT_DD_PATH
    table_name: str = DEFAULT_TABLE
    total_rows: int = 0
    splits: Dict[str, int] = None
    model: Optional[object] = None
    features: List[str] = None
    target: Optional[str] = None
    medians: Dict[str, float] = None


STATE_LOCK = threading.Lock()
STATE = AppState(splits=get_splits(0), features=[])


def html_page() -> str:
    return f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Fault Detection Dashboard</title>
    <style>
      :root {{
        --bg: #0b1020;
        --panel: rgba(255,255,255,0.06);
        --panel2: rgba(255,255,255,0.08);
        --text: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.62);
        --accent: #7c3aed;
        --accent2: #22c55e;
        --danger: #ef4444;
        --border: rgba(255,255,255,0.14);
        --shadow: 0 10px 30px rgba(0,0,0,0.45);
        --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: var(--sans);
        color: var(--text);
        background: radial-gradient(1200px 800px at 20% 10%, rgba(124, 58, 237, 0.35), transparent 60%),
                    radial-gradient(900px 700px at 80% 30%, rgba(34, 197, 94, 0.22), transparent 55%),
                    radial-gradient(900px 600px at 40% 90%, rgba(59, 130, 246, 0.18), transparent 60%),
                    var(--bg);
        min-height: 100vh;
      }}
      a {{ color: inherit; }}
      .wrap {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 22px 18px 60px;
      }}
      .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 18px 18px;
        border: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
        box-shadow: var(--shadow);
        border-radius: 16px;
      }}
      .brand {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .brand h1 {{
        font-size: 18px;
        margin: 0;
        letter-spacing: 0.2px;
      }}
      .brand .sub {{
        font-size: 12px;
        color: var(--muted);
      }}
      .status {{
        display: flex;
        gap: 10px;
        align-items: center;
        flex-wrap: wrap;
        justify-content: flex-end;
      }}
      .pill {{
        padding: 8px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.06);
        color: var(--muted);
        font-size: 12px;
        display: inline-flex;
        gap: 8px;
        align-items: center;
      }}
      .dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--danger);
        box-shadow: 0 0 0 4px rgba(239,68,68,0.14);
      }}
      .dot.ok {{
        background: var(--accent2);
        box-shadow: 0 0 0 4px rgba(34,197,94,0.14);
      }}
      .grid {{
        display: grid;
        grid-template-columns: 360px 1fr;
        gap: 16px;
        margin-top: 16px;
      }}
      .panel {{
        border: 1px solid var(--border);
        background: var(--panel);
        border-radius: 16px;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .panel .hd {{
        padding: 14px 14px 10px;
        border-bottom: 1px solid var(--border);
        background: linear-gradient(180deg, rgba(255,255,255,0.06), transparent);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }}
      .panel .hd h2 {{
        margin: 0;
        font-size: 13px;
        color: rgba(255,255,255,0.84);
        letter-spacing: 0.35px;
        text-transform: uppercase;
      }}
      .panel .bd {{
        padding: 14px;
      }}
      .row {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 10px;
      }}
      label {{
        font-size: 12px;
        color: var(--muted);
        display: block;
        margin-bottom: 6px;
      }}
      input[type="text"], input[type="number"], select, textarea {{
        width: 100%;
        padding: 10px 10px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: rgba(0,0,0,0.20);
        color: var(--text);
        outline: none;
      }}
      textarea {{
        min-height: 140px;
        font-family: var(--mono);
        font-size: 12px;
        color: rgba(255,255,255,0.82);
      }}
      select[multiple] {{
        min-height: 160px;
      }}
      .btns {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
      }}
      button {{
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.07);
        color: var(--text);
        padding: 10px 12px;
        border-radius: 12px;
        cursor: pointer;
        transition: transform 0.06s ease, background 0.12s ease;
      }}
      button.primary {{
        background: rgba(124,58,237,0.22);
        border-color: rgba(124,58,237,0.45);
      }}
      button.good {{
        background: rgba(34,197,94,0.18);
        border-color: rgba(34,197,94,0.42);
      }}
      button:active {{
        transform: translateY(1px);
      }}
      button[disabled] {{
        opacity: 0.6;
        cursor: not-allowed;
      }}
      .tabs {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}
      .tab {{
        padding: 8px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.06);
        color: var(--muted);
        font-size: 12px;
        cursor: pointer;
        user-select: none;
      }}
      .tab.active {{
        color: rgba(255,255,255,0.92);
        border-color: rgba(124,58,237,0.5);
        background: rgba(124,58,237,0.18);
      }}
      .cards {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0 8px;
      }}
      .card {{
        padding: 12px 12px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.06);
        border-radius: 14px;
      }}
      .card .k {{
        font-size: 11px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.3px;
      }}
      .card .v {{
        margin-top: 4px;
        font-size: 18px;
        font-weight: 650;
      }}
      .split {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        align-items: start;
      }}
      .tablewrap {{
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(0,0,0,0.18);
      }}
      table {{
        border-collapse: collapse;
        width: 100%;
        min-width: 760px;
      }}
      th, td {{
        padding: 10px 10px;
        border-bottom: 1px solid rgba(255,255,255,0.10);
        font-size: 12px;
        color: rgba(255,255,255,0.82);
        white-space: nowrap;
        text-overflow: ellipsis;
        overflow: hidden;
        max-width: 260px;
      }}
      th {{
        position: sticky;
        top: 0;
        background: rgba(10, 15, 32, 0.85);
        color: rgba(255,255,255,0.88);
        border-bottom: 1px solid rgba(255,255,255,0.18);
        font-weight: 600;
      }}
      .bar {{
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        overflow: hidden;
      }}
      .bar > div {{
        height: 100%;
        background: linear-gradient(90deg, rgba(124,58,237,0.95), rgba(34,197,94,0.95));
        width: 0%;
      }}
      .chat {{
        display: flex;
        flex-direction: column;
        gap: 10px;
        height: 520px;
      }}
      .chatlog {{
        flex: 1;
        overflow: auto;
        padding: 10px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(0,0,0,0.18);
      }}
      .msg {{
        padding: 10px 10px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        margin-bottom: 10px;
      }}
      .msg .meta {{
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-bottom: 6px;
        font-size: 11px;
        color: var(--muted);
      }}
      .msg .txt {{
        font-family: var(--mono);
        font-size: 12px;
        color: rgba(255,255,255,0.86);
        white-space: pre-wrap;
        word-break: break-word;
      }}
      .small {{
        font-size: 12px;
        color: var(--muted);
        line-height: 1.45;
      }}
      @media (max-width: 980px) {{
        .grid {{ grid-template-columns: 1fr; }}
        .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        table {{ min-width: 680px; }}
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="topbar">
        <div class="brand">
          <h1>Fault Detection Dashboard</h1>
          <div class="sub">HTML + CSS + JS UI with JSON APIs (no Streamlit)</div>
        </div>
        <div class="status">
          <div class="pill"><span id="connDot" class="dot"></span><span id="connText">Not connected</span></div>
          <div class="pill"><span>Table</span><span id="tableName" style="color: rgba(255,255,255,0.82); font-family: var(--mono);">—</span></div>
          <div class="pill"><span>Rows</span><span id="rowCount" style="color: rgba(255,255,255,0.82); font-family: var(--mono);">—</span></div>
        </div>
      </div>

      <div class="grid">
        <div class="panel">
          <div class="hd">
            <h2>Control</h2>
            <div class="tabs">
              <div id="tabConnect" class="tab active" onclick="showPanel('connect')">Connect</div>
              <div id="tabTrain" class="tab" onclick="showPanel('train')">Train</div>
              <div id="tabInfer" class="tab" onclick="showPanel('infer')">Inference</div>
            </div>
          </div>
          <div class="bd">
            <div id="panelConnect">
              <div class="row">
                <div>
                  <label>SQLite DB Path</label>
                  <input id="dbPath" type="text" />
                </div>
                <div>
                  <label>Data Dictionary Path</label>
                  <input id="ddPath" type="text" />
                </div>
                <div class="btns">
                  <button id="btnConnect" class="primary" onclick="connect()">Connect</button>
                  <button onclick="refreshPreview()">Refresh Preview</button>
                </div>
                <div class="small">
                  Tip: keep DB on local disk for faster training. This UI trains on the first 500k rows, evaluates on the next 200k, and uses the remainder as live inference.
                </div>
              </div>
            </div>

            <div id="panelTrain" style="display:none">
              <div class="row">
                <div>
                  <label>Target Column</label>
                  <select id="targetSelect"></select>
                </div>
                <div>
                  <label>Feature Columns (multi-select)</label>
                  <select id="featureSelect" multiple></select>
                </div>
                <div class="split">
                  <div>
                    <label>Options</label>
                    <div class="btns">
                      <button id="toggleNumeric" onclick="toggleNumeric()">Numeric-only: ON</button>
                      <button id="toggleAdmin" onclick="toggleAdmin()">Admin fields: OFF</button>
                      <button id="toggleFast" class="good" onclick="toggleFast()">Fast mode: ON</button>
                    </div>
                  </div>
                  <div>
                    <label>Chunk Size</label>
                    <select id="chunkSelect">
                      <option value="5000">5,000</option>
                      <option value="10000" selected>10,000</option>
                      <option value="25000">25,000</option>
                      <option value="50000">50,000</option>
                    </select>
                  </div>
                </div>
                <div class="btns">
                  <button id="btnTrain" class="primary" onclick="train()">Train</button>
                  <button id="btnDownload" onclick="downloadModel()" disabled>Download Model</button>
                </div>
              </div>
            </div>

            <div id="panelInfer" style="display:none">
              <div class="row">
                <div>
                  <label>Live Row Index</label>
                  <input id="rowIndex" type="number" value="0" min="0" />
                </div>
                <div class="btns">
                  <button id="btnPredict" class="primary" onclick="predict()">Predict</button>
                </div>
                <div class="small" id="inferHint">Train a model first.</div>
              </div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="hd">
            <h2>Workspace</h2>
            <div class="tabs">
              <div id="tabData" class="tab active" onclick="showWorkspace('data')">Data</div>
              <div id="tabMetrics" class="tab" onclick="showWorkspace('metrics')">Metrics</div>
              <div id="tabChat" class="tab" onclick="showWorkspace('chat')">Chat</div>
            </div>
          </div>
          <div class="bd">
            <div id="wsData">
              <div class="split">
                <div>
                  <div class="small" style="margin-bottom:8px">Preview (first 50 rows)</div>
                  <div class="tablewrap" id="previewWrap"></div>
                </div>
                <div>
                  <div class="small" style="margin-bottom:8px">Data Dictionary</div>
                  <textarea id="ddText" readonly placeholder="Connect to load data dictionary"></textarea>
                </div>
              </div>
            </div>

            <div id="wsMetrics" style="display:none">
              <div id="metricCards" class="cards"></div>
              <div class="split">
                <div>
                  <div class="small" style="margin-bottom:8px">Confusion Matrix</div>
                  <div class="tablewrap" id="cmWrap"></div>
                </div>
                <div>
                  <div class="small" style="margin-bottom:8px">Top Feature Signals</div>
                  <div id="featWrap"></div>
                </div>
              </div>
            </div>

            <div id="wsChat" style="display:none">
              <div class="chat">
                <div id="chatlog" class="chatlog"></div>
                <div class="btns">
                  <button onclick="clearChat()">Clear</button>
                  <button class="good" onclick="addMsg('system', 'Try: Connect → Refresh Preview → Train → Predict')">Hint</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <script>
      const state = {{
        connected: false,
        tableName: null,
        rowCount: 0,
        splits: null,
        columns: [],
        numericOnly: true,
        includeAdmin: false,
        fastMode: true,
        modelReady: false,
      }};

      function now() {{
        const d = new Date();
        return d.toLocaleTimeString();
      }}

      function addMsg(who, txt) {{
        const log = document.getElementById('chatlog');
        const el = document.createElement('div');
        el.className = 'msg';
        el.innerHTML = `
          <div class="meta"><span>${{who}}</span><span>${{now()}}</span></div>
          <div class="txt"></div>
        `;
        el.querySelector('.txt').textContent = txt;
        log.appendChild(el);
        log.scrollTop = log.scrollHeight;
      }}

      function clearChat() {{
        document.getElementById('chatlog').innerHTML = '';
      }}

      function setConnected(isConnected) {{
        state.connected = isConnected;
        const dot = document.getElementById('connDot');
        const txt = document.getElementById('connText');
        dot.className = isConnected ? 'dot ok' : 'dot';
        txt.textContent = isConnected ? 'Connected' : 'Not connected';
      }}

      function showPanel(which) {{
        document.getElementById('panelConnect').style.display = which === 'connect' ? '' : 'none';
        document.getElementById('panelTrain').style.display = which === 'train' ? '' : 'none';
        document.getElementById('panelInfer').style.display = which === 'infer' ? '' : 'none';
        document.getElementById('tabConnect').classList.toggle('active', which === 'connect');
        document.getElementById('tabTrain').classList.toggle('active', which === 'train');
        document.getElementById('tabInfer').classList.toggle('active', which === 'infer');
      }}

      function showWorkspace(which) {{
        document.getElementById('wsData').style.display = which === 'data' ? '' : 'none';
        document.getElementById('wsMetrics').style.display = which === 'metrics' ? '' : 'none';
        document.getElementById('wsChat').style.display = which === 'chat' ? '' : 'none';
        document.getElementById('tabData').classList.toggle('active', which === 'data');
        document.getElementById('tabMetrics').classList.toggle('active', which === 'metrics');
        document.getElementById('tabChat').classList.toggle('active', which === 'chat');
      }}

      function setConfig(cfg) {{
        document.getElementById('dbPath').value = cfg.db_path || '';
        document.getElementById('ddPath').value = cfg.dd_path || '';
        document.getElementById('tableName').textContent = cfg.table_name || '—';
      }}

      function formatPct(v) {{
        if (v === null || v === undefined) return '—';
        return (Math.round(v * 1000) / 10).toFixed(1) + '%';
      }}

      function renderPreview(columns, rows) {{
        const wrap = document.getElementById('previewWrap');
        if (!columns || columns.length === 0) {{
          wrap.innerHTML = '<div class="small">No preview loaded.</div>';
          return;
        }}
        let html = '<table><thead><tr>';
        for (const c of columns) html += `<th>${{c}}</th>`;
        html += '</tr></thead><tbody>';
        for (const r of rows) {{
          html += '<tr>';
          for (const c of columns) {{
            const val = r[c];
            html += `<td title="${{String(val ?? '')}}">${{val ?? ''}}</td>`;
          }}
          html += '</tr>';
        }}
        html += '</tbody></table>';
        wrap.innerHTML = html;
      }}

      function renderMetrics(m) {{
        const cards = document.getElementById('metricCards');
        cards.innerHTML = '';
        const items = [
          ['Accuracy', formatPct(m.accuracy)],
          ['Precision', formatPct(m.precision)],
          ['Recall', formatPct(m.recall)],
          ['F1', formatPct(m.f1)],
        ];
        for (const [k, v] of items) {{
          const el = document.createElement('div');
          el.className = 'card';
          el.innerHTML = `<div class="k">${{k}}</div><div class="v">${{v}}</div>`;
          cards.appendChild(el);
        }}

        const cm = m.confusion_matrix || [[0,0],[0,0]];
        const cmWrap = document.getElementById('cmWrap');
        cmWrap.innerHTML = `
          <table style="min-width: 420px">
            <thead>
              <tr><th></th><th>Pred 0</th><th>Pred 1</th></tr>
            </thead>
            <tbody>
              <tr><td>Actual 0</td><td>${{cm[0][0]}}</td><td>${{cm[0][1]}}</td></tr>
              <tr><td>Actual 1</td><td>${{cm[1][0]}}</td><td>${{cm[1][1]}}</td></tr>
            </tbody>
          </table>
        `;

        const featWrap = document.getElementById('featWrap');
        const feats = m.feature_ranking || [];
        if (!feats || feats.length === 0) {{
          featWrap.innerHTML = '<div class="small">No feature ranking available for this model.</div>';
          return;
        }}
        const maxImp = Math.max(...feats.map(x => x.importance || 0), 1e-9);
        let html = '';
        for (const f of feats.slice(0, 20)) {{
          const pct = Math.max(0, Math.min(100, (f.importance / maxImp) * 100));
          html += `
            <div class="card" style="margin-bottom:10px">
              <div class="k">${{f.feature}}</div>
              <div style="display:flex; justify-content:space-between; gap:10px; align-items:center; margin-top:6px">
                <div class="bar" style="flex:1"><div style="width:${{pct}}%"></div></div>
                <div style="font-family: var(--mono); font-size: 12px; color: rgba(255,255,255,0.80)">${{(Math.round(f.importance * 10000) / 10000).toFixed(4)}}</div>
              </div>
            </div>
          `;
        }}
        featWrap.innerHTML = html;
      }}

      async function api(path, opts) {{
        const res = await fetch(path, opts);
        const ct = res.headers.get('content-type') || '';
        const data = ct.includes('application/json') ? await res.json() : await res.text();
        if (!res.ok) {{
          const msg = (data && data.error) ? data.error : (typeof data === 'string' ? data : 'Request failed');
          throw new Error(msg);
        }}
        return data;
      }}

      async function loadConfig() {{
        const cfg = await api('/api/config');
        setConfig(cfg);
        document.getElementById('tableName').textContent = cfg.table_name || '—';
        addMsg('system', 'Loaded defaults. Click Connect to start.');
      }}

      async function connect() {{
        const btn = document.getElementById('btnConnect');
        btn.disabled = true;
        try {{
          addMsg('user', 'Connect');
          const payload = {{
            db_path: document.getElementById('dbPath').value,
            dd_path: document.getElementById('ddPath').value,
          }};
          const out = await api('/api/connect', {{
            method: 'POST',
            headers: {{ 'content-type': 'application/json' }},
            body: JSON.stringify(payload),
          }});
          setConnected(true);
          document.getElementById('tableName').textContent = out.table_name;
          document.getElementById('rowCount').textContent = String(out.total_rows);
          state.tableName = out.table_name;
          state.rowCount = out.total_rows;
          state.splits = out.splits;
          addMsg('system', `Connected. Table=${{out.table_name}}, total_rows=${{out.total_rows}}`);
          await refreshPreview();
          await refreshDictionary();
          await refreshColumns();
          document.getElementById('inferHint').textContent = `Live rows: ${{out.splits.live_count}} (offset ${{out.splits.live_offset}})`;
        }} catch (e) {{
          setConnected(false);
          addMsg('system', `Connect failed: ${{e.message}}`);
          alert(e.message);
        }} finally {{
          btn.disabled = false;
        }}
      }}

      async function refreshDictionary() {{
        try {{
          const txt = await api('/api/data_dictionary');
          document.getElementById('ddText').value = txt || '';
        }} catch (e) {{
          document.getElementById('ddText').value = '';
        }}
      }}

      async function refreshPreview() {{
        try {{
          const out = await api('/api/preview?limit=50');
          state.columns = out.columns || [];
          renderPreview(out.columns || [], out.rows || []);
          addMsg('system', 'Loaded data preview.');
        }} catch (e) {{
          renderPreview([], []);
          addMsg('system', `Preview failed: ${{e.message}}`);
        }}
      }}

      function selectedValues(selectEl) {{
        return Array.from(selectEl.selectedOptions).map(o => o.value);
      }}

      function fillSelect(selectEl, options, defaultValue) {{
        selectEl.innerHTML = '';
        for (const opt of options) {{
          const o = document.createElement('option');
          o.value = opt;
          o.textContent = opt;
          if (defaultValue && opt === defaultValue) o.selected = true;
          selectEl.appendChild(o);
        }}
      }}

      async function refreshColumns() {{
        const out = await api('/api/columns');
        const cols = out.columns || [];
        fillSelect(document.getElementById('targetSelect'), cols, out.default_target || cols[0]);
        await refreshFeatures();
      }}

      async function refreshFeatures() {{
        const target = document.getElementById('targetSelect').value;
        const out = await api(`/api/features?target=${{encodeURIComponent(target)}}&numeric_only=${{state.numericOnly ? 1 : 0}}&include_admin=${{state.includeAdmin ? 1 : 0}}`);
        const feats = out.features || [];
        const sel = document.getElementById('featureSelect');
        sel.innerHTML = '';
        const defaults = out.default_features || [];
        for (const f of feats) {{
          const o = document.createElement('option');
          o.value = f;
          o.textContent = f;
          if (defaults.includes(f)) o.selected = true;
          sel.appendChild(o);
        }}
        addMsg('system', `Loaded feature list (${{feats.length}}).`);
      }}

      function toggleNumeric() {{
        state.numericOnly = !state.numericOnly;
        document.getElementById('toggleNumeric').textContent = `Numeric-only: ${{state.numericOnly ? 'ON' : 'OFF'}}`;
        refreshFeatures();
      }}

      function toggleAdmin() {{
        state.includeAdmin = !state.includeAdmin;
        document.getElementById('toggleAdmin').textContent = `Admin fields: ${{state.includeAdmin ? 'ON' : 'OFF'}}`;
        refreshFeatures();
      }}

      function toggleFast() {{
        state.fastMode = !state.fastMode;
        document.getElementById('toggleFast').textContent = `Fast mode: ${{state.fastMode ? 'ON' : 'OFF'}}`;
        document.getElementById('toggleFast').classList.toggle('good', state.fastMode);
      }}

      async function train() {{
        if (!state.connected) {{
          alert('Connect first.');
          return;
        }}
        const btn = document.getElementById('btnTrain');
        btn.disabled = true;
        try {{
          showWorkspace('metrics');
          addMsg('user', 'Train model');
          const payload = {{
            target: document.getElementById('targetSelect').value,
            features: selectedValues(document.getElementById('featureSelect')),
            fast_mode: state.fastMode,
            chunk_size: parseInt(document.getElementById('chunkSelect').value, 10),
          }};
          if (!payload.features || payload.features.length === 0) {{
            alert('Select at least one feature.');
            return;
          }}
          addMsg('system', `Training started (fast_mode=${{payload.fast_mode}}, chunk=${{payload.chunk_size}})...`);
          const out = await api('/api/train', {{
            method: 'POST',
            headers: {{ 'content-type': 'application/json' }},
            body: JSON.stringify(payload),
          }});
          renderMetrics(out.metrics);
          state.modelReady = true;
          document.getElementById('btnDownload').disabled = false;
          addMsg('system', 'Training completed. Metrics updated.');
        }} catch (e) {{
          addMsg('system', `Training failed: ${{e.message}}`);
          alert(e.message);
        }} finally {{
          btn.disabled = false;
        }}
      }}

      function downloadModel() {{
        window.location.href = '/download/model';
      }}

      async function predict() {{
        if (!state.modelReady) {{
          alert('Train a model first.');
          return;
        }}
        const btn = document.getElementById('btnPredict');
        btn.disabled = true;
        try {{
          addMsg('user', 'Predict');
          const idx = parseInt(document.getElementById('rowIndex').value || '0', 10);
          const out = await api(`/api/predict?index=${{idx}}`);
          showWorkspace('chat');
          addMsg('system', `predicted_fault=${{out.predicted_fault}}, probability=${{out.probability}}`);
          addMsg('system', JSON.stringify(out.row, null, 2));
        }} catch (e) {{
          addMsg('system', `Predict failed: ${{e.message}}`);
          alert(e.message);
        }} finally {{
          btn.disabled = false;
        }}
      }}

      document.getElementById('targetSelect').addEventListener('change', refreshFeatures);
      loadConfig();
    </script>
  </body>
</html>
"""


app = Flask(__name__)

@app.errorhandler(InternalServerError)
def handle_500(e: InternalServerError):
    import traceback

    orig = getattr(e, "original_exception", None)
    if orig is None:
        return Response(repr(e), mimetype="text/plain"), 500
    return Response("".join(traceback.format_exception(type(orig), orig, orig.__traceback__)), mimetype="text/plain"), 500

@app.errorhandler(Exception)
def handle_exception(e: Exception):
    import traceback

    return Response(traceback.format_exc(), mimetype="text/plain"), 500


@app.get("/")
def index() -> Response:
    try:
        return Response(render_template("index.html"), mimetype="text/html")
    except Exception as e:
        return Response(f"Index error: {e}", mimetype="text/plain"), 500


@app.get("/api/config")
def api_config():
    return jsonify({"db_path": DEFAULT_DB_PATH, "dd_path": DEFAULT_DD_PATH, "table_name": DEFAULT_TABLE})


@app.get("/api/boom")
def api_boom():
    raise RuntimeError("boom")


@app.post("/api/connect")
def api_connect():
    payload = request.get_json(silent=True) or {}
    db_path = str(payload.get("db_path") or DEFAULT_DB_PATH)
    dd_path = str(payload.get("dd_path") or DEFAULT_DD_PATH)

    if not os.path.exists(db_path):
        return jsonify({"error": "Database file not found"}), 400

    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        return jsonify({"error": f"Connection error: {e}"}), 400

    try:
        table_name = resolve_required_table(conn, DEFAULT_TABLE)
        if table_name is None:
            return jsonify({"error": f"Required table not found: {DEFAULT_TABLE}"}), 400
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(1) FROM [{table_name}]")
        total_rows = int(cur.fetchone()[0])
    finally:
        conn.close()

    splits = get_splits(total_rows)
    with STATE_LOCK:
        STATE.connected = True
        STATE.db_path = db_path
        STATE.dd_path = dd_path
        STATE.table_name = table_name
        STATE.total_rows = total_rows
        STATE.splits = splits
        STATE.model = None
        STATE.features = []
        STATE.target = None
        STATE.medians = None

    return jsonify({"table_name": table_name, "total_rows": total_rows, "splits": splits})


@app.get("/api/data_dictionary")
def api_data_dictionary():
    with STATE_LOCK:
        dd_path = STATE.dd_path
    return Response(read_text_file(dd_path), mimetype="text/plain")


@app.get("/api/preview")
def api_preview():
    limit = int(request.args.get("limit") or 50)
    limit = max(1, min(200, limit))

    with STATE_LOCK:
        if not STATE.connected:
            return jsonify({"error": "Not connected"}), 400
        db_path = STATE.db_path
        table_name = STATE.table_name

    conn = sqlite3.connect(db_path)
    try:
        df = load_sql_range(conn, table_name, limit, 0)
    finally:
        conn.close()

    df = df.replace({np.nan: None})
    return jsonify({"columns": list(df.columns), "rows": df.to_dict(orient="records")})


@app.get("/api/columns")
def api_columns():
    with STATE_LOCK:
        if not STATE.connected:
            return jsonify({"error": "Not connected"}), 400
        db_path = STATE.db_path
        table_name = STATE.table_name

    conn = sqlite3.connect(db_path)
    try:
        df = load_sql_range(conn, table_name, 200, 0)
    finally:
        conn.close()

    cols = list(df.columns)
    default_target = "fault_occurred" if "fault_occurred" in cols else (cols[0] if cols else None)
    return jsonify({"columns": cols, "default_target": default_target})


@app.get("/api/features")
def api_features():
    target = str(request.args.get("target") or "")
    numeric_only = str(request.args.get("numeric_only") or "1") in {"1", "true", "True", "yes"}
    include_admin = str(request.args.get("include_admin") or "0") in {"1", "true", "True", "yes"}

    with STATE_LOCK:
        if not STATE.connected:
            return jsonify({"error": "Not connected"}), 400
        db_path = STATE.db_path
        table_name = STATE.table_name

    conn = sqlite3.connect(db_path)
    try:
        df = load_sql_range(conn, table_name, 1000, 0)
    finally:
        conn.close()

    if not target or target not in df.columns:
        return jsonify({"error": "Invalid target column"}), 400

    feats = candidate_features(df, target, numeric_only=numeric_only, include_admin=include_admin)
    defaults = [f for f in infer_feature_columns(df, target) if f in feats]
    return jsonify({"features": feats, "default_features": defaults})


@app.post("/api/train")
def api_train():
    payload = request.get_json(silent=True) or {}
    target = str(payload.get("target") or "")
    features = payload.get("features") or []
    features = [str(f) for f in features if str(f).strip()]
    fast_mode = bool(payload.get("fast_mode", True))
    chunk_size = int(payload.get("chunk_size") or 10000)
    chunk_size = max(1000, min(50000, chunk_size))

    with STATE_LOCK:
        if not STATE.connected:
            return jsonify({"error": "Not connected"}), 400
        db_path = STATE.db_path
        table_name = STATE.table_name
        splits = STATE.splits or get_splits(0)

    if not features:
        return jsonify({"error": "No features selected"}), 400

    conn = sqlite3.connect(db_path)
    try:
        df_head = load_sql_range(conn, table_name, 200, 0)
        if target not in df_head.columns:
            return jsonify({"error": "Target column not found in table"}), 400
        for f in features:
            if f not in df_head.columns:
                return jsonify({"error": f"Feature column not found: {f}"}), 400

        if fast_mode:
            model, medians = train_streaming_sgd(
                conn,
                table_name,
                features,
                target,
                train_count=int(splits["train_count"]),
                chunk=chunk_size,
            )
            if model is None or medians is None:
                return jsonify({"error": "Training failed (insufficient label variety or data)"}), 400
            metrics = evaluate_streaming(
                conn,
                table_name,
                model,
                features,
                target,
                medians,
                eval_count=int(splits["eval_count"]),
                eval_offset=int(splits["train_count"]),
                chunk=chunk_size,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier

            train_df = load_sql_features(conn, table_name, features + [target], int(splits["train_count"]), 0)
            eval_df = load_sql_features(
                conn,
                table_name,
                features + [target],
                int(splits["eval_count"]),
                int(splits["train_count"]),
            )
            train_df = transform_columns(train_df, features)
            eval_df = transform_columns(eval_df, features)
            medians = impute_median_fit(train_df, features)
            X_train = impute_median_apply(train_df, features, medians)
            y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0)
            y_train = (y_train > 0).astype(int).values
            X_eval = impute_median_apply(eval_df, features, medians)
            y_eval = pd.to_numeric(eval_df[target], errors="coerce").fillna(0)
            y_eval = (y_eval > 0).astype(int).values

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced",
            )
            model.fit(X_train, y_train)
            try:
                y_prob = model.predict_proba(X_eval)[:, 1]
            except Exception:
                y_prob = model.predict(X_eval).astype(float)
            y_pred = model.predict(X_eval)
            metrics = evaluate_predictions(y_eval, y_pred, y_prob)

        metrics["feature_ranking"] = model_feature_ranking(model, features)
    finally:
        conn.close()

    with STATE_LOCK:
        STATE.model = model
        STATE.features = features
        STATE.target = target
        STATE.medians = medians

    return jsonify({"metrics": metrics})


@app.get("/api/predict")
def api_predict():
    idx = int(request.args.get("index") or 0)
    if idx < 0:
        return jsonify({"error": "index must be >= 0"}), 400

    with STATE_LOCK:
        if not STATE.connected:
            return jsonify({"error": "Not connected"}), 400
        if STATE.model is None or not STATE.features or STATE.medians is None:
            return jsonify({"error": "Model not trained"}), 400
        db_path = STATE.db_path
        table_name = STATE.table_name
        splits = STATE.splits or get_splits(0)
        live_offset = int(splits.get("live_offset") or 0)
        live_count = int(splits.get("live_count") or 0)
        model = STATE.model
        features = list(STATE.features)
        medians = dict(STATE.medians)

    if live_count <= 0:
        return jsonify({"error": "No live rows available"}), 400

    if idx >= live_count:
        return jsonify({"error": f"index out of range (max {live_count - 1})"}), 400

    conn = sqlite3.connect(db_path)
    try:
        df = load_sql_range(conn, table_name, 1, live_offset + idx)
    finally:
        conn.close()

    if df.empty:
        return jsonify({"error": "No data at this index"}), 400

    row = df.iloc[0].copy()
    df_t = transform_columns(df, features)
    X = impute_median_apply(df_t, features, medians)
    try:
        proba = float(model.predict_proba(X)[0][1])
    except Exception:
        proba = float(model.predict(X)[0])
    pred = int(model.predict(X)[0])

    row_out = row.replace({np.nan: None}).to_dict()
    return jsonify({"predicted_fault": pred, "probability": round(proba, 6), "row": row_out})


@app.get("/download/model")
def download_model():
    with STATE_LOCK:
        if STATE.model is None or not STATE.features or STATE.medians is None:
            return jsonify({"error": "Model not trained"}), 400
        blob = serialize_model_payload(STATE.model, list(STATE.features), dict(STATE.medians))

    return Response(
        blob,
        mimetype="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="fault_detection_model.pkl"'},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=False)

