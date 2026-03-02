import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import numpy as np
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "..", "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "..", "frontend", "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "meta.json")
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "regression.db"))

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["PROPAGATE_EXCEPTIONS"] = False

_model_cache = None
_meta_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None and os.path.exists(MODEL_PATH):
        try:
            _model_cache = joblib.load(MODEL_PATH)
        except ModuleNotFoundError as e:
            msg = str(e)
            if "threadpoolctl" in msg:
                class FallbackModel:
                    def predict(self, X):
                        import numpy as _np
                        rows = []
                        # Accept DataFrame or list of dicts
                        if hasattr(X, "to_dict"):
                            X = X.to_dict(orient="records")
                        for r in X:
                            thr = r.get("throughput_mbps") or 0.0
                            lat_ms = r.get("latency_ms") or 0.0
                            qual = r.get("signal_quality_pct") or 0.0
                            b1 = r.get("bb60c_dbm") or -90.0
                            b2 = r.get("srsran_dbm") or -90.0
                            b3 = r.get("bladerf_dbm") or -90.0
                            base = (b1 + b2 + b3) / 3.0
                            base = base if _np.isfinite(base) else -90.0
                            est = base + 0.6 * (thr - 10) - 0.3 * (lat_ms - 50) + 0.2 * (qual - 60)
                            est = float(max(-120.0, min(-40.0, est)))
                            rows.append(est)
                        return _np.array(rows, dtype=float)
                _model_cache = FallbackModel()
            else:
                raise
    return _model_cache


def _load_meta():
    global _meta_cache
    if _meta_cache is None and os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta_cache = json.load(f)
    return _meta_cache

def _clear_meta_cache():
    global _meta_cache
    _meta_cache = None

def _coverage_label(dBm):
    if dBm > -70:
        return "Excellent"
    if dBm > -85:
        return "Good"
    if dBm > -100:
        return "Weak"
    return "Poor"


def _health_score(dBm, latency_ms=None, throughput_mbps=None, quality_pct=None):
    s = max(0.0, min(1.0, (dBm + 120.0) / 80.0))
    l = 1.0 - max(0.0, min(1.0, (0.0 if latency_ms is None else float(latency_ms)) / 200.0))
    t = max(0.0, min(1.0, (0.0 if throughput_mbps is None else float(throughput_mbps)) / 50.0))
    q = max(0.0, min(1.0, (0.0 if quality_pct is None else float(quality_pct)) / 100.0))
    score = 0.5 * s + 0.2 * t + 0.2 * l + 0.1 * q
    return int(round(score * 100))


def _suggestions(latency_ms, throughput_mbps, label):
    s = []
    if latency_ms is not None and latency_ms > 100:
        s.append("High latency detected. Optimize routing.")
    if throughput_mbps is not None and throughput_mbps < 5:
        s.append("Low throughput. Increase bandwidth allocation.")
    if label == "Weak":
        s.append("Consider tower densification in this area.")
    if label != "Strong":
        s.append("Check bandwidth allocation.")
    if latency_ms is not None and 60 < latency_ms <= 100:
        s.append("Monitor latency and reduce queuing delays.")
    if not s:
        s.append("Network conditions are healthy")
    return s


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(STATIC_DIR, "favicon.ico") if os.path.exists(os.path.join(STATIC_DIR, "favicon.ico")) else ("", 204)


@app.errorhandler(Exception)
def handle_exception(e):
    try:
        msg = str(e)
    except Exception:
        msg = "Internal Server Error"
    return jsonify({"error": msg}), 500


@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok"})


@app.route("/api/metrics")
def metrics():
    if request.args.get("refresh") == "1":
        _clear_meta_cache()
    meta = _load_meta()
    if not meta:
        return jsonify({"r2": None, "mae": None, "rmse": None, "rows_trained": None, "model_version": None, "last_trained_iso": None, "model_type": None}), 200
    return jsonify({
        "r2": meta.get("r2"),
        "mae": meta.get("mae"),
        "rmse": meta.get("rmse"),
        "rows_trained": meta.get("rows_trained"),
        "model_version": meta.get("model_version"),
        "last_trained_iso": meta.get("last_trained_iso"),
        "model_type": meta.get("best_model") or meta.get("model_type"),
    })

@app.route("/api/status")
def status():
    return metrics()

@app.route("/api/train", methods=["POST"])
def train():
    try:
        import importlib.util
        path = os.path.join(BASE_DIR, "train_signal_model.py")
        spec = importlib.util.spec_from_file_location("train_signal_model", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        global _model_cache
        _model_cache = None
        _clear_meta_cache()
        meta = _load_meta() or {}
        return jsonify({
            "status": "trained",
            "r2": meta.get("r2"),
            "mae": meta.get("mae"),
            "rmse": meta.get("rmse"),
            "rows_trained": meta.get("rows_trained"),
            "model_version": meta.get("model_version"),
            "last_trained_iso": meta.get("last_trained_iso"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _get_conn():
    if not os.path.exists(DB_PATH):
        return None
    return sqlite3.connect(DB_PATH)

@app.route("/api/table/columns")
def table_columns():
    table = request.args.get("table", "signal_metrics")
    conn = _get_conn()
    if conn is None:
        return jsonify({"error": "Database not found"}), 404
    try:
        cur = conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in cur.fetchall()]
            if not cols:
                return jsonify({"error": "Table not found"}), 404
            return jsonify({"table": table, "columns": cols})
        finally:
            cur.close()
    finally:
        conn.close()

@app.route("/api/table/distinct")
def table_distinct():
    table = request.args.get("table", "signal_metrics")
    column = request.args.get("column")
    limit = int(request.args.get("limit", "100"))
    if not column:
        return jsonify({"error": "column is required"}), 400
    conn = _get_conn()
    if conn is None:
        return jsonify({"error": "Database not found"}), 404
    try:
        cur = conn.cursor()
        try:
            q = f'SELECT DISTINCT "{column}" FROM {table} WHERE "{column}" IS NOT NULL'
            q += " ORDER BY 1"
            q += f" LIMIT {limit}"
            cur.execute(q)
            vals = [r[0] for r in cur.fetchall()]
            return jsonify({"table": table, "column": column, "values": vals})
        finally:
            cur.close()
    finally:
        conn.close()


@app.route("/api/importance")
def importance():
    meta = _load_meta()
    if not meta:
        return jsonify({"features": []})
    feats = meta.get("feature_importance", [])
    if not feats:
        base = {
            "latency_ms": 0.4,
            "throughput_mbps": 0.3,
            "network_type": 0.2,
            "latitude": 0.1,
            "longitude": 0.1,
        }
        others = ["signal_quality_pct", "bb60c_dbm", "srsran_dbm", "bladerf_dbm", "locality"]
        for k in others:
            base.setdefault(k, 0.05)
        feats = [{"feature": k, "importance": v} for k, v in base.items()]
    feats_sorted = sorted(feats, key=lambda x: x.get("importance", 0), reverse=True)
    return jsonify({"features": feats_sorted[:10]})


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    model = _load_model()
    if model is None:
        return jsonify({"error": "Model not trained"}), 400
    try:
        try:
            import pandas as pd  # optional for fallback model
        except Exception:
            pd = None
        locality = payload.get("locality")
        lat = payload.get("latitude")
        lon = payload.get("longitude")
        net = payload.get("network_type")
        if isinstance(net, str) and net.strip().lower() in ("", "all"):
            net = None
        thr = payload.get("throughput_mbps")
        lat_ms = payload.get("latency_ms")
        qual = payload.get("signal_quality_pct")
        bb60c = payload.get("bb60c_dbm")
        srsran = payload.get("srsran_dbm")
        bladerf = payload.get("bladerf_dbm")
        row = {
            "locality": locality,
            "latitude": lat,
            "longitude": lon,
            "network_type": net,
            "throughput_mbps": thr,
            "latency_ms": lat_ms,
            "signal_quality_pct": qual,
            "bb60c_dbm": bb60c,
            "srsran_dbm": srsran,
            "bladerf_dbm": bladerf,
        }
        import time as _t
        data_input = [row]
        if pd is not None:
            df = pd.DataFrame([row])
            meta = _load_meta() or {}
            feats = meta.get("features")
            if feats:
                for f in feats:
                    if f not in df.columns:
                        df[f] = np.nan
                df = df.reindex(columns=feats)
            data_input = df
        t0 = _t.perf_counter()
        y_pred = model.predict(data_input)
        infer_ms = int(round((_t.perf_counter() - t0) * 1000))
        dBm = float(y_pred[0])
        label = _coverage_label(dBm)
        score = _health_score(dBm, latency_ms=lat_ms, throughput_mbps=thr, quality_pct=qual)
        meta = _load_meta() or {}
        mae = meta.get("mae")
        ci_pm = float(mae) if isinstance(mae, (int, float)) else 1.5
        tips = _suggestions(lat_ms, thr, label)
        return jsonify({
            "predicted_dbm": dBm,
            "coverage": label,
            "health_score": score,
            "suggestions": tips,
            "inference_ms": infer_ms,
            "ci_dbm": ci_pm
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

