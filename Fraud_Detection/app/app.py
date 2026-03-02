import os
import sys
import sqlite3
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, Response
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_transaction, load_model as _load_model

def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DB_PATH = os.path.join(project_root(), "data", "classification.db")
MODELS_DIR = os.path.join(project_root(), "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURE_IMPORTANCES_PATH = os.path.join(MODELS_DIR, "feature_importances.json")
PR_CURVE_PATH = os.path.join(MODELS_DIR, "pr_curve.json")
TEST_EVAL_PATH = os.path.join(MODELS_DIR, "test_eval.json")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
model = _load_model()

def list_tables(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cur.fetchall()
    return [r[0] for r in rows]

def load_data(limit_rows: int | None = 200000) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        tables = list_tables(conn)
        if not tables:
            raise RuntimeError("No tables found in classification.db")
        table = tables[0]
        query = f"SELECT * FROM {table}"
        if limit_rows is not None:
            query += f" LIMIT {int(limit_rows)}"
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

app = Flask(__name__, template_folder="templates", static_folder="static")
data_frame: pd.DataFrame = load_data()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/summary")
def api_summary():
    tx_type = request.args.get("tx_type")
    if "isFraud" not in data_frame.columns:
        return jsonify({"non_fraud": 0, "fraud": 0, "fraud_rate": 0.0, "flagged": 0, "flagged_rate": 0.0, "imbalance_ratio": "1:0"})
    df_local = data_frame
    if tx_type and "type" in df_local.columns:
        df_local = df_local[df_local["type"] == tx_type]
    is_fraud_num = pd.to_numeric(df_local["isFraud"], errors="coerce").fillna(0).astype(int)
    non_fraud = int((is_fraud_num == 0).sum())
    fraud = int((is_fraud_num == 1).sum())
    total = non_fraud + fraud
    rate = float(fraud / total) if total > 0 else 0.0
    ratio = "∞" if fraud == 0 else f"1:{int(round(non_fraud / max(1, fraud)))}"
    flagged = 0
    flagged_rate = 0.0
    if "isFlaggedFraud" in df_local.columns:
        flagged_num = pd.to_numeric(df_local["isFlaggedFraud"], errors="coerce").fillna(0).astype(int)
        flagged = int((flagged_num == 1).sum())
        flagged_rate = float(flagged / len(df_local)) if len(df_local) > 0 else 0.0
    return jsonify({"non_fraud": non_fraud, "fraud": fraud, "fraud_rate": rate, "flagged": flagged, "flagged_rate": flagged_rate, "imbalance_ratio": ratio})

@app.route("/api/transactions")
def api_transactions():
    filter_value = request.args.get("filter", "all")
    limit = request.args.get("limit", type=int)
    page = request.args.get("page", type=int, default=1)
    tx_type = request.args.get("tx_type")
    df = data_frame
    if tx_type and "type" in df.columns:
        df = df[df["type"] == tx_type]
    if "isFraud" in df.columns:
        is_fraud_num = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)
        if filter_value == "fraud":
            df = df[is_fraud_num == 1]
        elif filter_value == "nonfraud":
            df = df[is_fraud_num == 0]
    if filter_value == "flagged" and "isFlaggedFraud" in df.columns:
        flagged_num = pd.to_numeric(df["isFlaggedFraud"], errors="coerce").fillna(0).astype(int)
        df = df[flagged_num == 1]
    total_rows = int(df.shape[0])
    if limit and limit > 0:
        start = max(0, (page - 1) * limit)
        end = start + limit
        df = df.iloc[start:end]
    cols = list(df.columns)
    df_safe = df.replace([np.inf, -np.inf], np.nan)
    rows = json.loads(df_safe.to_json(orient="records"))
    return jsonify({"columns": cols, "rows": rows, "total_rows": total_rows})

@app.route("/api/export")
def api_export():
    filter_value = request.args.get("filter", "all")
    tx_type = request.args.get("tx_type")
    df = data_frame
    if tx_type and "type" in df.columns:
        df = df[df["type"] == tx_type]
    if "isFraud" in df.columns:
        is_fraud_num = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)
        if filter_value == "fraud":
            df = df[is_fraud_num == 1]
        elif filter_value == "nonfraud":
            df = df[is_fraud_num == 0]
    if filter_value == "flagged" and "isFlaggedFraud" in df.columns:
        flagged_num = pd.to_numeric(df["isFlaggedFraud"], errors="coerce").fillna(0).astype(int)
        df = df[flagged_num == 1]
    df_safe = df.replace([np.inf, -np.inf], np.nan)
    csv_data = df_safe.to_csv(index=False)
    filename = f"{filter_value}_transactions.csv"
    return Response(csv_data, mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    threshold = float(payload.get("threshold", 0.5))
    result = predict_transaction(payload, threshold=threshold)
    try:
        log_path = os.path.join(project_root(), "data", "predictions_log.csv")
        row = [
            pd.Timestamp.now().isoformat(),
            payload.get("type", "TRANSFER"),
            float(payload.get("amount", 0.0)),
            float(result.get("probability", 0.0)),
            int(result.get("label", 0))
        ]
        header = "timestamp,type,amount,probability,label"
        if not os.path.exists(log_path):
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(header + "\n")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(",".join(map(str, row)) + "\n")
    except Exception:
        pass
    return jsonify(result)

@app.route("/api/metrics")
def api_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as fmet:
            return jsonify(json.load(fmet))
    return jsonify({"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0, "confusion_matrix": [[0, 0], [0, 0]], "count_test": 0})

@app.route("/api/feature_importances")
def api_feature_importances():
    if os.path.exists(FEATURE_IMPORTANCES_PATH):
        with open(FEATURE_IMPORTANCES_PATH, "r", encoding="utf-8") as fimps:
            return jsonify(json.load(fimps))
    return jsonify({"importances": []})

@app.route("/api/threshold_suggestion")
def api_threshold_suggestion():
    if os.path.exists(PR_CURVE_PATH):
        with open(PR_CURVE_PATH, "r", encoding="utf-8") as fpr:
            return jsonify(json.load(fpr).get("suggestions", {}))
    return jsonify({"best_f1": {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}})

@app.route("/api/pr_curve")
def api_pr_curve():
    if os.path.exists(PR_CURVE_PATH):
        with open(PR_CURVE_PATH, "r", encoding="utf-8") as fpr:
            return jsonify(json.load(fpr))
    return jsonify({"curve": [], "suggestions": {}})

@app.route("/api/confusion_sim")
def api_confusion_sim():
    threshold = float(request.args.get("threshold", 0.5))
    if os.path.exists(TEST_EVAL_PATH):
        with open(TEST_EVAL_PATH, "r", encoding="utf-8") as fev:
            payload = json.load(fev)
            y_test = payload.get("y_test", [])
            y_prob = payload.get("y_prob", [])
            if len(y_test) == len(y_prob) and len(y_test) > 0:
                y_pred = [1 if float(p) >= threshold else 0 for p in y_prob]
                tn = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 0 and yp == 0)
                fp = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 0 and yp == 1)
                fn = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 1 and yp == 0)
                tp = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 1 and yp == 1)
                return jsonify({"confusion_matrix": [[tn, fp], [fn, tp]], "count_test": len(y_test)})
    # Fallback: compute from current dataset
    if "isFraud" in data_frame.columns:
        df = data_frame.copy()
        y = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int).tolist()
        X_raw = df.drop(columns=["isFraud"])
        feature_cols = _load_feature_columns()
        X_proc = preprocess(X_raw, expected_cols=feature_cols if feature_cols else None)
        probs = model.predict_proba(X_proc)[:, 1].tolist()
        y_pred = [1 if float(p) >= threshold else 0 for p in probs]
        tn = sum(1 for yt, yp in zip(y, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y, y_pred) if yt == 1 and yp == 0)
        tp = sum(1 for yt, yp in zip(y, y_pred) if yt == 1 and yp == 1)
        return jsonify({"confusion_matrix": [[tn, fp], [fn, tp]], "count_test": len(y)})
    return jsonify({"confusion_matrix": [[0, 0], [0, 0]], "count_test": 0})

@app.route("/api/cost_sim")
def api_cost_sim():
    threshold = float(request.args.get("threshold", 0.5))
    cost_fp = float(request.args.get("cost_fp", 5.0))
    cost_fn = float(request.args.get("cost_fn", 500.0))
    data = api_confusion_sim().get_json()
    cm = data.get("confusion_matrix", [[0, 0], [0, 0]])
    fp = int((cm[0] or [0, 0])[1])
    fn = int((cm[1] or [0, 0])[0])
    expected_loss = fp * cost_fp + fn * cost_fn
    data["expected_loss"] = expected_loss
    return jsonify(data)

@app.route("/api/optimal_threshold")
def api_optimal_threshold():
    cost_fp = float(request.args.get("cost_fp", 5.0))
    cost_fn = float(request.args.get("cost_fn", 500.0))
    best_th = 0.5
    best_loss = float("inf")
    if os.path.exists(TEST_EVAL_PATH):
        with open(TEST_EVAL_PATH, "r", encoding="utf-8") as fev:
            payload = json.load(fev)
            y_test = payload.get("y_test", [])
            y_prob = payload.get("y_prob", [])
            if len(y_test) == len(y_prob) and len(y_test) > 0:
                for t in [i/100 for i in range(0, 101)]:
                    y_pred = [1 if float(p) >= t else 0 for p in y_prob]
                    fp = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 0 and yp == 1)
                    fn = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 1 and yp == 0)
                    loss = fp * cost_fp + fn * cost_fn
                    if loss < best_loss:
                        best_loss = loss
                        best_th = t
                return jsonify({"optimal_threshold": best_th, "min_expected_loss": best_loss})
    if "isFraud" in data_frame.columns:
        df = data_frame.copy()
        y = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int).tolist()
        X_raw = df.drop(columns=["isFraud"])
        feature_cols = _load_feature_columns()
        X_proc = preprocess(X_raw, expected_cols=feature_cols if feature_cols else None)
        probs = model.predict_proba(X_proc)[:, 1].tolist()
        for t in [i/100 for i in range(0, 101)]:
            y_pred = [1 if float(p) >= t else 0 for p in probs]
            fp = sum(1 for yt, yp in zip(y, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y, y_pred) if yt == 1 and yp == 0)
            loss = fp * cost_fp + fn * cost_fn
            if loss < best_loss:
                best_loss = loss
                best_th = t
        return jsonify({"optimal_threshold": best_th, "min_expected_loss": best_loss})
    return jsonify({"optimal_threshold": 0.5, "min_expected_loss": 0.0})

@app.route("/api/cost_curve")
def api_cost_curve():
    cost_fp = float(request.args.get("cost_fp", 5.0))
    cost_fn = float(request.args.get("cost_fn", 500.0))
    points = []
    if os.path.exists(TEST_EVAL_PATH):
        with open(TEST_EVAL_PATH, "r", encoding="utf-8") as fev:
            payload = json.load(fev)
            y_test = payload.get("y_test", [])
            y_prob = payload.get("y_prob", [])
            if len(y_test) == len(y_prob) and len(y_test) > 0:
                for t in [i/100 for i in range(0, 101)]:
                    y_pred = [1 if float(p) >= t else 0 for p in y_prob]
                    fp = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 0 and yp == 1)
                    fn = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 1 and yp == 0)
                    loss = fp * cost_fp + fn * cost_fn
                    points.append({"threshold": t, "loss": loss})
                return jsonify({"curve": points})
    if "isFraud" in data_frame.columns:
        df = data_frame.copy()
        y = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int).tolist()
        X_raw = df.drop(columns=["isFraud"])
        feature_cols = _load_feature_columns()
        X_proc = preprocess(X_raw, expected_cols=feature_cols if feature_cols else None)
        probs = model.predict_proba(X_proc)[:, 1].tolist()
        for t in [i/100 for i in range(0, 101)]:
            y_pred = [1 if float(p) >= t else 0 for p in probs]
            fp = sum(1 for yt, yp in zip(y, y_pred) if yt == 0 and yp == 1)
            fn = sum(1 for yt, yp in zip(y, y_pred) if yt == 1 and yp == 0)
            loss = fp * cost_fp + fn * cost_fn
            points.append({"threshold": t, "loss": loss})
        return jsonify({"curve": points})
    return jsonify({"curve": []})

@app.route("/api/rate_sim")
def api_rate_sim():
    threshold = float(request.args.get("threshold", 0.5))
    if os.path.exists(TEST_EVAL_PATH):
        with open(TEST_EVAL_PATH, "r", encoding="utf-8") as fev:
            payload = json.load(fev)
            y_test = payload.get("y_test", [])
            y_prob = payload.get("y_prob", [])
            if len(y_test) == len(y_prob) and len(y_test) > 0:
                y_pred = [1 if float(p) >= threshold else 0 for p in y_prob]
                tp = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 1 and yp == 1)
                fp = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 0 and yp == 1)
                fn = sum(1 for yt, yp in zip(y_test, y_pred) if yt == 1 and yp == 0)
                total = len(y_test)
                alerts = tp + fp
                precision = tp / alerts if alerts > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                alert_rate = alerts / total if total > 0 else 0.0
                return jsonify({"precision": precision, "recall": recall, "alert_rate": alert_rate, "alerts": alerts, "total": total})
    if "isFraud" in data_frame.columns:
        df = data_frame.copy()
        y = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int).tolist()
        X_raw = df.drop(columns=["isFraud"])
        feature_cols = _load_feature_columns()
        X_proc = preprocess(X_raw, expected_cols=feature_cols if feature_cols else None)
        probs = model.predict_proba(X_proc)[:, 1].tolist()
        y_pred = [1 if float(p) >= threshold else 0 for p in probs]
        tp = sum(1 for yt, yp in zip(y, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y, y_pred) if yt == 1 and yp == 0)
        total = len(y)
        alerts = tp + fp
        precision = tp / alerts if alerts > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        alert_rate = alerts / total if total > 0 else 0.0
        return jsonify({"precision": precision, "recall": recall, "alert_rate": alert_rate, "alerts": alerts, "total": total})
    return jsonify({"precision": 0.0, "recall": 0.0, "alert_rate": 0.0, "alerts": 0, "total": 0})

@app.route("/api/model_compare")
def api_model_compare():
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        # keep workload modest to avoid UI fetch failures
        df = load_data(limit_rows=100000)
        if "isFraud" not in df.columns:
            return jsonify({})
        y = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)
        X_raw = df.drop(columns=["isFraud"])
        # sample at most 50k rows to train quickly and avoid memory pressure
        if len(X_raw) > 50000:
            sample_idx = np.random.RandomState(42).choice(len(X_raw), size=50000, replace=False)
            X_raw = X_raw.iloc[sample_idx]
            y = y.iloc[sample_idx]
        X_proc = preprocess(X_raw, expected_cols=None)
        mask = X_proc.notna().all(axis=1)
        X_proc = X_proc[mask]
        y = y[mask]
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42, stratify=y)
        def eval_model(clf):
            try:
                pipe = Pipeline(steps=[("impute", SimpleImputer(strategy="median")), ("clf", clf)])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                try:
                    y_prob = pipe.predict_proba(X_test)[:, 1]
                except Exception:
                    y_prob = None
                return {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None and len(np.unique(y_test)) > 1 else 0.0,
                }
            except Exception as e:
                return {"error": str(e), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0}
        res = {
            "HGB": json.load(open(METRICS_PATH, "r", encoding="utf-8")) if os.path.exists(METRICS_PATH) else {},
            "LogisticRegression": eval_model(LogisticRegression(max_iter=1000, class_weight="balanced")),
            "RandomForest": eval_model(RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight="balanced", n_jobs=-1))
        }
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)})

def _load_feature_columns() -> list[str]:
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
            return list((json.load(f) or {}).get("columns", []))
    return []

from src.preprocessing import preprocess
from src.predict import predict_transaction

@app.route("/api/explain", methods=["POST"])
def api_explain():
    payload = request.get_json(force=True)
    threshold = float(payload.get("threshold", 0.5))
    input_df = pd.DataFrame([{
        "type": payload.get("type", "TRANSFER"),
        "amount": float(payload.get("amount", 0.0)),
        "oldbalanceOrg": float(payload.get("oldbalanceOrg", 0.0)),
        "newbalanceOrig": float(payload.get("newbalanceOrig", 0.0)),
        "oldbalanceDest": float(payload.get("oldbalanceDest", 0.0)),
        "newbalanceDest": float(payload.get("newbalanceDest", 0.0)),
    }])
    try:
        def _finite(x: float, default: float = 0.0) -> float:
            try:
                return float(x) if np.isfinite(float(x)) else float(default)
            except Exception:
                return float(default)
        def _clean_contribs(items: list[dict]) -> list[dict]:
            out = []
            for it in items:
                eff = _finite(it.get("effect", 0.0), 0.0)
                out.append({
                    "feature": str(it.get("feature", "")),
                    "effect": eff,
                    "abs_effect": abs(eff),
                })
            out.sort(key=lambda z: z["abs_effect"], reverse=True)
            return out
        feature_cols = _load_feature_columns()
        X_live = preprocess(input_df, expected_cols=feature_cols if feature_cols else None)
        base_prob = float(predict_transaction(payload, threshold)["probability"])
        df_raw = data_frame.copy()
        if "isFraud" in df_raw.columns:
            df_raw = df_raw.drop(columns=["isFraud"])
        X_bg = preprocess(df_raw.head(500), expected_cols=feature_cols if feature_cols else None)
        try:
            import shap  # type: ignore
            f = lambda X: model.predict_proba(X)[:, 1]
            explainer = shap.Explainer(f, X_bg)
            explanation = explainer(X_live)
            vals = explanation.values[0].tolist()
            raw = [{"feature": col, "effect": float(val)} for col, val in zip(X_live.columns.tolist(), vals)]
            top = _clean_contribs(raw)[:10]
            base_value = _finite(np.ravel(explanation.base_values)[0], base_prob)
            return jsonify({
                "final_probability": base_prob,
                "base_value": base_value,
                "threshold": threshold,
                "contributions": top,
                "method": "shap"
            })
        except Exception:
            baselines = X_bg.median(numeric_only=True)
            contributions = []
            # approximate expected value as average probability on background
            try:
                bg_probs = model.predict_proba(X_bg)[:, 1]
                base_value = _finite(np.mean(bg_probs), base_prob)
            except Exception:
                base_value = float(base_prob)
            for col in X_live.columns:
                x_alt = X_live.copy()
                repl = baselines.get(col, 0.0)
                try:
                    repl = float(repl)
                except Exception:
                    repl = 0.0
                if not np.isfinite(repl):
                    repl = 0.0
                x_alt.iloc[0, x_alt.columns.get_loc(col)] = repl
                alt_prob = float(model.predict_proba(x_alt)[0, 1])
                effect = _finite(base_prob - alt_prob, 0.0)
                contributions.append({"feature": col, "effect": effect})
            top = _clean_contribs(contributions)[:10]
            return jsonify({
                "final_probability": base_prob,
                "base_value": base_value,
                "threshold": threshold,
                "contributions": top,
                "method": "fallback"
            })
    except Exception as e:
        # ensure valid JSON numbers only
        return jsonify({"error": str(e), "contributions": [], "final_probability": 0.0, "base_value": 0.0, "threshold": threshold, "method": "error"}), 200
@app.route("/api/prob_histogram")
def api_prob_histogram():
    bins = [i/20 for i in range(0, 21)]
    neg = [0 for _ in range(20)]
    pos = [0 for _ in range(20)]
    def bin_index(p: float) -> int:
        if p <= 0: return 0
        if p >= 1: return 19
        return min(19, max(0, int(p * 20)))
    if os.path.exists(TEST_EVAL_PATH):
        try:
            with open(TEST_EVAL_PATH, "r", encoding="utf-8") as fev:
                payload = json.load(fev)
                y_test = payload.get("y_test", [])
                y_prob = payload.get("y_prob", [])
                if len(y_test) == len(y_prob) and len(y_test) > 0:
                    for yt, p in zip(y_test, y_prob):
                        idx = bin_index(float(p))
                        if int(yt) == 1:
                            pos[idx] += 1
                        else:
                            neg[idx] += 1
                    return jsonify({"bins": bins, "neg": neg, "pos": pos})
        except Exception:
            pass
    if "isFraud" in data_frame.columns:
        try:
            df = data_frame.copy()
            y = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int).tolist()
            X_raw = df.drop(columns=["isFraud"])
            feature_cols = _load_feature_columns()
            X_proc = preprocess(X_raw, expected_cols=feature_cols if feature_cols else None)
            probs = model.predict_proba(X_proc)[:, 1].tolist()
            for yt, p in zip(y, probs):
                idx = bin_index(float(p))
                if int(yt) == 1:
                    pos[idx] += 1
                else:
                    neg[idx] += 1
            return jsonify({"bins": bins, "neg": neg, "pos": pos})
        except Exception:
            pass
    return jsonify({"bins": bins, "neg": neg, "pos": pos})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True, use_reloader=False)
