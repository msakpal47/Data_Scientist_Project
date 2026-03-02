import os
import json
import pickle
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import sqlite3
import sys
import numpy as np
import io
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "app", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")


def load_artifacts() -> Dict[str, Any]:
    paths = {
        "model": os.path.join(MODELS_DIR, "model.pkl"),
        "pre": os.path.join(MODELS_DIR, "scaler.pkl"),
        "cols": os.path.join(MODELS_DIR, "columns.pkl"),
        "importance": os.path.join(MODELS_DIR, "feature_importance.json"),
        "leaderboard": os.path.join(MODELS_DIR, "leaderboard.json"),
    }
    exists = all(os.path.exists(p) for p in [paths["model"], paths["cols"]])
    artifacts = {"available": exists, "paths": paths}
    if exists:
        try:
            with open(paths["model"], "rb") as f:
                artifacts["model"] = pickle.load(f)
            with open(paths["cols"], "rb") as f:
                artifacts["columns"] = pickle.load(f)
            if os.path.exists(paths["importance"]):
                with open(paths["importance"], "r", encoding="utf-8") as f:
                    artifacts["importance"] = json.load(f)
            if os.path.exists(paths["leaderboard"]):
                with open(paths["leaderboard"], "r", encoding="utf-8") as f:
                    artifacts["leaderboard"] = json.load(f)
        except Exception:
            artifacts["available"] = False
    return artifacts


app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
art = load_artifacts()

@app.errorhandler(500)
def _e500(e):
    try:
        import traceback
        print("ERROR_500", traceback.format_exc())
    except Exception:
        pass
    return "Internal Server Error", 500


def as_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    cols = art.get("columns")
    if not cols:
        return pd.DataFrame([payload])
    num = cols.get("numeric", [])
    cat = cols.get("categorical", [])
    ordered = {}
    for c in num:
        v = payload.get(c)
        try:
            ordered[c] = float(v) if v not in (None, "", "None") else 0.0
        except Exception:
            ordered[c] = 0.0
    for c in cat:
        v = payload.get(c)
        ordered[c] = str(v) if v not in (None, "", "None") else "Unknown"
    # Include any extra keys not part of training columns
    for k, v in payload.items():
        if k not in ordered:
            ordered[k] = v
    return pd.DataFrame([ordered])


def validate_inputs(payload: Dict[str, Any]) -> Dict[str, Any]:
    msg = None
    try:
        age = float(payload.get("age", 0) or 0)
        bmi = float(payload.get("bmi", 0) or 0)
        children = int(payload.get("children", 0) or 0)
        if age <= 0:
            msg = "Age must be greater than 0"
        elif bmi <= 0:
            msg = "BMI must be greater than 0"
        elif children < 0:
            msg = "Children cannot be negative"
    except Exception:
        msg = "Invalid numeric input"
    return {"ok": msg is None, "error": msg}

def risk_and_flags(payload: Dict[str, Any], pred: float) -> Dict[str, Any]:
    base = min(max(pred, 0.0) / 20000.0, 1.0)
    bmi = float(payload.get("bmi", 0) or 0)
    smoker = str(payload.get("smoker", "no")).lower() in {"yes", "true", "1"}
    age = float(payload.get("age", 0) or 0)
    exercise = str(payload.get("exercise_frequency", "low")).lower()
    risk = base
    risk += min(bmi / 50.0, 1.0) * 0.1
    risk += (0.1 if smoker else 0.0)
    risk += min(age / 100.0, 1.0) * 0.1
    if exercise in {"high", "daily"}:
        risk -= 0.05
    risk = max(0.0, min(risk, 1.0))
    fraud = False
    if smoker and exercise in {"none", "low"} and bmi > 40:
        fraud = True
    return {"risk_score": risk, "fraud_flag": fraud}


def get_tables() -> list:
    conn = sqlite3.connect(os.path.join(BASE_DIR, "regression.db"))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    names = [r[0] for r in cur.fetchall()]
    conn.close()
    return names


def load_data_from_db(table: str) -> pd.DataFrame:
    conn = sqlite3.connect(os.path.join(BASE_DIR, "regression.db"))
    df = pd.read_sql_query(f"SELECT * FROM [{table}]", conn)
    conn.close()
    return df


def get_table_schema(table: str) -> list:
    conn = sqlite3.connect(os.path.join(BASE_DIR, "regression.db"))
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    cols = [r[1] for r in cur.fetchall()]
    conn.close()
    return cols

def _union_medical_options(column: str) -> list:
    base = [
        "None", "Diabetes", "Hypertension", "Heart disease",
        "High blood pressure", "Asthma", "Cancer", "COPD",
        "Chronic kidney disease", "Thyroid disorder",
        "Arthritis", "Depression", "Anxiety", "Stroke", "Obesity",
    ]
    vals = []
    try:
        tables = []
        try:
            tables = get_tables()
        except Exception:
            pass
        table = "Insurance_Prediction" if "Insurance_Prediction" in tables else (tables[0] if tables else None)
        if table:
            conn = sqlite3.connect(os.path.join(BASE_DIR, "regression.db"))
            cur = conn.cursor()
            q = f"SELECT [{column}] FROM [{table}] WHERE [{column}] IS NOT NULL AND TRIM([{column}])<>'' LIMIT 1000"
            cur.execute(q)
            rows = cur.fetchall()
            conn.close()
            for r in rows:
                try:
                    s = str(r[0]).strip()
                    if s:
                        vals.append(s)
                except Exception:
                    continue
    except Exception:
        pass
    seen = set()
    out = []
    for v in base + vals:
        k = str(v).strip()
        low = k.lower()
        if not k or low in seen:
            continue
        seen.add(low)
        out.append(k)
    return out

@app.route("/", methods=["GET", "POST"])
def index():
    if not art.get("available"):
        return redirect(url_for("setup"))
    # refresh leaderboard each request to ensure UI shows latest
    latest = load_artifacts()
    leaderboard = latest.get("leaderboard", art.get("leaderboard", [])) or []
    # columns/importance for server-side fallback rendering
    columns = art.get("columns") or latest.get("columns") or {}
    importance = art.get("importance") or latest.get("importance") or {}
    shap_img = ""
    medical_options = _union_medical_options("medical_history")
    family_medical_options = _union_medical_options("family_medical_history")
    # server-side KPI defaults
    try:
        def _r2v(it):
            v = it.get("r2")
            if isinstance(v, (int, float)):
                return float(v)
            v = it.get("r2_test")
            return float(v) if isinstance(v, (int, float)) else 0.0
        best = max(leaderboard, key=_r2v) if leaderboard else None
        best_model = (best or {}).get("name") or ""
        best_r2 = _r2v(best) if best else None
        best_mae = (best or {}).get("mae")
    except Exception:
        best_model, best_r2, best_mae = "", None, None
    # server-side importance table pairs
    try:
        if isinstance(importance, dict):
            importance_pairs = sorted(
                [(k, float(importance[k])) for k in importance.keys()],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:20]
        else:
            importance_pairs = []
    except Exception:
        importance_pairs = []
    # seed a baseline prediction for initial page load
    seed_prediction = None
    try:
        if art.get("available"):
            seed_payload = {
                "age": 30, "bmi": 25, "children": 0, "smoker": "no",
                "medical_history": "None", "family_medical_history": "None",
                "exercise_frequency": "Frequently", "coverage_level": "Basic",
                "region": "southeast", "occupation": "Blue collar", "gender": "female",
            }
            df0 = as_dataframe(seed_payload)
            sp = float(art["model"].predict(df0)[0])
            seed_prediction = max(0.0, sp)
    except Exception:
        seed_prediction = None
    if request.method == "POST":
        form_data = request.form.to_dict()
        v = validate_inputs(form_data)
        if not v["ok"]:
            return render_template("index.html", leaderboard=leaderboard, columns=columns, importance=importance, importance_pairs=importance_pairs, shap_img=shap_img, error_message=v["error"], medical_options=medical_options, family_medical_options=family_medical_options, best_model=best_model, best_r2=best_r2, best_mae=best_mae)
        df = as_dataframe(form_data)
        pred = float(art["model"].predict(df)[0])
        pred = max(0.0, pred)
        return render_template("index.html", leaderboard=leaderboard, columns=columns, importance=importance, importance_pairs=importance_pairs, shap_img=shap_img, prediction=pred, medical_options=medical_options, family_medical_options=family_medical_options, best_model=best_model, best_r2=best_r2, best_mae=best_mae)
    return render_template("index.html", leaderboard=leaderboard, columns=columns, importance=importance, importance_pairs=importance_pairs, shap_img=shap_img, prediction=seed_prediction, medical_options=medical_options, family_medical_options=family_medical_options, best_model=best_model, best_r2=best_r2, best_mae=best_mae)


@app.route("/predict", methods=["POST"])
def predict():
    if not art.get("available"):
        return jsonify({"error": "model_unavailable"}), 400
    data = request.get_json() or request.form.to_dict()
    v = validate_inputs(data)
    if not v["ok"]:
        return jsonify({"error": "invalid_input", "detail": v["error"]}), 400
    df = as_dataframe(data)
    pred = float(art["model"].predict(df)[0])
    pred = max(0.0, pred)
    return jsonify({"prediction": pred})

@app.route("/predict-report", methods=["POST"])
def predict_report():
    if not art.get("available"):
        return jsonify({"error": "model_unavailable"}), 400
    data = request.get_json(silent=True) or request.form.to_dict()
    v = validate_inputs(data)
    if not v["ok"]:
        return jsonify({"error": "invalid_input", "detail": v["error"]}), 400
    df = as_dataframe(data)
    pred = float(art["model"].predict(df)[0])
    pred = max(0.0, pred)
    out = dict(data)
    out.update({"prediction": pred})
    try:
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        rep = pd.DataFrame([out]).to_csv(index=False)
        buf = io.BytesIO(rep.encode("utf-8"))
        buf.seek(0)
        return send_file(buf, mimetype="text/csv", as_attachment=True, download_name=f"prediction_report_{ts}.csv")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        imp_path = os.path.join(MODELS_DIR, "feature_importance.json")
        if os.path.exists(imp_path):
            with open(imp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    art["importance"] = data
                    return jsonify(data)
    except Exception:
        pass
    # Fallback: compute from current model in memory
    try:
        model = art.get("model")
        if model:
            pre = model.named_steps.get("preprocessor")
            reg = model.named_steps.get("regressor")
            if pre and reg:
                # try direct attributes
                feature_names = _ohe_feature_names(pre)
                values = None
                if hasattr(reg, "feature_importances_"):
                    values = reg.feature_importances_
                elif hasattr(reg, "coef_"):
                    import numpy as _np
                    values = _np.abs(_np.ravel(reg.coef_))
                if values is not None:
                    if feature_names and len(feature_names) == len(values):
                        scores = {f: float(v) for f, v in zip(feature_names, values)}
                    else:
                        scores = {f"f{i}": float(v) for i, v in enumerate(values)}
                    if isinstance(scores, dict) and scores:
                        art["importance"] = scores
                        try:
                            with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
                                json.dump(scores, f)
                        except Exception:
                            pass
                        return jsonify(scores)
                # permutation importance as last resort
                try:
                    from sklearn.inspection import permutation_importance as _pi  # type: ignore
                    tables = []
                    try:
                        tables = get_tables()
                    except Exception:
                        pass
                    table = "Insurance_Prediction" if "Insurance_Prediction" in tables else (tables[0] if tables else None)
                    if table:
                        df = load_data_from_db(table)
                        cols = art.get("columns") or {}
                        tgt = cols.get("target", "charges")
                        y = df[tgt].values if tgt in df.columns else None
                        X = df.drop(columns=[tgt]) if tgt in df.columns else df
                        keep = (cols.get("numeric", []) or []) + (cols.get("categorical", []) or [])
                        X = X[[c for c in keep if c in X.columns]].dropna().head(800)
                        if y is not None and len(X) and len(X) == len(y[: len(X)]):
                            r = _pi(model, X, y[: len(X)], n_repeats=3, random_state=42, n_jobs=-1, scoring="r2")
                            vals = r.importances_mean
                            names = _ohe_feature_names(pre) or [f"f{i}" for i in range(len(vals))]
                            if len(names) != len(vals):
                                names = [f"f{i}" for i in range(len(vals))]
                            scores = {str(n): float(v) for n, v in zip(names, vals)}
                            art["importance"] = scores
                            try:
                                with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
                                    json.dump(scores, f)
                            except Exception:
                                pass
                            return jsonify(scores)
                except Exception:
                    pass
    except Exception:
        pass
    return jsonify(art.get("importance") or {})

@app.route("/model/columns", methods=["GET"])
def model_columns():
    # Ensure columns are available even if artifacts were refreshed elsewhere
    if not art.get("columns"):
        try:
            refreshed = load_artifacts()
            if refreshed.get("columns"):
                art["columns"] = refreshed["columns"]
        except Exception:
            pass
    cols = art.get("columns") or {}
    return jsonify(cols)

@app.route("/model/leaderboard", methods=["GET"])
def model_leaderboard():
    try:
        refreshed = load_artifacts()
        lb = refreshed.get("leaderboard") or art.get("leaderboard") or []
        if lb:
            art["leaderboard"] = lb
        return jsonify(lb)
    except Exception:
        return jsonify([])

@app.route("/predict_json", methods=["POST"])
@app.route("/predict-json", methods=["POST"])
def predict_json():
    if not art.get("available"):
        return jsonify({"error": "model_unavailable"}), 400
    data = request.get_json() or {}
    v = validate_inputs(data)
    if not v["ok"]:
        return jsonify({"error": "invalid_input", "detail": v["error"]}), 400
    df = as_dataframe(data)
    pred = float(art["model"].predict(df)[0])
    pred = max(0.0, pred)
    return jsonify({"prediction": pred})


@app.route("/tables", methods=["GET"])
def tables():
    try:
        return jsonify(get_tables())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/schema/types", methods=["GET"])
def schema_types():
    table = request.args.get("table") or ""
    if not table:
        return jsonify({"error": "missing_table"}), 400
    try:
        conn = sqlite3.connect(os.path.join(BASE_DIR, "regression.db"))
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info('{table}')")
        rows = cur.fetchall()
        conn.close()
        data = [{"name": r[1], "type": r[2]} for r in rows]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _generate_shap_image_b64(max_rows: int = 500) -> str:
    try:
        import shap  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import io, base64
    except Exception:
        return ""
    if not art.get("available"):
        return ""
    try:
        model = art["model"]
        pre = model.named_steps.get("preprocessor")
        reg = model.named_steps.get("regressor")
        tables = []
        try:
            tables = get_tables()
        except Exception:
            pass
        table = "Insurance_Prediction" if "Insurance_Prediction" in tables else (tables[0] if tables else None)
        if table:
            df = load_data_from_db(table)
            cols = art.get("columns") or {}
            tgt = cols.get("target", "charges")
            if tgt in df.columns:
                df = df.drop(columns=[tgt])
            keep = (cols.get("numeric", []) or []) + (cols.get("categorical", []) or [])
            df = df[[c for c in keep if c in df.columns]].dropna().head(max_rows)
        else:
            return ""
        Xt = pre.transform(df)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        try:
            explainer = shap.TreeExplainer(reg)
            sv = explainer.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[0]
        except Exception:
            explainer = shap.KernelExplainer(reg.predict, Xt[:50])
            sv = explainer.shap_values(Xt[:200], nsamples=50)
        plt.figure()
        shap.summary_plot(sv, Xt, show=False)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""

@app.route("/schema", methods=["GET"])
def schema():
    table = request.args.get("table") or ""
    if not table:
        return jsonify({"error": "missing_table"}), 400
    try:
        cols = get_table_schema(table)
        return jsonify(cols)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/category-values", methods=["GET"])
def category_values():
    col = (request.args.get("column") or "").strip()
    if not col:
        return jsonify({"error": "missing_column"}), 400
    try:
        tables = []
        try:
            tables = get_tables()
        except Exception:
            pass
        table = "Insurance_Prediction" if "Insurance_Prediction" in tables else (tables[0] if tables else None)
        if not table:
            return jsonify({"error": "no_table"}), 400
        conn = sqlite3.connect(os.path.join(BASE_DIR, "regression.db"))
        cur = conn.cursor()
        q = f"SELECT [{col}] as v, COUNT(1) c FROM [{table}] WHERE [{col}] IS NOT NULL AND TRIM([{col}])<>'' GROUP BY [{col}] ORDER BY c DESC, v ASC LIMIT 100"
        cur.execute(q)
        rows = cur.fetchall()
        conn.close()
        vals = []
        for v, _ in rows:
            try:
                s = str(v)
                if s not in vals:
                    vals.append(s)
            except Exception:
                continue
        return jsonify({"values": vals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/setup", methods=["GET"])
def setup():
    return render_template("setup.html")


@app.route("/train", methods=["POST"])
def train():
    j = request.get_json(silent=True) or {}
    target = (request.form.get("target") or j.get("target") or "charges").strip()
    table = (request.form.get("table") or j.get("table") or "").strip()
    algo = (request.form.get("model") or request.form.get("algo") or j.get("model") or j.get("algo") or "").strip() or None
    if not table:
        return jsonify({"error": "missing_table"}), 400
    tables = get_tables()
    if table not in tables:
        return jsonify({"error": "invalid_table"}), 400
    try:
        from training import train_model
        df = load_data_from_db(table)
        res = train_model.train_df(df, target_col=target, models_dir=MODELS_DIR, prefer_model=algo)
        global art
        art = load_artifacts()
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def _ohe_feature_names(pre: Any) -> list:
    try:
        num_feats = pre.transformers_[0][2]
        cat_t = pre.transformers_[1]
        cat_pipe = cat_t[1]
        cat_feats = cat_t[2]
        oh = getattr(cat_pipe, "named_steps", {}).get("oh") if hasattr(cat_pipe, "named_steps") else None
        if oh is None and hasattr(cat_pipe, "get_feature_names_out"):
            oh = cat_pipe
        cat_names = oh.get_feature_names_out(cat_feats).tolist() if oh else [str(c) for c in cat_feats]
        return list(num_feats) + cat_names
    except Exception:
        return []


@app.route("/shap/global", methods=["GET"])
def shap_global():
    try:
        import shap  # type: ignore
    except Exception as e:
        return jsonify({"error": "shap_not_installed", "detail": str(e)}), 501
    if not art.get("available"):
        return jsonify({"error": "model_unavailable"}), 400
    try:
        model = art["model"]
        pre = model.named_steps.get("preprocessor")
        reg = model.named_steps.get("regressor")
        feature_names = _ohe_feature_names(pre) if pre else []
        # Try to sample from the primary table if available
        tables = []
        try:
            tables = get_tables()
        except Exception:
            pass
        table = "Insurance_Prediction" if "Insurance_Prediction" in tables else (tables[0] if tables else None)
        X_sample = None
        if table:
            df = load_data_from_db(table)
            cols = art.get("columns") or {}
            target = cols.get("target", "charges")
            if target in df.columns:
                df = df.drop(columns=[target])
            # keep only known columns
            keep = (cols.get("numeric", []) or []) + (cols.get("categorical", []) or [])
            df = df[[c for c in keep if c in df.columns]].dropna().head(800)
            X_sample = df
        else:
            # fallback: build a tiny dataframe with zeros/Unknowns
            cols = art.get("columns") or {}
            obj = {}
            for c in cols.get("numeric", []):
                obj[c] = 0
            for c in cols.get("categorical", []):
                obj[c] = "Unknown"
            X_sample = pd.DataFrame([obj] * 50)
        # transform using fitted preprocessor
        Xt = pre.transform(X_sample)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        try:
            explainer = shap.TreeExplainer(reg)
            sv = explainer.shap_values(Xt)
            if isinstance(sv, list):  # some models return list
                sv = sv[0]
        except Exception:
            # Fallback to model-agnostic explainer on numeric matrix
            explainer = shap.KernelExplainer(reg.predict, Xt[:50])
            sv = explainer.shap_values(Xt[:200], nsamples=50)
        vals = np.mean(np.abs(sv), axis=0).tolist()
        if feature_names and len(feature_names) == len(vals):
            data = {f: float(v) for f, v in zip(feature_names, vals)}
        else:
            data = {str(i): float(v) for i, v in enumerate(vals)}
        return jsonify({"feature_importance": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shap/global_img", methods=["GET"])
def shap_global_img():
    try:
        import shap  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import io, base64
    except Exception as e:
        return jsonify({"error": "shap_matplotlib_not_installed", "detail": str(e)}), 501
    if not art.get("available"):
        return jsonify({"error": "model_unavailable"}), 400
    try:
        model = art["model"]
        pre = model.named_steps.get("preprocessor")
        reg = model.named_steps.get("regressor")
        tables = []
        try:
            tables = get_tables()
        except Exception:
            pass
        table = "Insurance_Prediction" if "Insurance_Prediction" in tables else (tables[0] if tables else None)
        if table:
            df = load_data_from_db(table)
            cols = art.get("columns") or {}
            target = cols.get("target", "charges")
            if target in df.columns:
                df = df.drop(columns=[target])
            keep = (cols.get("numeric", []) or []) + (cols.get("categorical", []) or [])
            df = df[[c for c in keep if c in df.columns]].dropna().head(500)
        else:
            df = pd.DataFrame([{}])
        Xt = pre.transform(df)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        try:
            explainer = shap.TreeExplainer(reg)
            sv = explainer.shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[0]
        except Exception:
            explainer = shap.KernelExplainer(reg.predict, Xt[:50])
            sv = explainer.shap_values(Xt[:200], nsamples=50)
        plt.figure()
        shap.summary_plot(sv, Xt, show=False)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return jsonify({"image": img_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/shap/explain", methods=["POST"])
def shap_explain():
    try:
        import shap  # type: ignore
    except Exception as e:
        return jsonify({"error": "shap_not_installed", "detail": str(e)}), 501
    if not art.get("available"):
        return jsonify({"error": "model_unavailable"}), 400
    try:
        payload = request.get_json() or request.form.to_dict()
        df = as_dataframe(payload)
        model = art["model"]
        pre = model.named_steps.get("preprocessor")
        reg = model.named_steps.get("regressor")
        feature_names = _ohe_feature_names(pre) if pre else []
        Xt = pre.transform(df)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        try:
            if hasattr(reg, "apply") or hasattr(reg, "estimators_"):
                explainer = shap.TreeExplainer(reg)
                sv = explainer.shap_values(Xt)
                if isinstance(sv, list):
                    sv = sv[0]
                sv = sv[0]
                base_value = float(explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0])
            elif hasattr(reg, "coef_"):
                # linear models: shap via LinearExplainer
                explainer = shap.LinearExplainer(reg, Xt)
                sv = explainer.shap_values(Xt)[0]
                base_value = float(explainer.expected_value)
            else:
                explainer = shap.KernelExplainer(reg.predict, Xt)
                sv = explainer.shap_values(Xt, nsamples=50)[0]
                base_value = float(explainer.expected_value)
        except Exception:
            return jsonify({"error": "shap_failed"}), 500
        vals = sv.tolist()
        feats = feature_names if feature_names and len(feature_names) == len(vals) else [f"f{i}" for i in range(len(vals))]
        pairs = sorted([{"name": n, "value": float(v), "abs": float(abs(v))} for n, v in zip(feats, vals)], key=lambda x: x["abs"], reverse=True)[:10]
        return jsonify({"base_value": base_value, "contributions": pairs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-form", methods=["GET", "POST"])
def predict_form():
    if request.method == "GET":
        return redirect(url_for("index"))
    if not art.get("available"):
        return redirect(url_for("setup"))
    form_data = request.form.to_dict()
    df = as_dataframe(form_data)
    pred = float(art["model"].predict(df)[0])
    pred = max(0.0, pred)
    extra = risk_and_flags(form_data, pred)
    leaderboard = art.get("leaderboard", [])
    medical_options = _union_medical_options("medical_history")
    family_medical_options = _union_medical_options("family_medical_history")
    return render_template("index.html", leaderboard=leaderboard, prediction=pred, medical_options=medical_options, family_medical_options=family_medical_options, **extra)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
