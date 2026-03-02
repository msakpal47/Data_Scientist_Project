from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd


def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    best_path = os.path.join(base_dir, "models", "best_model.pkl")
    model_path = os.path.join(base_dir, "models", "fuel_model.pkl")
    cols_path = os.path.join(base_dir, "models", "feature_columns.pkl")
    model = None
    feature_cols = None
    chosen_path = best_path if os.path.exists(best_path) else model_path
    if os.path.exists(chosen_path):
        try:
            with open(chosen_path, "rb") as f:
                model = pickle.load(f)
        except Exception:
            model = None
    if os.path.exists(cols_path):
        try:
            with open(cols_path, "rb") as f:
                feature_cols = pickle.load(f)
        except Exception:
            feature_cols = None
    run_log_path = os.path.join(base_dir, "models", "run_log.json")

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/api-docs", methods=["GET"])
    def api_docs():
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openapi.json")
        if os.path.exists(p):
            try:
                import json as _json
                with open(p, "r") as f:
                    return jsonify(_json.load(f))
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        return jsonify({"error": "openapi_not_found"}), 404

    @app.route("/predict", methods=["POST"])
    def predict():
        if model is None:
            return jsonify({"error": "Model not available"}), 400
        data = request.get_json(silent=True) or {}
        df = pd.DataFrame([data])
        if feature_cols and isinstance(feature_cols, dict):
            expected = feature_cols.get("numeric", []) + feature_cols.get("categorical", [])
            for c in expected:
                if c not in df.columns:
                    df[c] = 0
            df = df[expected]
        try:
            y_pred = model.predict(df)
            val = float(np.asarray(y_pred).ravel()[0])
            lower = val * 0.95
            upper = val * 1.05
            return jsonify({"prediction": val, "lower": lower, "upper": upper})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/explain", methods=["POST"])
    def explain():
        if model is None:
            return jsonify({"error": "Model not available"}), 400
        body = request.get_json(silent=True) or {}
        df = pd.DataFrame([body.get("payload") or body])
        if feature_cols and isinstance(feature_cols, dict):
            expected = feature_cols.get("numeric", []) + feature_cols.get("categorical", [])
            for c in expected:
                if c not in df.columns:
                    df[c] = 0
            df = df[expected]
        try:
            pre = model.named_steps.get("pre")
            est = model.named_steps.get("model")
        except Exception:
            return jsonify({"error": "invalid_model"}), 400
        try:
            import shap  # optional
            X_tr = pre.transform(df)
            expl = shap.TreeExplainer(est)
            sv = expl.shap_values(X_tr)
            if isinstance(sv, list):
                sv = sv[0]
            shap_row = sv[0].tolist()
            base = float(np.ravel(expl.expected_value)[0]) if np.ndim(expl.expected_value) else float(expl.expected_value)
            num_cols = list(pre.transformers_[0][2]) if len(pre.transformers_) >= 1 else []
            cat_cols = list(pre.transformers_[1][2]) if len(pre.transformers_) >= 2 else []
            ohe = None
            if hasattr(pre, "named_transformers_") and "cat" in pre.named_transformers_:
                cat_step = pre.named_transformers_["cat"]
                if hasattr(cat_step, "named_steps") and "ohe" in cat_step.named_steps:
                    ohe = cat_step.named_steps["ohe"]
            idx = 0
            aggr = {}
            for c in num_cols:
                if idx < len(shap_row):
                    aggr[c] = float(shap_row[idx])
                idx += 1
            if ohe is not None and hasattr(ohe, "categories_"):
                for c_name, cats in zip(cat_cols, ohe.categories_):
                    span = len(cats)
                    val = sum(shap_row[idx: idx + span]) if idx + span <= len(shap_row) else 0.0
                    aggr[c_name] = float(val)
                    idx += span
            items = [{"feature": k, "value": v, "abs": abs(v)} for k, v in aggr.items()]
            items.sort(key=lambda x: x["abs"], reverse=True)
            return jsonify({"baseline": base, "contributions": items[:10]})
        except Exception:
            try:
                b = float(np.asarray(model.predict(df)).ravel()[0])
                num_cols = feature_cols.get("numeric", []) if isinstance(feature_cols, dict) else list(df.columns)
                contribs = {}
                for c in num_cols:
                    try:
                        df2 = df.copy()
                        val = df2.iloc[0][c]
                        delta = val * 0.05 if val != 0 else 0.05
                        df2[c] = val + delta
                        cval = float(np.asarray(model.predict(df2)).ravel()[0])
                        contribs[c] = cval - b
                    except Exception:
                        continue
                items = [{"feature": k, "value": v, "abs": abs(v)} for k, v in contribs.items()]
                items.sort(key=lambda x: x["abs"], reverse=True)
                return jsonify({"baseline": b, "contributions": items[:10], "approx": True})
            except Exception as e:
                return jsonify({"error": str(e)}), 400

    @app.route("/model-metrics", methods=["GET"])
    def model_metrics():
        if os.path.exists(run_log_path):
            try:
                with open(run_log_path, "r") as f:
                    import json as _json
                    j = _json.load(f)
                return jsonify(j)
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        return jsonify({"error": "metrics_not_found"}), 404

    @app.route("/feature-importance", methods=["GET"])
    def feature_importance():
        if model is None:
            return jsonify({"error": "Model not available"}), 400
        try:
            if hasattr(model, "named_steps"):
                pre = model.named_steps.get("pre")
                est = model.named_steps.get("model")
            else:
                return jsonify({"error": "invalid_model"}), 400
            imp = getattr(est, "feature_importances_", None)
            if imp is None:
                return jsonify({"importances": []})
            imp = np.asarray(imp)
            num_cols = []
            cat_cols = []
            if hasattr(pre, "transformers_") and len(pre.transformers_) >= 2:
                num_cols = list(pre.transformers_[0][2])
                cat_cols = list(pre.transformers_[1][2])
            ohe = None
            if hasattr(pre, "named_transformers_") and "cat" in pre.named_transformers_:
                cat_step = pre.named_transformers_["cat"]
                if hasattr(cat_step, "named_steps") and "ohe" in cat_step.named_steps:
                    ohe = cat_step.named_steps["ohe"]
            out = {}
            idx = 0
            for c in num_cols:
                if idx < len(imp):
                    out[c] = float(imp[idx])
                idx += 1
            if ohe is not None and hasattr(ohe, "categories_"):
                for c_name, cats in zip(cat_cols, ohe.categories_):
                    span = len(cats)
                    val = float(imp[idx: idx + span].sum()) if idx + span <= len(imp) else 0.0
                    out[c_name] = val
                    idx += span
            items = sorted([{"feature": k, "importance": v} for k, v in out.items()], key=lambda x: x["importance"], reverse=True)
            return jsonify({"importances": items})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/simulate", methods=["POST"])
    def simulate():
        if model is None:
            return jsonify({"error": "Model not available"}), 400
        body = request.get_json(silent=True) or {}
        payload = body.get("payload") or {}
        pct = float(body.get("ep_delta_pct") or 10.0)
        feature_name = body.get("feature_name") or "ep (KW)"
        df_base = pd.DataFrame([payload])
        df_new = df_base.copy()
        if feature_name in df_new.columns:
            try:
                df_new[feature_name] = df_new[feature_name].astype(float) * (1.0 + pct / 100.0)
            except Exception:
                pass
        # ensure schema
        def ensure(df):
            if feature_cols and isinstance(feature_cols, dict):
                expected = feature_cols.get("numeric", []) + feature_cols.get("categorical", [])
                for c in expected:
                    if c not in df.columns:
                        df[c] = 0
                df = df[expected]
            return df
        try:
            b = float(np.asarray(model.predict(ensure(df_base))).ravel()[0])
            c = float(np.asarray(model.predict(ensure(df_new))).ravel()[0])
            dp = ((c - b) / b * 100.0) if b != 0 else 0.0
            return jsonify({"baseline": b, "changed": c, "delta_pct": dp, "feature_name": feature_name, "pct": pct})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
