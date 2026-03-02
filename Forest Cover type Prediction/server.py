import json
import os

import joblib
import numpy as np
import pandas as pd
import __main__
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import sqlite3
from sklearn.base import BaseEstimator, TransformerMixin


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    s = str(value).strip()
    if s == "":
        return None
    return float(s)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = 0
    out = out[columns]
    return out


def _make_placeholder_model(class_labels: list[int]):
    class _Placeholder:
        def __init__(self, labels):
            self._labels = labels or [1]

        def predict(self, X):
            return np.array([self._labels[0]] * len(X))

    return _Placeholder(class_labels)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if "Horizontal_Distance_To_Hydrology" in X.columns and "Vertical_Distance_To_Hydrology" in X.columns:
            X["Hydrology_Distance"] = (
                (X["Horizontal_Distance_To_Hydrology"] ** 2 + X["Vertical_Distance_To_Hydrology"] ** 2) ** 0.5
            )
        if "Horizontal_Distance_To_Roadways" in X.columns and "Horizontal_Distance_To_Fire_Points" in X.columns:
            X["Road_Fire_Distance_Ratio"] = X["Horizontal_Distance_To_Roadways"] / (
                X["Horizontal_Distance_To_Fire_Points"] + 1e-3
            )
        if "Elevation" in X.columns and "Slope" in X.columns:
            X["Elevation_Slope_Interaction"] = X["Elevation"] * X["Slope"]
        if {"Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"}.issubset(set(X.columns)):
            X["Mean_Hillshade"] = (X["Hillshade_9am"] + X["Hillshade_Noon"] + X["Hillshade_3pm"]) / 3.0
        if "Aspect" in X.columns:
            rad = np.deg2rad(X["Aspect"] % 360)
            X["Aspect_sin"] = np.sin(rad)
            X["Aspect_cos"] = np.cos(rad)
            X = X.drop(columns=["Aspect"])
        return X


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    models_dir = os.path.join(app.root_path, "models")
    metadata_path = os.path.join(models_dir, "metadata.json")
    model_path = os.path.join(models_dir, "model.pkl")

    metadata = None
    model = None

    def _read_lookup_table(db_path, table_candidates, id_candidates=("id", "code", "idx"), name_candidates=("name", "label", "title", "type", "description")):
        if not os.path.exists(db_path):
            return None
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            chosen = None
            for t in table_candidates:
                if t in tables:
                    chosen = t
                    break
            if chosen is None:
                needle = None
                for tc in table_candidates:
                    tl = str(tc).lower()
                    if "wilderness" in tl:
                        needle = "wilderness"
                        break
                    if "soil" in tl:
                        needle = "soil"
                        break
                if needle is None:
                    needle = "wilderness"
                for t in tables:
                    tl = str(t).lower()
                    if needle in tl:
                        chosen = t
                        break
            if chosen is None:
                con.close()
                return None
            cols = [r[1] for r in cur.execute(f"PRAGMA table_info({chosen})")]
            id_col = next((c for c in id_candidates if c in cols), None)
            name_col = next((c for c in name_candidates if c in cols), None)
            if id_col is None or name_col is None:
                if len(cols) >= 2:
                    id_col, name_col = cols[0], cols[1]
                else:
                    con.close()
                    return None
            rows = list(cur.execute(f"SELECT {id_col}, {name_col} FROM {chosen} ORDER BY {id_col} ASC"))
            con.close()
            return [{"id": int(r[0]), "name": str(r[1])} for r in rows]
        except Exception:
            return None

    def get_metadata():
        nonlocal metadata
        if metadata is None:
            if os.path.exists(metadata_path):
                metadata = _load_json(metadata_path)
            else:
                metadata = {"base_columns": [], "class_labels": [1]}
        return metadata

    def get_model():
        nonlocal model
        if model is None:
            meta = get_metadata()
            if os.path.exists(model_path):
                setattr(__main__, "FeatureEngineer", FeatureEngineer)
                model = joblib.load(model_path)
            else:
                model = _make_placeholder_model(meta.get("class_labels") or [1])
        return model

    @app.get("/")
    def home():
        meta = get_metadata()
        return render_template(
            "index.html",
            base_columns=meta.get("base_columns", []),
            model_available=os.path.exists(model_path),
        )

    @app.get("/api/metadata")
    def api_metadata():
        meta = get_metadata()
        return jsonify(
            {
                **meta,
                "model_available": os.path.exists(model_path),
                "metadata_available": os.path.exists(metadata_path),
            }
        )

    @app.get("/api/options")
    def api_options():
        meta = get_metadata()
        cols = meta.get("base_columns", []) or []
        wilderness_cols = [c for c in cols if c.startswith("Wilderness_Area_")]
        soil_cols = [c for c in cols if c.startswith("Soil_Type_")]

        db_path = os.path.join(app.root_path, "classification.db")
        wild_rows = _read_lookup_table(
            db_path,
            table_candidates=("wilderness_areas", "wilderness", "WildernessAreas", "Wilderness"),
        )
        soil_rows = _read_lookup_table(
            db_path,
            table_candidates=("soil_types", "soils", "SoilTypes", "Soils"),
        )

        if not wild_rows:
            wild_rows = [{"id": i, "name": f"Wilderness Area {i}"} for i in range(len(wilderness_cols))]
        if not soil_rows:
            soil_rows = [{"id": i, "name": f"Soil Type {i}"} for i in range(len(soil_cols))]

        wilderness = [{"key": f"Wilderness_Area_{w['id']}", "name": w["name"]} for w in wild_rows]
        soils = [{"key": f"Soil_Type_{s['id']}", "name": s["name"]} for s in soil_rows]

        return jsonify({"wilderness": wilderness, "soils": soils})

    @app.get("/api/column_values")
    def api_column_values():
        columns_param = request.args.get("columns", "")
        columns = [c for c in [s.strip() for s in columns_param.split(",")] if c]
        out = {}
        if not columns:
            return jsonify(out)
        db_path = os.path.join(app.root_path, "classification.db")
        table = "forest_cov_type"
        if not os.path.exists(db_path):
            return jsonify(out)
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            for col in columns:
                try:
                    rows = list(cur.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY {col} ASC LIMIT 300"))
                    vals = []
                    for r in rows:
                        v = r[0]
                        try:
                            if isinstance(v, (int, float, np.number)) or (isinstance(v, str) and v.strip() != ""):
                                vals.append(float(v))
                        except Exception:
                            pass
                    if not vals:
                        rng = list(cur.execute(f"SELECT MIN({col}), MAX({col}) FROM {table} WHERE {col} IS NOT NULL"))
                        if rng and rng[0][0] is not None and rng[0][1] is not None:
                            mn, mx = float(rng[0][0]), float(rng[0][1])
                            steps = 30
                            if mx == mn:
                                vals = [mn]
                            else:
                                vals = [round(mn + (mx - mn) * i / steps, 2) for i in range(steps + 1)]
                    out[col] = vals
                except Exception:
                    out[col] = []
            con.close()
        except Exception:
            pass
        return jsonify(out)

    @app.get("/api/insights")
    def api_insights():
        db_path = os.path.join(app.root_path, "insights.db")
        results = []
        if not os.path.exists(db_path):
            return jsonify({"items": results})
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute(
                "CREATE TABLE IF NOT EXISTS model_metrics(accuracy REAL, macro_f1 REAL, model_name TEXT, sample_n INTEGER, trained_at TEXT)"
            )
            rows = list(
                cur.execute(
                    "SELECT accuracy, macro_f1, model_name, sample_n, trained_at FROM model_metrics ORDER BY trained_at DESC LIMIT 20"
                )
            )
            con.close()
            for r in rows:
                try:
                    results.append(
                        {
                            "accuracy": float(r[0]) if r[0] is not None else None,
                            "macro_f1": float(r[1]) if r[1] is not None else None,
                            "model_name": str(r[2]) if r[2] is not None else "",
                            "sample_n": int(r[3]) if r[3] is not None else None,
                            "trained_at": str(r[4]) if r[4] is not None else "",
                        }
                    )
                except Exception:
                    continue
        except Exception:
            results = []
        return jsonify({"items": results})

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Invalid JSON payload"}), 400

        meta = get_metadata()
        cols = meta.get("base_columns", [])
        if not isinstance(cols, list) or not cols:
            return jsonify({"error": "metadata.json is missing base_columns"}), 500

        if isinstance(payload, dict) and "rows" in payload:
            rows = payload["rows"]
        else:
            rows = payload

        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or not rows:
            return jsonify({"error": "Provide a feature object or a list of feature objects"}), 400

        normalized_rows = []
        for r in rows:
            if not isinstance(r, dict):
                return jsonify({"error": "Each row must be a JSON object"}), 400
            normalized = {}
            for k, v in r.items():
                try:
                    normalized[k] = _safe_float(v)
                except Exception:
                    normalized[k] = v
            normalized_rows.append(normalized)

        try:
            df = pd.DataFrame(normalized_rows)
            df = _ensure_columns(df, cols)

            clf = get_model()
            pred = clf.predict(df)
            pred_list = [int(p) if isinstance(p, (int, np.integer)) else p for p in pred]
            probabilities = None
            class_labels = get_metadata().get("class_labels", [])
            try:
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(df)
                    if isinstance(proba, (list, np.ndarray)):
                        first = np.array(proba)[0]
                        probabilities = [float(x) for x in first.tolist()]
            except Exception:
                probabilities = None
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify(
            {
                "predictions": pred_list,
                "model_available": os.path.exists(model_path),
                "probabilities": probabilities,
                "class_labels": class_labels,
            }
        )

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
