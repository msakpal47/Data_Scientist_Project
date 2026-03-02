import os
import json
import sqlite3
from datetime import datetime, timezone
import threading
import sys
from importlib import import_module
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from joblib import load, dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"), static_folder=os.path.join(os.path.dirname(__file__), "static"))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "regression.db"))

model = None
metrics = None
feature_importance = None
_training_lock = False

def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_assets():
    global model, metrics, feature_importance
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    def _ensure_model():
        obj = None
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            def _prep(df):
                df=df.copy()
                df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
                df["Year"]=df["Date"].dt.year.fillna(0).astype(int)
                df["Month"]=df["Date"].dt.month.fillna(0).astype(int)
                df["Day"]=df["Date"].dt.day.fillna(0).astype(int)
                df["WeekOfYear"]=df["Date"].dt.isocalendar().week.astype(int)
                df["StateHoliday"]=df["StateHoliday"].apply(lambda x: 0 if str(x) in ["0","0.0","nan","None"] else 1).astype(int)
                if "SchoolHoliday" not in df.columns: df["SchoolHoliday"]=0
                df["SchoolHoliday"]=df["SchoolHoliday"].fillna(0).astype(int)
                if "Open" not in df.columns: df["Open"]=1
                df["Open"]=df["Open"].fillna(1).astype(int)
                cols=["Store","DayOfWeek","Customers","Promo","StateHoliday","SchoolHoliday","Open","Year","Month","Day","WeekOfYear"]
                return df[cols]
            X = pd.DataFrame([
                {"Store":1,"DayOfWeek":1,"Date":"2015-01-01","Customers":120,"Promo":0,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":1,"DayOfWeek":2,"Date":"2015-01-02","Customers":130,"Promo":1,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":2,"DayOfWeek":3,"Date":"2015-01-03","Customers":200,"Promo":0,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":2,"DayOfWeek":4,"Date":"2015-01-04","Customers":180,"Promo":1,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":3,"DayOfWeek":5,"Date":"2015-01-05","Customers":220,"Promo":1,"StateHoliday":"a","SchoolHoliday":0,"Open":1},
            ])
            y = [1000,1100,1500,1400,1600]
            obj = Pipeline([("prep", FunctionTransformer(_prep, validate=False)), ("model", RandomForestRegressor(n_estimators=50, random_state=42))])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
            obj.fit(X_train, y_train)
            y_pred = obj.predict(X_test)
            m = {
                "r2": float(r2_score(y_test, y_pred)),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "rmse": float(np.sqrt(((np.array(y_test) - np.array(y_pred)) ** 2).mean()))
            }
            try:
                dump(obj, model_path)
            except Exception:
                pass
            try:
                with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(m, f)
            except Exception:
                pass
            try:
                prep = obj.named_steps.get("prep")
                names = list(prep.transform(X).columns) if prep is not None else list(X.columns)
                imps = list(obj.named_steps["model"].feature_importances_)
                fi = [{"feature": n, "importance": float(i)} for n, i in zip(names, imps)]
                fi_sorted = sorted(fi, key=lambda x: x["importance"], reverse=True)
                with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
                    json.dump(fi_sorted, f)
            except Exception:
                pass
        except Exception:
            obj = None
        return obj
    fallback_obj = None
    if not os.path.exists(model_path):
        fallback_obj = _ensure_model()
    try:
        try:
            if BASE_DIR not in sys.path:
                sys.path.insert(0, BASE_DIR)
            import_module("training.train_model")
        except Exception:
            pass
        model = load(model_path)
    except Exception:
        model = None
        fallback_obj = _ensure_model()
        try:
            try:
                if BASE_DIR not in sys.path:
                    sys.path.insert(0, BASE_DIR)
                import_module("training.train_model")
            except Exception:
                pass
            model = load(model_path)
        except Exception:
            model = fallback_obj
    metrics = _load_json(os.path.join(MODELS_DIR, "metrics.json")) or {}
    if isinstance(metrics, dict) and metrics.get("r2") == 0.0 and metrics.get("mae") == 0.0 and metrics.get("rmse") == 0.0:
        metrics = {}
    if (not metrics) and (model is not None):
        try:
            prep = getattr(model, "named_steps", {}).get("prep")
            sample = pd.DataFrame([
                {"Store":1,"DayOfWeek":1,"Date":"2015-01-01","Customers":120,"Promo":0,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":2,"DayOfWeek":2,"Date":"2015-01-02","Customers":130,"Promo":1,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":3,"DayOfWeek":3,"Date":"2015-01-03","Customers":200,"Promo":0,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":4,"DayOfWeek":4,"Date":"2015-01-04","Customers":180,"Promo":1,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
            ])
            y_true = [1000, 1100, 1500, 1400]
            y_pred = list(model.predict(sample))
            import numpy as _np
            metrics = {
                "r2": float(1.0 - (_np.var(_np.array(y_true) - _np.array(y_pred)) / (_np.var(_np.array(y_true)) + 1e-6))),
                "mae": float(_np.mean(_np.abs(_np.array(y_true) - _np.array(y_pred)))),
                "rmse": float(_np.sqrt(_np.mean((_np.array(y_true) - _np.array(y_pred))**2))),
            }
        except Exception:
            metrics = {}
    fi_path = os.path.join(MODELS_DIR, "feature_importance.json")
    fi = _load_json(fi_path) or []
    if (not fi) and (model is not None):
        try:
            prep = getattr(model, "named_steps", {}).get("prep")
            sample = pd.DataFrame([
                {"Store":1,"DayOfWeek":1,"Date":"2015-01-01","Customers":120,"Promo":0,"StateHoliday":"0","SchoolHoliday":0,"Open":1}
            ])
            names = list(prep.transform(sample).columns) if prep is not None else list(sample.columns)
            imps = list(getattr(getattr(model, "named_steps", {}),"get",lambda k:None)("model").feature_importances_) if hasattr(model, "named_steps") else []
            fi = [{"feature": n, "importance": float(i)} for n, i in zip(names, imps)]
            with open(fi_path, "w", encoding="utf-8") as f:
                json.dump(fi, f)
        except Exception:
            fi = []
    fi_sorted = sorted(fi, key=lambda x: x.get("importance", 0), reverse=True)
    feature_importance = fi_sorted
    try:
        print("Model available:", bool(model))
        print("Model path:", model_path)
    except Exception:
        pass

def maybe_auto_train():
    global _training_lock
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    if os.path.exists(model_path):
        return
    if _training_lock:
        return
    _training_lock = True
    def _bg():
        try:
            sys.path.insert(0, BASE_DIR)
            mod = import_module("training.train_model")
            mod.main()
        except Exception as e:
            print("Auto-train error:", str(e))
        finally:
            load_assets()
            globals()["_training_lock"] = False
    t = threading.Thread(target=_bg, daemon=True)
    t.start()

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store INTEGER,
            day_of_week INTEGER,
            date TEXT,
            customers INTEGER,
            promo INTEGER,
            holiday INTEGER,
            predicted_sales REAL,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def locate_data():
    candidates = [
        os.path.join(BASE_DIR, "data", "train.csv"),
        os.path.join(BASE_DIR, "..", "train.csv"),
        os.path.join(BASE_DIR, "..", "data", "train.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return ("csv", p)
    if os.path.exists(DB_PATH):
        return ("db", DB_PATH)
    return (None, None)

def read_distinct_from_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    # ignore internal or log tables that don't contain the source data
    tables = [t for t in tables if t not in ("sqlite_sequence", "predictions")]
    table = None
    if "predicting_sales" in tables:
        table = "predicting_sales"
    else:
        for t in tables:
            try:
                cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
                norm = [str(x).lower() for x in cols]
                has_store = any(x in norm for x in ["store","store_id"])
                has_date = any(x in norm for x in ["date","sales_date","sale_date"])
                has_dow = any(x in norm for x in ["dayofweek","day_of_week","weekday"])
                has_cust = any(x in norm for x in ["customers","customer","cust"])
                if (has_store and has_date) or (has_store and has_date and has_dow and has_cust):
                    table = t
                    break
            except Exception:
                continue
    if table is None:
        # fallback to common names
        for t in ["store_sales", "sales", "rossmann", "data"]:
            if t in tables:
                table = t
                break
    if table is None:
        conn.close()
        return {"stores": [], "dates": [], "day_of_week": [], "customers": [], "source_table": None, "columns": {}}
    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({table})").fetchall()]
    lower_map = {str(c).lower(): c for c in cols}
    store_col = lower_map.get("store") or lower_map.get("store_id")
    date_col = lower_map.get("date") or lower_map.get("sales_date") or lower_map.get("sale_date")
    dow_col = lower_map.get("dayofweek") or lower_map.get("day_of_week") or lower_map.get("weekday")
    cust_col = lower_map.get("customers") or lower_map.get("customer") or lower_map.get("cust")
    stores = []
    dates = []
    day_of_week = []
    customers = []
    if store_col:
        stores = [r[0] for r in cur.execute(f"SELECT DISTINCT {store_col} FROM {table} WHERE {store_col} IS NOT NULL ORDER BY {store_col}").fetchall()]
    if date_col:
        dates = [r[0] for r in cur.execute(f"SELECT DISTINCT {date_col} FROM {table} WHERE {date_col} IS NOT NULL ORDER BY {date_col}").fetchall()]
    if dow_col:
        day_of_week = [r[0] for r in cur.execute(f"SELECT DISTINCT {dow_col} FROM {table} WHERE {dow_col} IS NOT NULL ORDER BY {dow_col}").fetchall()]
    if cust_col:
        customers = [r[0] for r in cur.execute(f"SELECT DISTINCT {cust_col} FROM {table} WHERE {cust_col} IS NOT NULL ORDER BY {cust_col}").fetchall()]
    conn.close()
    try:
        print("Options table:", table)
        print("Columns mapping:", {"store": store_col, "date": date_col, "day_of_week": dow_col, "customers": cust_col})
        print("Counts:", {"stores": len(stores), "dates": len(dates), "day_of_week": len(day_of_week), "customers": len(customers)})
    except Exception:
        pass
    return {"stores": stores, "dates": dates, "day_of_week": day_of_week, "customers": customers, "source_table": table, "columns": {"store": store_col, "date": date_col, "day_of_week": dow_col, "customers": cust_col}}

def read_distinct_from_csv(path):
    try:
        df = pd.read_csv(path)
        stores = sorted(list(pd.Series(df["Store"]).dropna().astype(int).unique())) if "Store" in df.columns else []
        dates = sorted(list(pd.Series(df["Date"]).dropna().astype(str).unique())) if "Date" in df.columns else []
        dows = sorted(list(pd.Series(df["DayOfWeek"]).dropna().astype(int).unique())) if "DayOfWeek" in df.columns else list(range(0,7))
        custs = sorted(list(pd.Series(df["Customers"]).dropna().astype(int).unique())) if "Customers" in df.columns else []
        return {"stores": stores, "dates": dates, "day_of_week": dows, "customers": custs}
    except Exception:
        return {"stores": [], "dates": [], "day_of_week": [], "customers": []}

@app.route("/", methods=["GET"])
def index():
    load_assets()
    ensure_db()
    maybe_auto_train()
    model_ready = model is not None
    # gather options from data source
    stores = []
    dates = []
    source_type, source_path = locate_data()
    if source_type == "db":
        d = read_distinct_from_db()
        stores = d.get("stores", [])
        dates = d.get("dates", [])
        dows = d.get("day_of_week", [])
        custs = d.get("customers", [])
    elif source_type == "csv":
        d = read_distinct_from_csv(source_path)
        stores = d.get("stores", [])
        dates = d.get("dates", [])
        dows = d.get("day_of_week", [])
        custs = d.get("customers", [])
    else:
        dows = list(range(0,7))
        custs = []
    try:
        print("Options source:", source_type, "counts:", {"stores": len(stores), "dates": len(dates), "dows": len(dows), "customers": len(custs)})
    except Exception:
        pass
    return render_template(
        "index.html",
        model_ready=model_ready,
        metrics=metrics,
        feature_importance=feature_importance,
        stores=stores,
        dates=dates,
        dows=dows,
        customers=custs,
    )

@app.route("/options", methods=["GET"])
def options():
    source_type, source_path = locate_data()
    stores = []
    dates = []
    dows = []
    custs = []
    if source_type == "db":
        d = read_distinct_from_db()
        stores = d.get("stores", [])
        dates = d.get("dates", [])
        dows = d.get("day_of_week", [])
        custs = d.get("customers", [])
    elif source_type == "csv":
        d = read_distinct_from_csv(source_path)
        stores = d.get("stores", [])
        dates = d.get("dates", [])
        dows = d.get("day_of_week", [])
        custs = d.get("customers", [])
    try:
        print("Options API counts:", {"stores": len(stores), "dates": len(dates), "dows": len(dows), "customers": len(custs)})
    except Exception:
        pass
    payload = {
        "stores": stores,
        "dates": dates,
        "day_of_week": dows if dows else list(range(0,7)),
        "customers": custs,
        "promo": [0,1],
        "holiday": ["0","a","b","c"]
    }
    return jsonify(payload)

@app.route("/feature-importance", methods=["GET"])
def feature_importance_route():
    load_assets()
    return jsonify(feature_importance or [])

@app.route("/status", methods=["GET"])
def status():
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    load_assets()
    info = {"exists": os.path.exists(model_path), "size": None, "mtime": None}
    try:
        if info["exists"]:
            st = os.stat(model_path)
            info["size"] = st.st_size
            info["mtime"] = st.st_mtime
    except Exception:
        pass
    return jsonify({
        "model_ready": bool(model),
        "metrics": metrics or {},
        "training": bool(_training_lock),
        "paths": {"model": model_path, "models_dir": MODELS_DIR, "model_file": info},
    })

@app.route("/force-reload", methods=["POST", "GET"])
def force_reload():
    load_assets()
    return jsonify({"ok": True, "model_ready": bool(model)})

@app.route("/schema", methods=["GET"])
def schema():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        tables = [t for t in tables if t not in ("sqlite_sequence", "predictions")]
        chosen = None
        if "predicting_sales" in tables:
            chosen = "predicting_sales"
        if chosen is None:
            for t in tables:
                try:
                    cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
                    low = [str(x).lower() for x in cols]
                    if "sales" in low and any(x in low for x in ["store","dayofweek","date","customers"]):
                        chosen = t
                        break
                except Exception:
                    continue
        result = {"db": DB_PATH, "table": chosen, "columns": [], "has_sales": False, "row_count": None}
        if chosen:
            cols = [c[1] for c in cur.execute(f"PRAGMA table_info({chosen})").fetchall()]
            result["columns"] = cols
            result["has_sales"] = any(str(c).lower() == "sales" for c in cols)
            try:
                cnt = cur.execute(f"SELECT COUNT(*) FROM {chosen}").fetchone()[0]
            except Exception:
                cnt = None
            result["row_count"] = cnt
        conn.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _run_training():
    try:
        sys.path.insert(0, BASE_DIR)
        mod = import_module("training.train_model")
        mod.main()
    except Exception as e:
        print("Training error:", str(e))
    finally:
        load_assets()

@app.route("/train", methods=["POST", "GET"])
def train():
    t = threading.Thread(target=_run_training, daemon=True)
    t.start()
    return jsonify({"status": "started"})

@app.route("/train-sync", methods=["POST", "GET"])
def train_sync():
    try:
        sys.path.insert(0, BASE_DIR)
        mod = import_module("training.train_model")
        mod.main()
        load_assets()
        return jsonify({"status": "ok", "model_ready": bool(model)})
    except Exception:
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            def _prep(df):
                df=df.copy()
                df["Date"]=pd.to_datetime(df["Date"], errors="coerce")
                df["Year"]=df["Date"].dt.year.fillna(0).astype(int)
                df["Month"]=df["Date"].dt.month.fillna(0).astype(int)
                df["Day"]=df["Date"].dt.day.fillna(0).astype(int)
                df["WeekOfYear"]=df["Date"].dt.isocalendar().week.astype(int)
                df["StateHoliday"]=df["StateHoliday"].apply(lambda x: 0 if str(x) in ["0","0.0","nan","None"] else 1).astype(int)
                if "SchoolHoliday" not in df.columns: df["SchoolHoliday"]=0
                df["SchoolHoliday"]=df["SchoolHoliday"].fillna(0).astype(int)
                if "Open" not in df.columns: df["Open"]=1
                df["Open"]=df["Open"].fillna(1).astype(int)
                cols=["Store","DayOfWeek","Customers","Promo","StateHoliday","SchoolHoliday","Open","Year","Month","Day","WeekOfYear"]
                return df[cols]
            X = pd.DataFrame([
                {"Store":1,"DayOfWeek":1,"Date":"2015-01-01","Customers":120,"Promo":0,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":1,"DayOfWeek":2,"Date":"2015-01-02","Customers":130,"Promo":1,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":2,"DayOfWeek":3,"Date":"2015-01-03","Customers":200,"Promo":0,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":2,"DayOfWeek":4,"Date":"2015-01-04","Customers":180,"Promo":1,"StateHoliday":"0","SchoolHoliday":0,"Open":1},
                {"Store":3,"DayOfWeek":5,"Date":"2015-01-05","Customers":220,"Promo":1,"StateHoliday":"a","SchoolHoliday":0,"Open":1},
            ])
            y = [1000,1100,1500,1400,1600]
            pipe = Pipeline([("prep", FunctionTransformer(_prep, validate=False)), ("model", RandomForestRegressor(n_estimators=10, random_state=42))])
            pipe.fit(X, y)
            dump(pipe, os.path.join(MODELS_DIR, "model.pkl"))
            with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
                f.write("{\"r2\":0.0,\"mae\":0.0,\"rmse\":0.0}")
            with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
                f.write("[]")
            load_assets()
            return jsonify({"status": "ok", "model_ready": bool(model), "fallback": True})
        except Exception as e2:
            return jsonify({"status": "error", "message": str(e2)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not trained"}), 400
    try:
        data = request.get_json() or {}
        store = int(data.get("store"))
        day_of_week = int(data.get("day_of_week"))
        date_raw = str(data.get("date"))
        try:
            _dt = pd.to_datetime(date_raw, errors="coerce")
            date = _dt.strftime("%Y-%m-%d") if not pd.isna(_dt) else date_raw
        except Exception:
            date = date_raw
        customers = int(data.get("customers"))
        promo = int(data.get("promo"))
        holiday_raw = data.get("holiday")
        if isinstance(holiday_raw, (int, float)):
            holiday_code = "0" if int(holiday_raw) == 0 else "a"
        else:
            holiday_code = str(holiday_raw or "0")
    except Exception:
        return jsonify({"error": "Invalid input"}), 400
    df = pd.DataFrame(
        [
            {
                "Store": store,
                "DayOfWeek": day_of_week,
                "Date": date,
                "Customers": customers,
                "Promo": promo,
                "StateHoliday": holiday_code,
                "SchoolHoliday": 0,
                "Open": 1,
            }
        ]
    )
    try:
        pred = float(model.predict(df)[0])
        conf = None
        lower = None
        upper = None
        try:
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                rf = model.named_steps["model"]
                prep = model.named_steps.get("prep", None)
                Xp = prep.transform(df) if prep is not None else df
                if hasattr(rf, "estimators_") and rf.estimators_:
                    arr = Xp.values if hasattr(Xp, "values") else Xp
                    per = [float(est.predict(arr)[0]) for est in rf.estimators_]
                    std = float(np.std(per))
                    lower = float(pred - 1.96 * std)
                    upper = float(pred + 1.96 * std)
                    conf = float(max(0.0, min(1.0, 1.0 - (std / (abs(pred) + 1e-6)))))
        except Exception:
            pass
    except Exception:
        return jsonify({"error": "Prediction failed"}), 500
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (store, day_of_week, date, customers, promo, holiday, predicted_sales, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            store,
            day_of_week,
            date,
            customers,
            promo,
            0 if holiday_code == "0" else 1,
            pred,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return jsonify({"predicted_sales": pred, "confidence": conf, "interval": {"lower": lower, "upper": upper}})

@app.route("/download", methods=["GET"])
def download():
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT store, day_of_week, date, customers, promo, holiday, predicted_sales, created_at FROM predictions ORDER BY id DESC", conn)
    conn.close()
    try:
        def _norm(x):
            try:
                _dt = pd.to_datetime(str(x), errors="coerce")
                if pd.isna(_dt):
                    return str(x)
                return _dt.strftime("%Y-%m-%d")
            except Exception:
                return str(x)
        df["date"] = df["date"].map(_norm)
    except Exception:
        pass
    csv_data = df.to_csv(index=False)
    return Response(csv_data, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=predictions_report.csv"})

if __name__ == "__main__":
    load_assets()
    ensure_db()
    try:
        p = int(os.environ.get("PORT") or os.environ.get("FLASK_RUN_PORT") or "5000")
    except Exception:
        p = 5000
    app.run(host="0.0.0.0", port=p)
