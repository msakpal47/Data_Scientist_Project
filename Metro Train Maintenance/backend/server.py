import json
import os
import sqlite3
from datetime import datetime, timezone
import math

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
import threading
import time
import uuid
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from flask import abort
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
try:
    from imblearn.over_sampling import SMOTE  # optional
except Exception:
    SMOTE = None
try:
    from xgboost import XGBClassifier  # optional
except Exception:  # pragma: no cover
    XGBClassifier = None

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "classification.db")
TABLE = "fault_detection_manufacturing"

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fault_model.pkl")
MODEL_PROD_PATH = os.path.join(MODEL_DIR, "fault_model_production.pkl")
SCHEMA_PATH = os.path.join(MODEL_DIR, "schema.json")

FRONTEND_TEMPLATES = os.path.join(BASE_DIR, "..", "frontend", "templates")
FRONTEND_STATIC = os.path.join(BASE_DIR, "..", "frontend", "static")

# In-memory caches (avoid repeated disk I/O and DB scans)
MODEL_CACHE: dict | None = None
SCHEMA_CACHE: dict | None = None
_FV_CACHE = {}
TRAIN_JOBS = {}

FEATURES = [
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE",
    "ROTATIONSPEED",
    "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
    "TTF_FlowCool Pressure Dropped Below Limit",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "TTF_Flowcool leak",
]
TARGET = "fault_occurred"

app = Flask(__name__, static_folder=FRONTEND_STATIC, template_folder=FRONTEND_TEMPLATES)

def _is_leak_feature(name: str) -> bool:
    return name.lower().startswith("ttf_")

def _effective_features() -> list[str]:
    return [f for f in FEATURES if not _is_leak_feature(f)]

def _load_df(columns: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    if columns:
        col_sql = ", ".join([f'"{c}"' for c in columns])
    else:
        col_sql = "*"
    sql = f'SELECT {col_sql} FROM "{TABLE}"'
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(sql, con)
    con.close()
    return df

def _load_training_df(columns: list[str], max_negative_rows: int) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    col_sql = ", ".join([f'"{c}"' for c in columns])
    pos_sql = f'SELECT {col_sql} FROM "{TABLE}" WHERE "{TARGET}" = 1'
    neg_sql = (
        f'SELECT {col_sql} FROM "{TABLE}" WHERE "{TARGET}" = 0 '
        f"ORDER BY RANDOM() LIMIT {int(max_negative_rows)}"
    )
    df_pos = pd.read_sql_query(pos_sql, con)
    df_neg = pd.read_sql_query(neg_sql, con)
    con.close()
    return pd.concat([df_pos, df_neg], ignore_index=True)

def _clean_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TARGET])
    y = df[TARGET].astype(int)
    X = df[FEATURES]
    return X, y

def _build_pipeline(model_type: str = "logreg", calibrate: bool = False, calibration_method: str = "sigmoid", pos_weight: float | None = None, speed_mode: bool = False) -> Pipeline:
    if model_type == "rf":
        base_clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
    elif model_type == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed. Please add xgboost to requirements.")
        # scale_pos_weight approximates class imbalance
        base_clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=float(pos_weight) if pos_weight else 1.0,
        )
    else:
        base_clf = LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced",
            solver="lbfgs",
        )
    clf = base_clf
    if calibrate:
        clf = CalibratedClassifierCV(base_estimator=base_clf, method=calibration_method, cv=3)
    # Speed-friendly defaults
    if isinstance(base_clf, RandomForestClassifier):
        if speed_mode:
            base_clf.set_params(n_estimators=80, max_depth=10, n_jobs=-1, class_weight="balanced", random_state=42)
        else:
            base_clf.set_params(n_estimators=100, max_depth=12, n_jobs=-1, class_weight="balanced", random_state=42)
    try:
        # XGBoost speed adjustments
        from xgboost import XGBClassifier as _X
        if isinstance(base_clf, _X) and speed_mode:
            base_clf.set_params(n_estimators=120, max_depth=5, subsample=0.8, colsample_bytree=0.8, learning_rate=0.08)
    except Exception:
        pass
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])

def _compute_schema(df: pd.DataFrame, feature_names: list[str] | None = None) -> dict:
    feats = feature_names if feature_names is not None else _effective_features()
    schema = {"table": TABLE, "target": TARGET, "features": []}
    for f in feats:
        s = pd.to_numeric(df[f], errors="coerce")
        schema["features"].append(
            {
                "name": f,
                "min": float(np.nanmin(s)),
                "max": float(np.nanmax(s)),
                "median": float(np.nanmedian(s)),
            }
        )
    schema["generated_at"] = datetime.now(timezone.utc).isoformat()
    return schema

def _load_schema_file() -> dict | None:
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_schema(schema: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

def _get_schema_cached() -> dict:
    global SCHEMA_CACHE
    if SCHEMA_CACHE is None:
        schema = _load_schema_file()
        if schema is None:
            df = _load_df(columns=_effective_features() + [TARGET], limit=50000)
            schema = _compute_schema(df, feature_names=_effective_features())
            _save_schema(schema)
        SCHEMA_CACHE = schema
    return SCHEMA_CACHE

def _sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(x) for x in obj]
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return [_sanitize_json(x) for x in obj.tolist()]
        if isinstance(obj, (_np.floating, _np.integer)):
            obj = float(obj)
    except Exception:
        pass
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

def _load_model_bundle() -> dict | None:
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE
    if os.path.exists(MODEL_PATH):
        MODEL_CACHE = joblib.load(MODEL_PATH)
        return MODEL_CACHE
    return None

def _save_model_bundle(bundle: dict) -> None:
    global MODEL_CACHE
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    MODEL_CACHE = bundle

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.get("/api/schema")
def api_schema():
    return jsonify(_get_schema_cached())

@app.get("/api/time_columns")
def api_time_columns():
    cols = _list_time_columns()
    auto = _detect_time_column()
    return jsonify({"ok": True, "columns": cols, "auto": auto})

def _detect_time_column() -> str | None:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info('{TABLE}')")
    cols = cur.fetchall()
    con.close()
    names = [c[1] for c in cols]
    lower = [n.lower() for n in names]
    for cand in ["timestamp", "time", "datetime", "ts"]:
        if cand in lower:
            return names[lower.index(cand)]
    # Check types for datetime hints
    for c in cols:
        t = (c[2] or "").lower()
        if "date" in t or "time" in t:
            return c[1]
    return None

def _list_time_columns() -> list[str]:
    """Return a list of column names that look like time/timestamp/datetime."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info('{TABLE}')")
    cols = cur.fetchall()
    con.close()
    names = [c[1] for c in cols]
    types = [(c[2] or "").lower() for c in cols]
    out = []
    for n, t in zip(names, types):
        nl = n.lower()
        if any(k in nl for k in ["timestamp", "datetime", "time", "date"]) or any(k in t for k in ["timestamp", "datetime", "time", "date"]):
            out.append(n)
    # ensure deterministic order and de-dup
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq

def _train_from_payload(payload: dict) -> dict:
    t0 = time.perf_counter()
    speed_mode = bool(payload.get("speed_mode", False))
    max_negative_rows = payload.get("max_negative_rows", payload.get("max_rows", 200000))
    if speed_mode and not max_negative_rows:
        max_negative_rows = 50000
    max_negative_rows = int(max_negative_rows) if max_negative_rows else 200000
    model_type = str(payload.get("model_type", "logreg")).lower()
    calibrate = bool(payload.get("calibrate", False))
    if speed_mode:
        calibrate = False
    calibration_method = str(payload.get("calibration_method", "sigmoid"))
    solver = str(payload.get("solver", "liblinear" if model_type == "logreg" else "lbfgs"))
    if speed_mode and model_type == "logreg":
        solver = "liblinear"
    t1 = time.perf_counter()
    train_features = _effective_features()
    user_time_col = str(payload.get("timestamp_column", "")).strip()
    time_col = user_time_col if user_time_col else None
    split_type = (payload.get("split_type") or "").strip().lower()
    cols = train_features + [TARGET]
    if time_col and time_col not in cols:
        cols.append(time_col)
    use_temporal = bool(time_col) or split_type in {"temporal", "time", "time_based"}
    if use_temporal:
        # Load full dataframe to preserve temporal order, then split by time
        df = _load_df(columns=cols, limit=None)
    else:
        # Faster path with negative downsampling at SQL level
        df = _load_training_df(columns=cols, max_negative_rows=max_negative_rows)
    t2 = time.perf_counter()
    schema = _compute_schema(df, feature_names=train_features)
    _save_schema(schema)
    global SCHEMA_CACHE
    SCHEMA_CACHE = schema
    X = df[train_features].apply(pd.to_numeric, errors="coerce")
    y = df[TARGET].astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Target has <2 classes after cleaning.")
    # Initialize metrics dict early (used in split warnings below)
    metrics = {}
    split_strategy = "random"
    if (split_type in {"temporal", "time", "time_based"}) and not time_col:
        time_col = _detect_time_column()
    # Activate temporal split only if the time column actually exists in the loaded frame
    if time_col and time_col in df.columns:
        try:
            df_sorted = df.sort_values(by=time_col)
            X_sorted = df_sorted[train_features].apply(pd.to_numeric, errors="coerce")
            y_sorted = df_sorted[TARGET].astype(int)
            n = len(df_sorted)
            if n < 2:
                raise RuntimeError("Not enough rows for temporal split.")
            pos_mask_all = (y_sorted == 1)
            total_pos = int(pos_mask_all.sum())
            if total_pos == 0:
                # Train anyway, but flag invalid temporal test (no positives at all)
                idx = max(1, min(int(n * 0.8), n - 1))
                X_train, X_test = X_sorted.iloc[:idx], X_sorted.iloc[idx:]
                y_train, y_test = y_sorted.iloc[:idx], y_sorted.iloc[idx:]
                metrics["invalid_temporal_test"] = True
                metrics["split_warning"] = "no_positive_events_in_dataset"
            else:
                # Recommended production-safe split: boundary at last positive timestamp
                last_pos_time = df_sorted.loc[pos_mask_all, time_col].max()
                test_mask = df_sorted[time_col] >= last_pos_time
                train_mask = ~test_mask
                # Ensure both sides non-empty
                if test_mask.sum() == 0 or train_mask.sum() == 0:
                    raise RuntimeError("Temporal boundary produced empty train or test set.")
                X_train, X_test = X_sorted.loc[train_mask], X_sorted.loc[test_mask]
                y_train, y_test = y_sorted.loc[train_mask], y_sorted.loc[test_mask]
                # Ensure training contains at least one positive and negative where possible
                if y_train.nunique() < 2 and total_pos > 0:
                    pos_idx = np.where(pos_mask_all.values)[0]
                    cut_idx = int(pos_idx[-1])
                    cut_idx = max(1, min(cut_idx, n - 1))
                    X_train, X_test = X_sorted.iloc[:cut_idx], X_sorted.iloc[cut_idx:]
                    y_train, y_test = y_sorted.iloc[:cut_idx], y_sorted.iloc[cut_idx:]
            # Downsample negatives in TRAIN ONLY for speed/perf without altering temporal test
            if use_temporal and max_negative_rows and max_negative_rows > 0:
                train_pos_idx = y_train[y_train == 1].index
                train_neg_idx = y_train[y_train == 0].index
                if len(train_neg_idx) > max_negative_rows:
                    rng = np.random.RandomState(42)
                    keep_neg = rng.choice(train_neg_idx, size=max_negative_rows, replace=False)
                else:
                    keep_neg = train_neg_idx
                keep_idx = np.concatenate([train_pos_idx, keep_neg])
                keep_idx = np.sort(keep_idx)
                X_train = X_train.loc[keep_idx]
                y_train = y_train.loc[keep_idx]
            # Final guardrails and flags
            if y_train.nunique() < 2:
                raise RuntimeError("Temporal split resulted in a single-class training set.")
            split_strategy = "temporal"
            test_pos = int((y_test == 1).sum())
            if test_pos == 0:
                metrics["invalid_temporal_test"] = True
                metrics["split_warning"] = "temporal_test_no_positive"
        except Exception as e:
            raise RuntimeError(f"Temporal split failed: {e}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        if time_col:
            metrics["split_warning"] = f"time_column_not_found:{time_col}"
    # Ensure both classes appear in test where possible; keep temporal split if requested
    if split_strategy == "temporal":
        test_pos = int(pd.Series(y_test).sum())
        if test_pos == 0 and "invalid_temporal_test" not in metrics:
            metrics["invalid_temporal_test"] = True
            metrics["split_warning"] = "temporal_test_no_positive"
    else:
        # Random path: ensure both classes in test/train via stratification
        if len(pd.Series(y_test).unique()) < 2 or len(pd.Series(y_train).unique()) < 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            split_strategy = "random"
            metrics["split_warning"] = "random_split_stratified_adjusted"
    pos_weight = None
    if model_type == "xgb":
        n_pos = max(int((y_train == 1).sum()), 1)
        n_neg = max(int((y_train == 0).sum()), 1)
        pos_weight = n_neg / n_pos
    if model_type == "logreg" and solver in {"liblinear", "lbfgs"}:
        def _build_logreg(slv: str):
            return Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000 if slv == "liblinear" else 2000, n_jobs=-1, class_weight="balanced", solver=slv)),
            ])
        pipe = _build_logreg(solver)
        if calibrate:
            pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(base_estimator=LogisticRegression(max_iter=1000 if solver == "liblinear" else 2000, n_jobs=-1, class_weight="balanced", solver=solver), method=calibration_method, cv=3))
            ])
    else:
        pipe = _build_pipeline(model_type=model_type, calibrate=calibrate, calibration_method=calibration_method, pos_weight=pos_weight, speed_mode=speed_mode)
    smote_enabled = bool(payload.get("smote", False))
    t3 = time.perf_counter()
    if smote_enabled and SMOTE is not None:
        try:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            pipe.fit(X_res, y_res)
        except Exception:
            pipe.fit(X_train, y_train)
            metrics_note = "SMOTE_failed"
            metrics["smote_warning"] = True
    else:
        pipe.fit(X_train, y_train)
    t4 = time.perf_counter()
    # Metrics dict already initialized earlier
    # Dataset health info
    metrics["dataset_info"] = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_train": int((y_train == 1).sum()),
        "pos_test": int((y_test == 1).sum()),
    }
    # Optional cross validation on full dataset (stratified)
    cv_folds = 0
    try:
        cv_folds = int(payload.get("cv_folds", 0) or 0)
    except Exception:
        cv_folds = 0
    if cv_folds and cv_folds >= 2:
        try:
            cv_pipe = _build_pipeline(model_type=model_type, calibrate=calibrate, calibration_method=calibration_method, pos_weight=pos_weight, speed_mode=speed_mode)
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            f1_scores = cross_val_score(cv_pipe, X, y, cv=skf, scoring="f1")
            metrics["cv"] = {
                "folds": int(cv_folds),
                "f1_mean": float(np.mean(f1_scores)),
                "f1_std": float(np.std(f1_scores)),
                "f1_scores": [float(x) for x in f1_scores.tolist()],
            }
            try:
                if hasattr(cv_pipe, "predict_proba"):
                    proba_oof = cross_val_predict(cv_pipe, X, y, cv=skf, method="predict_proba")[:, 1]
                    metrics["cv"]["avg_precision"] = float(average_precision_score(y, proba_oof))
                    metrics["cv"]["roc_auc"] = float(roc_auc_score(y, proba_oof))
            except Exception:
                pass
        except Exception:
            # CV is best-effort; ignore failures
            pass
    try:
        if hasattr(pipe, "predict_proba"):
            y_prob = pipe.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            step = max(1, int(len(fpr) / 100))
            metrics["roc"] = {"fpr": [float(x) for x in fpr[::step]], "tpr": [float(x) for x in tpr[::step]]}
            precision, recall, thr = precision_recall_curve(y_test, y_prob)
            metrics["avg_precision"] = float(average_precision_score(y_test, y_prob))
            if len(thr) > 0:
                p = precision[:-1]
                r = recall[:-1]
                f1_arr = 2 * (p * r) / (p + r + 1e-8)
                best_idx = int(np.nanargmax(f1_arr))
                best_thr = float(thr[best_idx])
                metrics["best_threshold"] = best_thr
                # Compute primary metrics at the best threshold to align with UI
                y_hat_best = (y_prob >= best_thr).astype(int)
                acc_best = float(accuracy_score(y_test, y_hat_best))
                prec_best = float((y_hat_best & (y_test == 1)).sum() / (y_hat_best.sum() + 1e-8))
                rec_best = float((y_hat_best & (y_test == 1)).sum() / ((y_test == 1).sum() + 1e-8))
                f1_best = float(2 * prec_best * rec_best / (prec_best + rec_best + 1e-8))
                metrics["accuracy"] = acc_best
                metrics["precision"] = prec_best
                metrics["recall"] = rec_best
                metrics["f1"] = f1_best
                tn, fp, fn, tp = confusion_matrix(y_test, y_hat_best).ravel()
                metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
                # Best threshold subject to recall target (maximize precision)
                try:
                    recall_target = float(payload.get("recall_target", 0.7))
                except Exception:
                    recall_target = 0.7
                cand_idx = np.where(r >= recall_target)[0]
                if cand_idx.size > 0:
                    best_p_idx = int(cand_idx[np.nanargmax(p[cand_idx])])
                    best_thr_rt = float(thr[best_p_idx])
                    y_hat_rt = (y_prob >= best_thr_rt).astype(int)
                    tn_rt, fp_rt, fn_rt, tp_rt = confusion_matrix(y_test, y_hat_rt).ravel()
                    prec_rt = float(tp_rt / (tp_rt + fp_rt + 1e-8))
                    rec_rt = float(tp_rt / (tp_rt + fn_rt + 1e-8))
                    f1_rt = float(2 * prec_rt * rec_rt / (prec_rt + rec_rt + 1e-8))
                    metrics["recall_target"] = recall_target
                    metrics["best_threshold_at_recall"] = best_thr_rt
                    metrics["precision_at_recall_target"] = prec_rt
                    metrics["recall_at_recall_target"] = rec_rt
                    metrics["f1_at_recall_target"] = f1_rt
                    metrics["confusion_at_recall_target"] = {"tn": int(tn_rt), "fp": int(fp_rt), "fn": int(fn_rt), "tp": int(tp_rt)}
                # Evaluate confusion and metrics at common thresholds and best
                eval_thrs = [0.6, 0.7, 0.8, best_thr]
                if "best_threshold_at_recall" in metrics:
                    eval_thrs.append(float(metrics["best_threshold_at_recall"]))
                eval_thrs = sorted(set(eval_thrs))
                evals = []
                for th in eval_thrs:
                    y_hat = (y_prob >= th).astype(int)
                    tn_, fp_, fn_, tp_ = confusion_matrix(y_test, y_hat).ravel()
                    prec_e = float(tp_ / (tp_ + fp_ + 1e-8))
                    rec_e = float(tp_ / (tp_ + fn_ + 1e-8))
                    f1_e = float(2 * prec_e * rec_e / (prec_e + rec_e + 1e-8))
                    evals.append({
                        "threshold": float(th),
                        "precision": prec_e,
                        "recall": rec_e,
                        "f1": f1_e,
                        "confusion": {"tn": int(tn_), "fp": int(fp_), "fn": int(fn_), "tp": int(tp_)},
                    })
                metrics["thresholds_eval"] = evals
            pr_step = max(1, int(len(precision) / 100))
            metrics["pr"] = {"precision": [float(x) for x in precision[::pr_step]], "recall": [float(x) for x in recall[::pr_step]]}
    except Exception:
        pass
    # Fallback metrics if we couldn't compute via probabilities
    if "accuracy" not in metrics:
        y_pred = pipe.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        try:
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", pos_label=1, zero_division=0)
            metrics.update({"accuracy": acc, "precision": float(prec), "recall": float(rec), "f1": float(f1)})
        except Exception:
            metrics["accuracy"] = acc
        try:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        except Exception:
            pass
    importance = []
    try:
        est = pipe.named_steps["clf"]
        inner = None
        if hasattr(est, "coef_") or hasattr(est, "feature_importances_"):
            inner = est
        elif hasattr(est, "calibrated_classifiers_") and est.calibrated_classifiers_:
            inner = getattr(est.calibrated_classifiers_[0], "estimator", None)
        if inner is not None:
            if hasattr(inner, "coef_"):
                coefs = inner.coef_[0]
                trained_names = list(X_train.columns)
                for name, w in zip(trained_names, coefs):
                    importance.append({"name": name, "weight": float(w), "abs": float(abs(w)), "direction": "up" if w > 0 else "down"})
            elif hasattr(inner, "feature_importances_"):
                imps = inner.feature_importances_
                trained_names = list(X_train.columns)
                for name, w in zip(trained_names, imps):
                    importance.append({"name": name, "weight": float(w), "abs": float(abs(w)), "direction": None})
            importance.sort(key=lambda x: x["abs"], reverse=True)
            importance = importance[:10]
    except Exception:
        pass
    # Persist default operational threshold (prefer best@recall, else best F1)
    default_thr = float(metrics.get("best_threshold_at_recall") or metrics.get("best_threshold") or 0.5)
    bundle = {
        "model": pipe,
        "features": train_features,
        "target": TARGET,
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "model_type": model_type,
        "importance": importance,
        "split": split_strategy,
        "default_threshold": default_thr,
        "recall_target": float(payload.get("recall_target", 0.7)),
        "smote": bool(smote_enabled and SMOTE is not None),
    }
    if "dataset_info" in metrics:
        bundle["dataset_info"] = metrics["dataset_info"]
    if "invalid_temporal_test" in metrics:
        bundle["invalid_temporal_test"] = bool(metrics["invalid_temporal_test"])
    if "cv" in metrics:
        bundle["cv"] = {
            "folds": int(metrics["cv"].get("folds", 0)),
            "f1_mean": float(metrics["cv"].get("f1_mean", 0.0)),
            "f1_std": float(metrics["cv"].get("f1_std", 0.0)),
            "avg_precision": float(metrics["cv"].get("avg_precision", 0.0)) if isinstance(metrics["cv"].get("avg_precision", None), (int, float)) else None,
            "roc_auc": float(metrics["cv"].get("roc_auc", 0.0)) if isinstance(metrics["cv"].get("roc_auc", None), (int, float)) else None,
        }
    _save_model_bundle(bundle)
    t5 = time.perf_counter()
    print(f"timing:prep={t1-t0:.3f}s load_df={t2-t1:.3f}s build={t3-t2:.3f}s fit={t4-t3:.3f}s save={t5-t4:.3f}s total={t5-t0:.3f}s")
    return _sanitize_json({"ok": True, **metrics, "model_type": model_type, "importance": importance, "calibrated": calibrate, "calibration_method": calibration_method, "solver": solver, "speed_mode": speed_mode, "split": split_strategy, "default_threshold": default_thr, "recall_target": float(payload.get("recall_target", 0.7)), "smote": bool(smote_enabled and SMOTE is not None)})

@app.post("/api/train")
def api_train():
    payload = request.get_json(silent=True) or {}
    try:
        out = _train_from_payload(payload)
        return jsonify(out)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.post("/api/train_async")
def api_train_async():
    payload = request.get_json(silent=True) or {}
    job_id = str(uuid.uuid4())
    TRAIN_JOBS[job_id] = {"status": "queued", "result": None, "error": None, "started": None, "ended": None}
    def _worker():
        TRAIN_JOBS[job_id]["status"] = "running"
        TRAIN_JOBS[job_id]["started"] = datetime.now(timezone.utc).isoformat()
        try:
            with app.app_context():
                result = _train_from_payload(payload)
            TRAIN_JOBS[job_id]["result"] = _sanitize_json(result)
            TRAIN_JOBS[job_id]["status"] = "completed"
        except Exception as e:
            TRAIN_JOBS[job_id]["error"] = str(e)
            TRAIN_JOBS[job_id]["status"] = "failed"
        TRAIN_JOBS[job_id]["ended"] = datetime.now(timezone.utc).isoformat()
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return jsonify({"ok": True, "job_id": job_id})

@app.get("/api/train_status")
def api_train_status():
    job_id = request.args.get("job_id", type=str)
    if not job_id or job_id not in TRAIN_JOBS:
        return jsonify({"ok": False, "error": "invalid_job_id"}), 400
    return jsonify({"ok": True, **TRAIN_JOBS[job_id]})

@app.post("/api/predict")
def api_predict():
    bundle = _load_model_bundle()
    if bundle is None:
        return jsonify({"ok": False, "error": "Model not trained. Call /api/train first."}), 400

    payload = request.get_json(silent=True) or {}
    threshold = payload.get("threshold", None)
    try:
        threshold = float(threshold) if threshold is not None else None
    except Exception:
        threshold = None
    x = []
    for f in bundle["features"]:
        v = payload.get(f, None)
        x.append(np.nan if v is None else float(v))
    X = pd.DataFrame([x], columns=bundle["features"])
    model = bundle["model"]

    proba = None
    pred = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
        if threshold is None:
            threshold = float(bundle.get("default_threshold", 0.5))
        pred = int(1 if proba >= threshold else 0)
    else:
        pred = int(model.predict(X)[0])
    return jsonify({"ok": True, "prediction": pred, "probability": proba, "threshold": threshold})

@app.get("/api/feature_values")
def api_feature_values():
    name = request.args.get("name", type=str)
    limit = request.args.get("limit", default=200, type=int)
    if not name or name not in _effective_features():
        abort(400, description="Invalid or missing feature name.")
    limit = max(1, min(int(limit or 200), 2000))
    # Simple in-process cache to avoid repeated scans
    cache_key = (name, limit)
    now = datetime.now(timezone.utc).timestamp()
    ttl = 600.0
    global _FV_CACHE
    try:
        _FV_CACHE
    except NameError:
        _FV_CACHE = {}
    # purge expired
    for k in list(_FV_CACHE.keys()):
        if now - _FV_CACHE[k]["t"] > ttl:
            _FV_CACHE.pop(k, None)
    if cache_key in _FV_CACHE:
        vals = _FV_CACHE[cache_key]["v"]
    else:
        con = sqlite3.connect(DB_PATH)
        sql = (
            f'SELECT DISTINCT CAST("{name}" AS REAL) AS v '
            f'FROM "{TABLE}" WHERE "{name}" IS NOT NULL '
            f"ORDER BY v ASC LIMIT ?"
        )
        df = pd.read_sql_query(sql, con, params=(limit,))
        con.close()
        vals = [float(x) for x in df["v"].dropna().tolist()]
        _FV_CACHE[cache_key] = {"v": vals, "t": now}
    return jsonify({"name": name, "values": vals})

@app.post("/api/schema_refresh")
def api_schema_refresh():
    limit = request.get_json(silent=True) or {}
    lim = int(limit.get("limit", 50000))
    df = _load_df(columns=FEATURES + [TARGET], limit=lim)
    schema = _compute_schema(df)
    _save_schema(schema)
    global SCHEMA_CACHE
    SCHEMA_CACHE = schema
    return jsonify({"ok": True, "schema": schema})

@app.post("/api/model_promote")
def api_model_promote():
    """Save the current trained model bundle as the Production model."""
    bundle = _load_model_bundle()
    if bundle is None:
        return jsonify({"ok": False, "error": "No trained model to promote"}), 400
    # Enforce production guardrails: temporal split and SMOTE off
    if bundle.get("split") != "temporal":
        return jsonify({"ok": False, "error": "Production promotion requires Time-Based Split. Retrain with split set to temporal."}), 400
    if bundle.get("smote") is True:
        return jsonify({"ok": False, "error": "Production promotion requires SMOTE off. Retrain with SMOTE unchecked."}), 400
    # Additional guard: temporal test must include positive events
    if bundle.get("invalid_temporal_test") is True:
        return jsonify({"ok": False, "error": "Invalid time-based split: No positive events in test window."}), 400
    try:
        di = bundle.get("dataset_info") or {}
        if int(di.get("pos_test") or 0) == 0:
            return jsonify({"ok": False, "error": "Invalid time-based split: No positive events in test window."}), 400
    except Exception:
        pass
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle_copy = dict(bundle)
    bundle_copy["production"] = True
    bundle_copy["promoted_at"] = datetime.now(timezone.utc).isoformat()
    joblib.dump(bundle_copy, MODEL_PROD_PATH)
    return jsonify({"ok": True, "path": MODEL_PROD_PATH, "default_threshold": bundle_copy.get("default_threshold")})

@app.get("/api/model_info")
def api_model_info():
    """Return info about the currently loaded training model and production model (if any)."""
    info = {"ok": True}
    cur = _load_model_bundle()
    if cur:
        info["current"] = {
            "features": cur.get("features"),
            "model_type": cur.get("model_type"),
            "split": cur.get("split"),
            "default_threshold": cur.get("default_threshold"),
            "recall_target": cur.get("recall_target"),
        }
    if os.path.exists(MODEL_PROD_PATH):
        try:
            prod = joblib.load(MODEL_PROD_PATH)
            info["production"] = {
                "features": prod.get("features"),
                "model_type": prod.get("model_type"),
                "promoted_at": prod.get("promoted_at"),
                "default_threshold": prod.get("default_threshold"),
            }
        except Exception:
            info["production_error"] = "Failed to load production bundle"
    return jsonify(info)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=False)
