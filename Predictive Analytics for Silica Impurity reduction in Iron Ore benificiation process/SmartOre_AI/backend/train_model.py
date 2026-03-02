import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import time
from datetime import datetime


def _resolve_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(base_dir)
    db_path = os.path.join(project_root, "regression.db")
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    return db_path, model_path, scaler_path


def _find_table_and_target(conn):
    targets = [
        "% Silica Concentrate",
        "Silica Concentrate",
        "silica_concentrate",
        "SiO2 Concentrate",
        "SIO2 Concentrate",
        "si_conc",
        "silica",
    ]
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = [c[1] for c in cur.fetchall()]
        for target in targets:
            if target in cols:
                return t, target
    raise RuntimeError("No table with a silica concentrate target column was found")


def _load_dataframe(conn, table):
    df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
    return df


def _select_and_engineer_features(df):
    expected = [
        "% Iron Feed",
        "% Silica Feed",
        "Starch Flow",
        "Amina Flow",
        "Ore Pulp Flow",
        "Ore Pulp pH",
        "Ore Pulp Density",
        "Flotation Column 01 Air Flow",
        "Flotation Column 02 Air Flow",
        "Flotation Column 03 Air Flow",
        "Flotation Column 04 Air Flow",
        "Flotation Column 05 Air Flow",
        "Flotation Column 06 Air Flow",
        "Flotation Column 07 Air Flow",
        "Flotation Column 02 Level",
        "Flotation Column 03 Level",
        "Flotation Column 04 Level",
        "Flotation Column 05 Level",
        "Flotation Column 06 Level",
        "Flotation Column 07 Level",
    ]
    available = [c for c in expected if c in df.columns]
    X = df[available].copy()
    for c in X.columns:
        if X[c].dtype == object:
            s = X[c].astype(str).str.replace(r"[^\d,\.\-]+", "", regex=True)
            s = s.str.replace(",", ".", regex=False)
            X[c] = pd.to_numeric(s, errors="coerce")
    air_cols = [c for c in X.columns if "Air Flow" in c]
    if len(air_cols) > 0:
        X["Avg Air Flow"] = X[air_cols].mean(axis=1)
    feature_order = list(X.columns)
    return X, feature_order


def train_and_save():
    db_path, model_path, scaler_path = _resolve_paths()
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    try:
        table, target_col = _find_table_and_target(conn)
        df = _load_dataframe(conn, table)
    finally:
        conn.close()
    X_raw, feature_order = _select_and_engineer_features(df)
    if target_col not in df.columns:
        raise RuntimeError("Target column not found in dataframe")
    y_series = df[target_col]
    if y_series.dtype == object:
        ys = y_series.astype(str).str.replace(r"[^\d,\.\-]+", "", regex=True)
        ys = ys.str.replace(",", ".", regex=False)
        y_series = pd.to_numeric(ys, errors="coerce")
    y = y_series.values
    mask = ~np.isnan(y)
    X_raw = X_raw.loc[mask]
    y = y[mask]
    sample_frac = float(os.environ.get("TRAIN_SAMPLE_FRAC", "1.0"))
    if 0.0 < sample_frac < 1.0:
        rs = np.random.RandomState(42)
        n = max(1, int(len(y) * sample_frac))
        idx = rs.choice(len(y), size=n, replace=False)
        X_raw = X_raw.iloc[idx]
        y = y[idx]
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw.values)
    X_imputed = X_imputed.astype(np.float32, copy=False)
    scaler = None
    X_scaled = X_imputed
    algo = os.environ.get("MODEL_ALGO", "HGBR").upper()
    max_depth_env = os.environ.get("TRAIN_MAX_DEPTH", None)
    max_depth = int(max_depth_env) if (max_depth_env and max_depth_env.isdigit()) else None
    if algo == "HGBR":
        tune = (os.environ.get("TUNE_HGBR") or os.environ.get("TUNE") or "0").lower() in ("1", "true", "yes")
        if tune:
            X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            lrs = [0.05, 0.1, 0.2]
            depths = [6, 10, 12]
            iters = [150, 200]
            leaves = [10, 20]
            best = None
            best_params = None
            for lr in lrs:
                for d in depths:
                    for it in iters:
                        for msl in leaves:
                            m = HistGradientBoostingRegressor(
                                max_depth=d,
                                learning_rate=lr,
                                max_iter=it,
                                min_samples_leaf=msl,
                                random_state=42,
                            )
                            m.fit(X_tr, y_tr)
                            r2v = float(m.score(X_val, y_val))
                            if best is None or r2v > best:
                                best = r2v
                                best_params = {"max_depth": d, "learning_rate": lr, "max_iter": it, "min_samples_leaf": msl}
            model = HistGradientBoostingRegressor(random_state=42, **best_params)
            model.fit(X_scaled, y)
            medians = SimpleImputer(strategy="median").fit(X_raw.values).statistics_.tolist()
            r2 = float(best) if best is not None else float(model.score(X_scaled, y))
            meta = {
                "version": os.environ.get("MODEL_VERSION", "1.0"),
                "trained_at": datetime.utcnow().isoformat() + "Z",
                "r2": r2,
                "rows": int(len(y)),
            }
            importances = None
            payload = {"model": model, "features": feature_order, "medians": medians, "meta": meta, "importances": importances}
            joblib.dump(payload, model_path)
            joblib.dump(scaler, scaler_path)
            return {
                "model_path": model_path,
                "scaler_path": scaler_path,
                "features": feature_order,
                "rows_trained": int(len(y)),
                "elapsed_sec": round(time.time() - t0, 2),
                "algo": algo,
                "max_depth": best_params.get("max_depth") if best_params else (max_depth if max_depth is not None else "None"),
                "sample_frac": sample_frac,
            }
        else:
            model = HistGradientBoostingRegressor(
                max_depth=max_depth if max_depth is not None else 10,
                learning_rate=float(os.environ.get("TRAIN_LR", "0.1")),
                max_iter=int(os.environ.get("TRAIN_MAX_ITER", "150")),
                random_state=42,
            )
            # HGBR does not require scaling, keep scaler=None and X_scaled as X_imputed
    elif algo == "RIDGE":
        alpha = float(os.environ.get("TRAIN_ALPHA", "1.0"))
        model = Ridge(alpha=alpha)
        scaler = StandardScaler(copy=False)
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        model = HistGradientBoostingRegressor(
            max_depth=max_depth if max_depth is not None else 10,
            learning_rate=float(os.environ.get("TRAIN_LR", "0.1")),
            max_iter=int(os.environ.get("TRAIN_MAX_ITER", "150")),
            random_state=42,
        )
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    if isinstance(model, HistGradientBoostingRegressor):
        model_eval = HistGradientBoostingRegressor(
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            max_iter=model.max_iter,
            random_state=42,
        )
    elif isinstance(model, Ridge):
        model_eval = Ridge(alpha=model.alpha)
    else:
        model_eval = model
    model_eval.fit(X_tr, y_tr)
    y_tr_pred = model_eval.predict(X_tr)
    y_te_pred = model_eval.predict(X_te)
    r2_train = float(r2_score(y_tr, y_tr_pred))
    r2_test = float(r2_score(y_te, y_te_pred))
    mae_train = float(mean_absolute_error(y_tr, y_tr_pred))
    mae_test = float(mean_absolute_error(y_te, y_te_pred))
    rmse_train = float(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
    rmse_test = float(np.sqrt(mean_squared_error(y_te, y_te_pred)))
    model.fit(X_scaled, y)
    medians = imputer.statistics_.tolist()
    r2 = float(model.score(X_scaled, y))
    meta = {
        "version": os.environ.get("MODEL_VERSION", "1.0"),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "r2": r2,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "mae_train": mae_train,
        "mae_test": mae_test,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "rows": int(len(y)),
    }
    importances = None
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_.tolist()
        importances = [{"feature": f, "importance": float(v)} for f, v in zip(feature_order, vals)]
    elif hasattr(model, "coef_") and model.coef_ is not None:
        vals = np.abs(np.ravel(model.coef_)).tolist()
        importances = [{"feature": f, "importance": float(v)} for f, v in zip(feature_order, vals)]
    shap_top = None
    try:
        import shap
        rs = np.random.RandomState(42)
        n = min(2000, X_tr.shape[0])
        idx = rs.choice(X_tr.shape[0], size=n, replace=False)
        X_sample = X_tr[idx]
        explainer = shap.TreeExplainer(model_eval)
        sv = explainer.shap_values(X_sample)
        vals = np.abs(sv).mean(axis=0).tolist()
        shap_top = [{"feature": f, "importance": float(v)} for f, v in zip(feature_order, vals)]
    except Exception:
        shap_top = None
    payload = {"model": model, "features": feature_order, "medians": medians, "meta": meta, "importances": importances}
    if shap_top is not None:
        payload["shap_top"] = shap_top
    joblib.dump(payload, model_path)
    joblib.dump(scaler, scaler_path)
    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "features": feature_order,
        "rows_trained": int(len(y)),
        "elapsed_sec": round(time.time() - t0, 2),
        "algo": algo,
        "max_depth": max_depth if max_depth is not None else "None",
        "sample_frac": sample_frac,
    }


if __name__ == "__main__":
    os.environ.setdefault("MODEL_ALGO", "HGBR")
    os.environ.setdefault("TRAIN_SAMPLE_FRAC", "0.10")
    os.environ.setdefault("TRAIN_MAX_DEPTH", "12")
    os.environ.setdefault("TRAIN_LR", "0.1")
    os.environ.setdefault("TRAIN_MAX_ITER", "150")
    res = train_and_save()
    print(res)
