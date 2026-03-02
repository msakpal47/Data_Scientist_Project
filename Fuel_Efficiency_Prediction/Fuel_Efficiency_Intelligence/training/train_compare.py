import argparse
import os
import sqlite3
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def read_table(db_path, table, limit=None):
    con = sqlite3.connect(db_path)
    if limit and limit > 0:
        q = f"SELECT * FROM '{table}' ORDER BY RANDOM() LIMIT {int(limit)}"
    else:
        q = f"SELECT * FROM '{table}'"
    df = pd.read_sql_query(q, con)
    con.close()
    return df


def infer_sets(df, target):
    cols = [c for c in df.columns if c != target]
    num, cat = [], []
    for c in cols:
        (num if pd.api.types.is_numeric_dtype(df[c]) else cat).append(c)
    return num, cat


def build_pre(num_cols, cat_cols):
    num = Pipeline([("scaler", StandardScaler())])
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat = Pipeline([("ohe", enc)])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_path", type=str, default="")
    ap.add_argument("--table", type=str, default="fuel_efficiency_automobile")
    ap.add_argument("--target", type=str, default="Fuel consumption ")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--cv_sample", type=int, default=12000)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)
    db_candidates = [
        os.path.join(base, "data", "Regression.db"),
        os.path.join(base, "..", "regression.db"),
        os.path.join(os.getcwd(), "regression.db"),
    ]
    db_path = args.db_path or next((p for p in db_candidates if os.path.exists(p)), "")
    if not db_path:
        print(json.dumps({"status": "error", "error": "db_not_found"}))
        return
    df = read_table(db_path, args.table, args.limit)
    if args.target not in df.columns:
        print(json.dumps({"status": "error", "error": "target_not_found"}))
        return
    df = df.replace({np.inf: np.nan, -np.inf: np.nan}).dropna(subset=[args.target])
    num_cols, cat_cols = infer_sets(df, args.target)
    X = df[num_cols + cat_cols]
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)
    pre = build_pre(num_cols, cat_cols)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=None, random_state=args.random_state, n_jobs=0),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", objective="reg:squarederror", n_jobs=0, random_state=args.random_state),
    }

    leaderboard = []
    best = {"name": None, "r2": -1e9, "pipe": None}
    for name, est in models.items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = float(mean_squared_error(y_test, pred) ** 0.5)
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        cv_mean = None
        cv_std = None
        if args.cv_folds > 1:
            n = min(args.cv_sample, len(X))
            X_cv = X.sample(n=n, random_state=args.random_state)
            y_cv = y.loc[X_cv.index]
            scores = cross_val_score(Pipeline([("pre", pre), ("model", est)]), X_cv, y_cv, scoring="r2", cv=args.cv_folds, n_jobs=1)
            cv_mean = float(np.mean(scores))
            cv_std = float(np.std(scores))
        leaderboard.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2, "cv_r2_mean": cv_mean, "cv_r2_std": cv_std})
        if r2 > best["r2"]:
            best = {"name": name, "r2": r2, "pipe": pipe}

    with open(os.path.join(models_dir, "model_leaderboard.json"), "w") as f:
        json.dump({"rows": int(len(df)), "leaderboard": leaderboard, "selected": best["name"]}, f, indent=2)
    if best["pipe"] is not None:
        with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
            pickle.dump(best["pipe"], f)
    print(json.dumps({"status": "ok", "selected": best["name"], "r2": best["r2"]}))


if __name__ == "__main__":
    main()

