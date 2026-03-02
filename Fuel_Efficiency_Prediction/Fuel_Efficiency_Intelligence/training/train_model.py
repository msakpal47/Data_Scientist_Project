import argparse
import os
import sqlite3
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def default_db_candidates():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [
        os.path.join(base, "data", "Regression.db"),
        os.path.join(base, "..", "regression.db"),
        os.path.join(os.getcwd(), "regression.db"),
    ]


def find_db(provided):
    if provided and os.path.exists(provided):
        return provided
    for c in default_db_candidates():
        if os.path.exists(c):
            return c
    return None


def read_table(db_path, table, limit=None):
    con = sqlite3.connect(db_path)
    if limit and limit > 0:
        q = f"SELECT * FROM '{table}' ORDER BY RANDOM() LIMIT {int(limit)}"
    else:
        q = f"SELECT * FROM '{table}'"
    df = pd.read_sql_query(q, con)
    con.close()
    return df


def infer_feature_sets(df, target):
    cols = [c for c in df.columns if c != target]
    num = []
    cat = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def build_pipeline(num_features, cat_features):
    num_pipe = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True))])
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([("ohe", enc)])
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_features), ("cat", cat_pipe, cat_features)]
    )
    model = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=0,
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    return pipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", type=str, default="")
    parser.add_argument("--table", type=str, default="fuel_efficiency_automobile")
    parser.add_argument("--target", type=str, default="Fuel consumption ")
    parser.add_argument("--limit", type=int, default=200000)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=0)
    parser.add_argument("--cv_sample", type=int, default=15000)
    args = parser.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)
    db_path = find_db(args.db_path)
    if not db_path:
        err = {"status": "error", "error": "db_not_found"}
        with open(os.path.join(models_dir, "run_log.json"), "w") as f:
            json.dump(err, f)
        print(json.dumps(err))
        return
    df = read_table(db_path, args.table, args.limit)
    if args.target not in df.columns:
        err = {"status": "error", "error": "target_not_found"}
        with open(os.path.join(models_dir, "run_log.json"), "w") as f:
            json.dump(err, f)
        print(json.dumps(err))
        return
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    df = df.dropna(subset=[args.target])
    num_cols, cat_cols = infer_feature_sets(df, args.target)
    X = df[num_cols + cat_cols]
    y = df[args.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
        )
    pipe = build_pipeline(num_cols, cat_cols)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    cv_mean = None
    cv_std = None
    if args.cv_folds and args.cv_folds > 1:
        n = min(args.cv_sample, len(X))
        X_cv = X.sample(n=n, random_state=args.random_state)
        y_cv = y.loc[X_cv.index]
        pipe_cv = build_pipeline(num_cols, cat_cols)
        if hasattr(pipe_cv.named_steps.get("model"), "set_params"):
            pipe_cv.named_steps["model"].set_params(n_estimators=min(300, pipe_cv.named_steps["model"].get_params().get("n_estimators", 300)))
        scores = cross_val_score(pipe_cv, X_cv, y_cv, scoring="r2", cv=args.cv_folds, n_jobs=1)
        cv_mean = float(np.mean(scores))
        cv_std = float(np.std(scores))
    model_path = os.path.join(models_dir, "fuel_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    feat_path = os.path.join(models_dir, "feature_columns.pkl")
    with open(feat_path, "wb") as f:
        pickle.dump({"numeric": num_cols, "categorical": cat_cols}, f)
    res = {
        "status": "ok",
        "model_path": model_path,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "rows": int(len(df)),
    }
    if cv_mean is not None:
        res["cv_r2_mean"] = cv_mean
        res["cv_r2_std"] = cv_std
    with open(os.path.join(models_dir, "run_log.json"), "w") as f:
        json.dump(res, f)
    print(json.dumps(res))


if __name__ == "__main__":
    main()

