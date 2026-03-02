import os
import json
import pickle
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


def detect_feature_types(df: pd.DataFrame, target: str) -> Tuple[list, list]:
    cols = [c for c in df.columns if c != target]
    num_cols = []
    cat_cols = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True, dtype=np.float32))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def candidate_models() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    models: Dict[str, Tuple[Any, Dict[str, list]]] = {
        "rf": (
            RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1),
            {},
        ),
        "et": (
            ExtraTreesRegressor(random_state=42, n_estimators=400, n_jobs=-1),
            {},
        ),
        "gbr": (
            GradientBoostingRegressor(random_state=42),
            {},
        ),
        "ridge": (
            Ridge(),
            {"regressor__alpha": [0.1, 1.0, 10.0]},
        ),
        "lasso": (
            Lasso(random_state=42),
            {},
        ),
        "en": (
            ElasticNet(random_state=42),
            {},
        ),
        "sgd": (
            SGDRegressor(random_state=42, max_iter=1000, tol=1e-3),
            {"regressor__alpha": [1e-4, 1e-3, 1e-2]},
        ),
    }
    try:
        import xgboost as xgb  # type: ignore
        models["xgb"] = (
            xgb.XGBRegressor(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                tree_method="hist",
            ),
            {},
        )
    except Exception:
        pass
    return models


def compute_feature_importance(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    try:
        pre = pipeline.named_steps["preprocessor"]
        feature_names = None
        try:
            feature_names = pre.get_feature_names_out()
            feature_names = [str(n).split("__", 1)[-1] for n in feature_names]
        except Exception:
            num_feats = pre.transformers_[0][2]
            cat_pipe = pre.transformers_[1][1]
            oh = cat_pipe.named_steps["oh"]
            cat_feats = pre.transformers_[1][2]
            cat_names = oh.get_feature_names_out(cat_feats).tolist()
            feature_names = list(num_feats) + cat_names
        try:
            Xt = pre.transform(X.head(500))
            n_transformed = Xt.shape[1]
        except Exception:
            n_transformed = len(feature_names) if feature_names is not None else 0
        if feature_names is None or len(feature_names) != n_transformed:
            feature_names = [f"f{i}" for i in range(n_transformed)]
        reg = pipeline.named_steps["regressor"]
        if hasattr(reg, "feature_importances_"):
            values = reg.feature_importances_
            if len(values) == len(feature_names):
                return {f: float(v) for f, v in zip(feature_names, values)}
        if hasattr(reg, "coef_"):
            import numpy as _np
            values = _np.abs(_np.ravel(reg.coef_))
            if len(values) == len(feature_names):
                return {f: float(v) for f, v in zip(feature_names, values)}
        if isinstance(X, pd.DataFrame) and len(X) > 1000:
            X = X.sample(1000, random_state=42)
            y = y.loc[X.index] if isinstance(y, pd.Series) else y[X.index]
        result = permutation_importance(pipeline, X, y, n_repeats=3, random_state=42, n_jobs=-1, scoring="r2")
        importances = result.importances_mean
        if len(importances) == len(feature_names):
            return {f: float(v) for f, v in zip(feature_names, importances)}
        return {f"f{i}": float(v) for i, v in enumerate(importances)}
    except Exception:
        return {}


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(csv_path: str, target_col: str = "charges", models_dir: str = "models") -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError("Target column not found")
    return train_df(df, target_col=target_col, models_dir=models_dir)


def train_df(df: pd.DataFrame, target_col: str = "charges", models_dir: str = "models", prefer_model: Optional[str] = None) -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    if target_col not in df.columns:
        raise ValueError("Target column not found")
    if len(df) > 50000:
        df = df.sample(50000, random_state=42).reset_index(drop=True)
    num_cols, cat_cols = detect_feature_types(df, target_col)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best = None
    best_name = None
    best_score = -np.inf
    best_metrics = {}
    search_results = []
    models = candidate_models()
    # optional: force a specific model name if provided and present
    items = {prefer_model: models[prefer_model]} if prefer_model and prefer_model in models else models
    for name, (reg, param_grid) in items.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", reg)])
        if param_grid:
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=3, scoring="r2", n_jobs=-1)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            r2_val = gs.best_score_
            info = {"name": name, "best_params": gs.best_params_, "cv_r2": float(r2_val)}
        else:
            pipe.fit(X_train, y_train)
            model = pipe
            y_cv_pred = model.predict(X_val)
            r2_val = r2_score(y_val, y_cv_pred)
            info = {"name": name, "best_params": {}, "cv_r2": float(r2_val)}
        y_pred = model.predict(X_val)
        y_pred_tr = model.predict(X_train)
        m = metrics(y_val, y_pred)
        m_train = {"r2_train": float(r2_score(y_train, y_pred_tr))}
        info.update(m)
        # explicit test alias for UI compatibility
        info["r2_test"] = float(m.get("r2", 0.0))
        info.update(m_train)
        search_results.append(info)
        if m["r2"] > best_score:
            best = model
            best_name = name
            best_score = m["r2"]
            best_metrics = m
    importance = compute_feature_importance(best, X_train, y_train if isinstance(y_train, (pd.Series, np.ndarray)) else np.array(y_train))
    model_path = os.path.join(models_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best, f)
    pre_path = os.path.join(models_dir, "scaler.pkl")
    with open(pre_path, "wb") as f:
        pickle.dump(best.named_steps["preprocessor"], f)
    cols_info = {
        "numeric": num_cols,
        "categorical": cat_cols,
        "target": target_col,
    }
    cols_path = os.path.join(models_dir, "columns.pkl")
    with open(cols_path, "wb") as f:
        pickle.dump(cols_info, f)
    imp_path = os.path.join(models_dir, "feature_importance.json")
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(importance, f)
    leaderboard_path = os.path.join(models_dir, "leaderboard.json")
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        json.dump(search_results, f)
    return {
        "model_path": model_path,
        "preprocessor_path": pre_path,
        "columns_path": cols_path,
        "importance_path": imp_path,
        "leaderboard_path": leaderboard_path,
        "best_model": best_name,
        "metrics": best_metrics,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", default="charges")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    res = train(args.csv, args.target, args.out)
    print(json.dumps(res))
