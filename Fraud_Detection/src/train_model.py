import os
import json
import pickle
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .preprocessing import preprocess
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve

def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def data_path() -> str:
    return os.path.join(project_root(), "data", "classification.db")

def models_dir() -> str:
    return os.path.join(project_root(), "models")

def list_tables(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cur.fetchall()
    return [r[0] for r in rows]

def load_data(limit_rows: int | None = 200000) -> pd.DataFrame:
    db_path = data_path()
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")
    conn = sqlite3.connect(db_path)
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

def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_col = "isFraud"
    if target_col not in df.columns:
        raise RuntimeError("Expected target column 'isFraud' not found in data")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

def train_and_save():
    df = load_data()
    X_raw, y = prepare_data(df)
    X_proc = preprocess(X_raw, expected_cols=None)
    mask = X_proc.notna().all(axis=1)
    X_proc = X_proc[mask]
    y = y[mask]
    feature_columns = list(X_proc.columns)
    imputer = SimpleImputer(strategy="median")
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, random_state=42)
    pipe = Pipeline(steps=[("impute", imputer), ("clf", clf)])
    X_train, X_temp, y_train, y_temp = train_test_split(X_proc, y, test_size=0.33, random_state=42, stratify=y)
    X_test, X_live, y_test, y_live = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    classes = np.array([0, 1])
    cls_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_map = {cls: w for cls, w in zip(classes, cls_weights)}
    sample_weight = y_train.map(weight_map).to_numpy()
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
    os.makedirs(models_dir(), exist_ok=True)
    with open(os.path.join(models_dir(), "fraud_model.pkl"), "wb") as f:
        pickle.dump(pipe, f)
    with open(os.path.join(models_dir(), "feature_columns.json"), "w", encoding="utf-8") as fcols:
        json.dump({"columns": feature_columns}, fcols)
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=3, random_state=42, scoring="roc_auc")
    importances = perm.importances_mean.tolist()
    feat_imp = [{"feature": c, "importance": float(i)} for c, i in zip(feature_columns, importances)]
    feat_imp = sorted(feat_imp, key=lambda x: x["importance"], reverse=True)
    with open(os.path.join(models_dir(), "feature_importances.json"), "w", encoding="utf-8") as fimps:
        json.dump({"importances": feat_imp}, fimps)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "count_test": int(len(y_test)),
        "fraud_in_train": int(y_train.sum()),
        "fraud_in_test": int(y_test.sum()),
        "fraud_in_live": int(y_live.sum()),
        "predicted_fraud_test": int(np.sum(y_pred)),
        "y_test_distribution": {int(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
        "y_pred_distribution": {int(k): int(v) for k, v in pd.Series(y_pred).value_counts().items()},
        "evaluated_at": datetime.now().isoformat()
    }
    with open(os.path.join(models_dir(), "metrics.json"), "w", encoding="utf-8") as fmet:
        json.dump(metrics, fmet)
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test, y_prob)
    pr_items = []
    for i in range(len(pr_thresh)):
        p = float(pr_prec[i+1])
        r = float(pr_rec[i+1])
        f1v = float((2*p*r)/(p+r)) if (p+r) > 0 else 0.0
        pr_items.append({"threshold": float(pr_thresh[i]), "precision": p, "recall": r, "f1": f1v})
    pr_items = sorted(pr_items, key=lambda x: x["threshold"])
    best_f1 = max(pr_items, key=lambda x: x["f1"]) if pr_items else {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    target_prec = max([x for x in pr_items if x["precision"] >= 0.8], key=lambda x: x["recall"], default=best_f1)
    target_rec = max([x for x in pr_items if x["recall"] >= 0.9], key=lambda x: x["precision"], default=best_f1)
    pr_payload = {
        "curve": pr_items[:3000],
        "suggestions": {
            "best_f1": best_f1,
            "precision_0_8": target_prec,
            "recall_0_9": target_rec
        }
    }
    with open(os.path.join(models_dir(), "pr_curve.json"), "w", encoding="utf-8") as fpr:
        json.dump(pr_payload, fpr)
    test_eval = {
        "y_test": pd.Series(y_test).astype(int).tolist(),
        "y_prob": pd.Series(y_prob).astype(float).tolist()
    }
    with open(os.path.join(models_dir(), "test_eval.json"), "w", encoding="utf-8") as fev:
        json.dump(test_eval, fev)

if __name__ == "__main__":
    train_and_save()
