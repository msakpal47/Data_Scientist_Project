import argparse
import json
import os
import sqlite3

import numpy as np
from sklearn.metrics import roc_curve

from loan_pipeline import (
    TABLE_NAME,
    TARGET_COLUMN,
    build_pipeline,
    evaluate_binary_classifier,
    read_rows_from_sqlite,
    save_artifacts,
)


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(y_proba)
    if thresholds.size > 1000:
        q = np.linspace(0.0, 1.0, 1001)
        thresholds = np.quantile(y_proba, q)
        thresholds = np.unique(thresholds)
    best_thr = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        tp = int(((y_true == 1) & (y_pred_t == 1)).sum())
        fp = int(((y_true == 0) & (y_pred_t == 1)).sum())
        fn = int(((y_true == 1) & (y_pred_t == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return best_thr, best_f1


def get_row_count(db_path: str) -> int:
    with sqlite3.connect(db_path) as con:
        return int(con.execute(f'SELECT COUNT(*) FROM "{TABLE_NAME}"').fetchone()[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        default=os.path.join(os.path.dirname(__file__), "classification.db"),
    )
    parser.add_argument("--train-rows", type=int, default=50_000)
    parser.add_argument("--eval-rows", type=int, default=20_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--tune-policy",
        choices=["f1", "recall_at_fpr"],
        default="f1",
    )
    parser.add_argument("--target-fpr", type=float, default=0.1)
    parser.add_argument(
        "--model-type",
        choices=["auto", "sgd", "rf", "xgb"],
        default="auto",
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.path.dirname(__file__), "artifacts", "loan_eligibility_model.joblib"),
    )
    parser.add_argument(
        "--metadata-path",
        default=os.path.join(os.path.dirname(__file__), "artifacts", "train_metadata.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    total_rows = get_row_count(args.db_path)
    train_rows = max(1, min(int(args.train_rows), total_rows))
    eval_rows = max(0, min(int(args.eval_rows), total_rows - train_rows))
    if eval_rows <= 0 and total_rows > 1:
        eval_rows = max(1, min(20_000, total_rows // 5))
        train_rows = max(1, total_rows - eval_rows)
    live_offset = train_rows + eval_rows

    train_df = read_rows_from_sqlite(
        db_path=args.db_path,
        table_name=TABLE_NAME,
        limit=train_rows,
        offset=0,
    )
    eval_df = read_rows_from_sqlite(
        db_path=args.db_path,
        table_name=TABLE_NAME,
        limit=eval_rows,
        offset=train_rows,
    )

    if TARGET_COLUMN not in train_df.columns:
        raise KeyError(f"Missing target column: {TARGET_COLUMN}")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()

    X_eval = eval_df.drop(columns=[TARGET_COLUMN])
    y_eval = eval_df[TARGET_COLUMN].astype(int).to_numpy()

    pos = int(np.sum(y_train))
    neg = int(len(y_train) - pos)
    spw = float(neg / pos) if pos > 0 else None

    model = build_pipeline(random_state=args.random_state, model_type=args.model_type, scale_pos_weight=spw)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval)[:, 1]
    else:
        scores = model.decision_function(X_eval)
        y_proba = 1 / (1 + np.exp(-scores))
    if args.tune_policy == "recall_at_fpr":
        fpr, tpr, thr = roc_curve(y_eval, y_proba)
        mask = fpr <= float(args.target_fpr)
        if not np.any(mask):
            idx = int(np.argmax(tpr))
        else:
            idx_rel = int(np.argmax(tpr[mask]))
            idx = int(np.flatnonzero(mask)[idx_rel])
        best_thr = float(thr[idx])
        best_f1 = None
        y_pred = (y_proba >= best_thr).astype(int)
        recall_at_target_fpr = float(tpr[idx])
        fpr_at_threshold = float(fpr[idx])
        threshold_policy = "recall_at_fpr"
    else:
        best_thr, best_f1 = find_best_threshold(y_eval, y_proba)
        y_pred = (y_proba >= best_thr).astype(int)
        recall_at_target_fpr = None
        fpr_at_threshold = None
        threshold_policy = "f1"

    metrics = evaluate_binary_classifier(y_eval, y_pred, y_proba)
    top_features = []
    try:
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        clf = model.named_steps["model"]
        importances = None
        if hasattr(clf, "feature_importances_"):
            importances = getattr(clf, "feature_importances_")
        elif hasattr(clf, "coef_"):
            importances = np.abs(getattr(clf, "coef_")).ravel()
        if importances is not None and len(importances) == len(feature_names):
            idx = np.argsort(importances)[::-1][:10]
            for i in idx:
                top_features.append({"feature": str(feature_names[i]), "importance": float(importances[i])})
    except Exception:
        top_features = []
    extra_metadata = {
        "row_count": total_rows,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "live_offset": live_offset,
        "random_state": args.random_state,
        "model_type": args.model_type,
        "scale_pos_weight": spw,
        "top_features": top_features,
        "threshold": best_thr,
        "best_f1": best_f1,
        "threshold_policy": threshold_policy,
        "target_fpr": float(args.target_fpr) if threshold_policy == "recall_at_fpr" else None,
        "recall_at_target_fpr": recall_at_target_fpr,
        "fpr_at_threshold": fpr_at_threshold,
    }
    save_artifacts(model, args.model_path, args.metadata_path, metrics, extra_metadata=extra_metadata)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {args.model_path}")
    print(f"Saved metadata: {args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
