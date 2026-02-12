import argparse
import json
import os
import sqlite3

import numpy as np

from loan_pipeline import (
    TABLE_NAME,
    TARGET_COLUMN,
    build_pipeline,
    evaluate_binary_classifier,
    read_rows_from_sqlite,
    save_artifacts,
)


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

    model = build_pipeline(random_state=args.random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_eval)[:, 1]
    else:
        scores = model.decision_function(X_eval)
        y_proba = 1 / (1 + np.exp(-scores))

    metrics = evaluate_binary_classifier(y_eval, y_pred, y_proba)
    extra_metadata = {
        "row_count": total_rows,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "live_offset": live_offset,
        "random_state": args.random_state,
    }
    save_artifacts(model, args.model_path, args.metadata_path, metrics, extra_metadata=extra_metadata)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {args.model_path}")
    print(f"Saved metadata: {args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
