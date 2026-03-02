import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from loan_pipeline import TABLE_NAME, TARGET_COLUMN, read_rows_from_sqlite, load_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        default=os.path.join(os.path.dirname(__file__), "classification.db"),
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.path.dirname(__file__), "artifacts", "loan_eligibility_model.joblib"),
    )
    parser.add_argument(
        "--metadata-path",
        default=os.path.join(os.path.dirname(__file__), "artifacts", "train_metadata.json"),
    )
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument(
        "--output-json",
        default=os.path.join(os.path.dirname(__file__), "artifacts", "shap_top10.json"),
    )
    args = parser.parse_args()

    try:
        import shap  # type: ignore
    except Exception:
        print("SHAP not installed. Install with: pip install shap", file=sys.stderr)
        return 1

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_rows = int(meta.get("train_rows", 0))
    eval_rows = int(meta.get("eval_rows", 0))
    eval_offset = train_rows

    sample_n = max(1, min(int(args.sample_rows), max(eval_rows, 1)))

    df = read_rows_from_sqlite(
        db_path=args.db_path,
        table_name=TABLE_NAME,
        limit=sample_n,
        offset=eval_offset,
    )
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    model = load_model(args.model_path)

    try:
        X = model.named_steps["preprocess"].transform(model.named_steps["features"].transform(df))
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        X = df
        feature_names = getattr(df, "columns", None)

    clf = model.named_steps["model"]
    if hasattr(clf, "get_booster"):
        explainer = shap.TreeExplainer(clf)
    else:
        explainer = shap.Explainer(clf)

    try:
        shap_values = explainer(X)
        vals = np.abs(shap_values.values).mean(axis=0)
    except Exception:
        # Fallback: try transform to numpy
        X_np = np.asarray(X.todense() if hasattr(X, "todense") else X)
        shap_values = explainer(X_np)
        vals = np.abs(shap_values.values).mean(axis=0)

    if feature_names is None or len(feature_names) != len(vals):
        feature_names = [f"f{i}" for i in range(len(vals))]

    order = np.argsort(vals)[::-1][:10]
    top = [{"feature": str(feature_names[i]), "mean_abs_shap": float(vals[i])} for i in order]

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"top10": top}, f, indent=2)

    print(json.dumps({"top10": top}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
