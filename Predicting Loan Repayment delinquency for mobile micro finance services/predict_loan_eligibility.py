import argparse
import csv
import os

from loan_pipeline import (
    TABLE_NAME,
    TARGET_COLUMN,
    iter_rows_from_sqlite,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-path",
        default=os.path.join(os.path.dirname(__file__), "classification.db"),
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(os.path.dirname(__file__), "artifacts", "loan_eligibility_model.joblib"),
    )
    parser.add_argument("--offset", type=int, default=-1)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--output-csv",
        default=os.path.join(os.path.dirname(__file__), "artifacts", "live_predictions.csv"),
    )
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    model = load_model(args.model_path)
    if args.offset < 0:
        metadata_path = os.path.join(os.path.dirname(__file__), "artifacts", "train_metadata.json")
        if os.path.exists(metadata_path):
            import json

            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            args.offset = int(meta.get("live_offset", 0))
        else:
            args.offset = 0

    wrote_header = False
    total = 0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = None

        for chunk in iter_rows_from_sqlite(
            db_path=args.db_path,
            table_name=TABLE_NAME,
            chunk_size=args.chunk_size,
            offset=args.offset,
        ):
            if chunk is None or chunk.empty:
                break
            if TARGET_COLUMN in chunk.columns:
                chunk = chunk.drop(columns=[TARGET_COLUMN])

            proba = model.predict_proba(chunk)[:, 1]
            pred = (proba >= 0.5).astype(int)

            out = chunk.copy()
            out["prediction"] = pred
            out["probability"] = proba

            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(out.columns))
            if not wrote_header:
                writer.writeheader()
                wrote_header = True

            for row in out.to_dict(orient="records"):
                writer.writerow(row)

            total += len(out)
            print(f"Predicted rows: {total}")

            if args.max_rows and total >= args.max_rows:
                break

    if total == 0:
        raise RuntimeError("No rows found for prediction. Try a smaller --offset.")

    print(f"Saved predictions: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
