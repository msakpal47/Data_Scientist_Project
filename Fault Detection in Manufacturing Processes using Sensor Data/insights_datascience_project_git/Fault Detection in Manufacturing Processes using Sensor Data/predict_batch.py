import os
import sqlite3
import pickle
import argparse
import numpy as np
import pandas as pd

BASE = r"e:\Data_Scientist_Project\Classifcation Projects\Fault Detection in Manufacturing Processes using Sensor Data"
DB_DEFAULT = os.path.join(BASE, "classification.db")
TABLE_DEFAULT = "fault_detection_manufacturing"
ARTIFACT_PATH = os.path.join(BASE, "model_artifacts.pkl")
OUTPUT_PATH = os.path.join(BASE, "predictions_live.csv")

def load_sql_range(conn, table, limit, offset):
    q = f"SELECT * FROM [{table}] ORDER BY rowid LIMIT {int(limit)} OFFSET {int(offset)}"
    return pd.read_sql_query(q, conn)

def select_target(df):
    return "fault_occurred" if "fault_occurred" in df.columns else df.columns[-1]

def infer_features(df, target):
    excluded = {"time", "stage", "Lot", "runnum", "recipe", "recipe_step", target}
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    feats = [c for c in num_cols if c not in excluded]
    const = [c for c in feats if df[c].nunique(dropna=True) <= 1]
    return [c for c in feats if c not in const]

def impute_median_apply(df, features, medians):
    x = df[features].copy()
    for c in features:
        x[c] = x[c].fillna(medians.get(c, x[c].median()))
    return x.values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_DEFAULT)
    parser.add_argument("--table", default=TABLE_DEFAULT)
    parser.add_argument("--live_offset", type=int, default=700000)
    parser.add_argument("--live_count", type=int, default=110643)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--out", default=OUTPUT_PATH)
    args = parser.parse_args()
    with open(ARTIFACT_PATH, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    features_saved = payload["features"]
    medians = payload["medians"]
    conn = sqlite3.connect(args.db)
    lim = 2000 if args.dry_run else args.live_count
    df = load_sql_range(conn, args.table, lim, args.live_offset)
    target = select_target(df)
    features_current = infer_features(df, target)
    features = [c for c in features_saved if c in features_current]
    X = impute_median_apply(df, features, medians)
    try:
        proba = model.predict_proba(X)[:, 1]
    except:
        proba = model.predict(X).astype(float)
    pred = model.predict(X)
    out = pd.DataFrame({"predicted_fault": pred, "probability": proba})
    if "time" in df.columns:
        out["time"] = df["time"]
    out.to_csv(args.out, index=False)
    print("Saved predictions:", args.out)

if __name__ == "__main__":
    main()
