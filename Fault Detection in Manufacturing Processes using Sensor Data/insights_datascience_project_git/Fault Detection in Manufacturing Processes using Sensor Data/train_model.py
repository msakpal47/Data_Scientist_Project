import os
import sqlite3
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

BASE = r"e:\Data_Scientist_Project\Classifcation Projects\Fault Detection in Manufacturing Processes using Sensor Data"
DB_DEFAULT = os.path.join(BASE, "classification.db")
TABLE_DEFAULT = "fault_detection_manufacturing"
ARTIFACT_PATH = os.path.join(BASE, "model_artifacts.pkl")
REPORT_PATH = os.path.join(BASE, "final_report.txt")

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

def impute_median_fit(df, features):
    med = df[features].median(numeric_only=True)
    return med.to_dict()

def impute_median_apply(df, features, medians):
    x = df[features].copy()
    for c in features:
        x[c] = x[c].fillna(medians.get(c, x[c].median()))
    return x.values

def train_and_eval(train_df, eval_df, target, features):
    medians = impute_median_fit(train_df, features)
    X_train = impute_median_apply(train_df, features, medians)
    y_train = train_df[target].values
    X_eval = impute_median_apply(eval_df, features, medians)
    y_eval = eval_df[target].values
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_eval)
    try:
        prob = clf.predict_proba(X_eval)[:, 1]
    except:
        prob = preds.astype(float)
    acc = accuracy_score(y_eval, preds)
    prec = precision_score(y_eval, preds, zero_division=0)
    rec = recall_score(y_eval, preds, zero_division=0)
    f1 = f1_score(y_eval, preds, zero_division=0)
    roc = roc_auc_score(y_eval, prob) if len(np.unique(y_eval)) == 2 else np.nan
    cm = confusion_matrix(y_eval, preds, labels=[0, 1])
    return clf, medians, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc, "confusion_matrix": cm}

def write_artifacts(model, features, medians, path):
    payload = {"model": model, "features": features, "medians": medians}
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def write_report(model, features, metrics, path):
    lines = []
    lines.append("ML Model Selected: RandomForestClassifier")
    lines.append(f"Features Selected ({len(features)}):")
    for c in features:
        lines.append(f"- {c}")
    if hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        lines.append("Top Feature Importance:")
        for _, r in imp.head(20).iterrows():
            lines.append(f"- {r['feature']}: {r['importance']:.6f}")
    lines.append("Evaluation:")
    lines.append(f"- Accuracy: {metrics['accuracy']:.6f}")
    lines.append(f"- Precision: {metrics['precision']:.6f}")
    lines.append(f"- Recall: {metrics['recall']:.6f}")
    lines.append(f"- F1: {metrics['f1']:.6f}")
    lines.append(f"- ROC AUC: {metrics['roc_auc'] if not np.isnan(metrics['roc_auc']) else 'N/A'}")
    cm = metrics["confusion_matrix"]
    lines.append(f"- Confusion Matrix: [{cm[0,0]} {cm[0,1]} ; {cm[1,0]} {cm[1,1]}]")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DB_DEFAULT)
    parser.add_argument("--table", default=TABLE_DEFAULT)
    parser.add_argument("--train_count", type=int, default=500000)
    parser.add_argument("--eval_count", type=int, default=200000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)
    train_lim = 1000 if args.dry_run else args.train_count
    eval_lim = 1000 if args.dry_run else args.eval_count
    train_df = load_sql_range(conn, args.table, train_lim, 0)
    eval_df = load_sql_range(conn, args.table, eval_lim, 500000 if not args.dry_run else train_lim)
    target = select_target(train_df)
    features = infer_features(train_df, target)
    model, medians, metrics = train_and_eval(train_df, eval_df, target, features)
    write_artifacts(model, features, medians, ARTIFACT_PATH)
    write_report(model, features, metrics, REPORT_PATH)
    print("Artifacts:", ARTIFACT_PATH)
    print("Report:", REPORT_PATH)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
