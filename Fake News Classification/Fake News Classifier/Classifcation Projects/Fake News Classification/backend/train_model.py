import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
import csv
from datetime import datetime

from .preprocess import clean_text, normalize_label


TRAIN_COUNT = 55031
VAL_COUNT = 15723


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def db_path() -> Path:
    base = project_root()
    cands = [base / "classification.db"]
    for i, p in enumerate(base.parents):
        if i >= 3:
            break
        cands.append(p / "classification.db")
    for c in cands:
        if c.exists():
            return c
    return cands[0]


def discover_table(conn: sqlite3.Connection, preferred: str | None = None) -> str:
    if preferred:
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(%s)" % preferred)
            cols = [r[1] for r in cur.fetchall()]
            if {"text", "label"}.issubset(set(cols)):
                return preferred
        except Exception:
            pass
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    candidates = [
        "fack_news_classifier",
        "fake_news_classifier",
        "fake_news_classification",
        "news_classification",
    ]
    for t in candidates:
        if t in tables:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [r[1] for r in cur.fetchall()]
            if {"text", "label"}.issubset(set(cols)):
                return t
    # fallback: first table with required columns
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in cur.fetchall()]
        if {"text", "label"}.issubset(set(cols)):
            return t
    raise ValueError("No table with columns 'text' and 'label' found")


def load_data(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    q = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(q, conn)
    for col in ["Unnamed: 0.1", "Unnamed: 0", "Unnamed: 0_"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'text' and 'label' were not found in the table.")
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label_raw"] = df["label"].astype(str)
    df["label"] = df["label"].map(normalize_label)
    df = df.dropna(subset=["text", "label"])
    df = df[(df["label"] == 0) | (df["label"] == 1)]
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df[["text", "label", "label_raw"]]


def split_by_counts(df: pd.DataFrame, train_n: int, val_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    required = train_n + val_n
    if n < required:
        raise ValueError(f"Dataset has {n} rows, fewer than required {required} (train {train_n} + val {val_n}).")
    train_df = df.iloc[:train_n].copy()
    val_df = df.iloc[train_n:train_n + val_n].copy()
    prod_df = df.iloc[train_n + val_n:].copy()
    return train_df, val_df, prod_df


def train_and_eval(train_df: pd.DataFrame, val_df: pd.DataFrame, max_features: int = 20000, solver: str = "liblinear", class_weight: str | None = None, stop_words: str | None = "english", min_df: int = 2):
    vectorizer = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), max_features=max_features, stop_words=stop_words, min_df=min_df)
    X_train = vectorizer.fit_transform(train_df["text"].tolist())
    y_train = train_df["label"].astype(int).to_numpy()

    cw = None if not class_weight or class_weight.lower() == "none" else class_weight
    model = LogisticRegression(max_iter=1000, solver=solver, class_weight=cw)
    model.fit(X_train, y_train)

    X_val = vectorizer.transform(val_df["text"].tolist())
    y_val = val_df["label"].astype(int).to_numpy()

    y_pred = model.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1], average=None, zero_division=0)
    report = classification_report(y_val, y_pred, labels=[0, 1], output_dict=True, zero_division=0)

    return model, vectorizer, {
        "accuracy": acc,
        "precision": {"0": float(precision[0]), "1": float(precision[1])},
        "recall": {"0": float(recall[0]), "1": float(recall[1])},
        "f1": {"0": float(f1[0]), "1": float(f1[1])},
        "report": report,
    }


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def feature_importance(model: LogisticRegression, vectorizer: TfidfVectorizer, top_k: int = 20) -> dict:
    if not hasattr(model, "coef_"):
        return {}
    feats = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_pos_idx = np.argsort(coefs)[-top_k:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_k]
    top_pos = [(feats[i], float(coefs[i])) for i in top_pos_idx]
    top_neg = [(feats[i], float(coefs[i])) for i in top_neg_idx]
    return {"positive": top_pos, "negative": top_neg}


def save_artifacts(model, vectorizer, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(model, out_dir / "model.pkl")
    dump(vectorizer, out_dir / "vectorizer.pkl")


def update_project_summary(metrics: dict, val_size: int, prod_size: int):
    root = project_root()
    summary_csv = root / "Project_Summary.csv"
    exists = summary_csv.exists()
    header = [
        "Project Name",
        "Problem Identification",
        "Data Issues / EDA",
        "ML Model features",
        "Results",
        "Business Impact",
    ]
    res = f"accuracy={round(metrics.get('accuracy', 0.0), 4)}; when={datetime.utcnow().isoformat()}Z; val={val_size}; prod={prod_size}"
    row = [
        "Fake News Classification",
        "Binary classification of news factuality using text",
        "Missing text; label normalization; class imbalance; duplicates",
        "TF-IDF(LogReg); unigrams+bigrams; shared preprocessing",
        res,
        "Supports screening of misinformation; improves content trust",
    ]
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=str(db_path()), help="Path to classification.db")
    parser.add_argument("--out", type=str, default=str(project_root() / "models"))
    parser.add_argument("--table", type=str, default=None, help="Table name to read (overrides auto-detect)")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--solver", type=str, default="liblinear")
    parser.add_argument("--class_weight", type=str, default="balanced")
    parser.add_argument("--stop_words", type=str, default="english")
    parser.add_argument("--min_df", type=int, default=2)
    args = parser.parse_args()

    print("DB path:", args.db)
    if not os.path.exists(args.db):
        raise FileNotFoundError(f"Database not found at {args.db}")

    conn = sqlite3.connect(args.db)
    try:
        table = discover_table(conn, args.table)
        print("Using table:", table)
        df = load_data(conn, table)
    finally:
        conn.close()

    print("Total rows:", len(df))
    print("Label distribution (all):", df["label"].value_counts().to_dict())
    # Print class distribution in training subset
    train_df, val_df, prod_df = split_by_counts(df, TRAIN_COUNT, VAL_COUNT)
    print("Train class distribution:", train_df["label"].value_counts(normalize=True).to_dict())
    model, vectorizer, metrics = train_and_eval(
        train_df,
        val_df,
        max_features=args.max_features,
        solver=args.solver,
        class_weight=args.class_weight,
        stop_words=args.stop_words,
        min_df=args.min_df,
    )
    out_dir = Path(args.out)
    save_artifacts(model, vectorizer, out_dir)
    # Do not write threshold; app will default to 0.50
    import json
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": float(metrics.get("accuracy", 0.0))}, f)
    # Persist label map so UI reflects semantics correctly
    # Simple guess based on raw strings
    label0_name = "FAKE"
    label1_name = "TRUE"
    raws0 = train_df.loc[train_df["label"] == 0, "label_raw"].astype(str).str.lower()
    raws1 = train_df.loc[train_df["label"] == 1, "label_raw"].astype(str).str.lower()
    if any("real" in s or "true" in s or s.strip() == "1" for s in raws1.tolist()):
        label1_name = "TRUE"
    if any("fake" in s or "false" in s or s.strip() == "0" for s in raws0.tolist()):
        label0_name = "FAKE"
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"0": label0_name, "1": label1_name}, f)

    importances = feature_importance(model, vectorizer, top_k=args.top_k)

    print("Model: LogisticRegression + TF-IDF")
    print("Features: TF-IDF unigrams+bigrams, max_features=", args.max_features, "solver=", args.solver, "class_weight=", args.class_weight, "stop_words=", args.stop_words)
    print("Accuracy:", round(metrics["accuracy"], 4))
    X_val = vectorizer.transform(val_df["text"].tolist())
    y_val = val_df["label"].astype(int).to_numpy()
    y_pred = model.predict(X_val)
    from sklearn.metrics import classification_report
    print("Validation report:\n", classification_report(y_val, y_pred, labels=[0, 1], zero_division=0))
    print("Top Positive Features:", importances.get("positive", [])[:5])
    print("Top Negative Features:", importances.get("negative", [])[:5])
    print("Validation size:", len(val_df), "Production holdout size:", len(prod_df))

    update_project_summary(metrics, len(val_df), len(prod_df))


if __name__ == "__main__":
    main()
