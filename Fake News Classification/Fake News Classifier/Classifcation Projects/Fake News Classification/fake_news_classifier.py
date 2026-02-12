import argparse
import os
import re
import sqlite3
import sys
from typing import Dict, List, Optional, Tuple
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import json


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cur.fetchall()
    return [r[0] for r in rows]


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    rows = cur.fetchall()
    return [r[1] for r in rows]


def find_candidate_table(conn: sqlite3.Connection) -> Optional[str]:
    tables = list_tables(conn)
    for t in tables:
        cols = [c.lower() for c in get_columns(conn, t)]
        if "text" in cols and "label" in cols:
            return t
    if tables:
        return tables[0]
    return None


def load_df(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c.lower() for c in df.columns]
    rename_map = {df.columns[i]: cols[i] for i in range(len(cols))}
    df = df.rename(columns=rename_map)
    if "unnamed: 0" in df.columns:
        df = df.drop(columns=["unnamed: 0"])
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.lower()
        df["label"] = df["label"].map({"true": 1, "false": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["text", "label"])
    df = df[(df["label"] == 0) | (df["label"] == 1)]
    df["text"] = df["text"].astype(str)
    return df


class NaiveBayes:
    def __init__(self):
        self.class_priors: Dict[int, float] = {}
        self.word_counts: Dict[int, Dict[str, int]] = {}
        self.total_words: Dict[int, int] = {}
        self.vocab: set = set()

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, texts: List[str], labels: List[int]):
        classes = sorted(set(labels))
        self.word_counts = {c: {} for c in classes}
        self.total_words = {c: 0 for c in classes}
        for t, y in zip(texts, labels):
            for w in self.tokenize(t):
                self.vocab.add(w)
                self.word_counts[y][w] = self.word_counts[y].get(w, 0) + 1
                self.total_words[y] += 1
        counts = {c: labels.count(c) for c in classes}
        n = len(labels)
        self.class_priors = {c: (counts[c] + 1) / (n + len(classes)) for c in classes}

    def log_prob(self, y: int, words: List[str]) -> float:
        V = len(self.vocab)
        total = self.total_words[y] + V
        s = np.log(self.class_priors[y])
        for w in words:
            cw = self.word_counts[y].get(w, 0) + 1
            s += np.log(cw / total)
        return float(s)

    def predict(self, texts: List[str]) -> List[int]:
        preds = []
        for t in texts:
            words = self.tokenize(t)
            scores = {c: self.log_prob(c, words) for c in self.class_priors.keys()}
            pred = max(scores.items(), key=lambda x: x[1])[0]
            preds.append(int(pred))
        return preds

    def predict_proba(self, texts: List[str]) -> List[float]:
        probs = []
        classes = sorted(self.class_priors.keys())
        for t in texts:
            words = self.tokenize(t)
            logs = np.array([self.log_prob(c, words) for c in classes], dtype=float)
            m = float(np.max(logs))
            exps = np.exp(logs - m)
            p = exps / np.sum(exps)
            idx1 = classes.index(1) if 1 in classes else 0
            probs.append(float(p[idx1]))
        return probs


def train_and_eval(df: pd.DataFrame, test_size: float, random_state: int) -> Tuple[NaiveBayes, dict]:
    rng = np.random.default_rng(random_state)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df)
    ts = int(n * test_size)
    X = df["text"].tolist()
    y = df["label"].astype(int).tolist()
    X_train, X_test = X[:-ts], X[-ts:]
    y_train, y_test = y[:-ts], y[-ts:]
    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tp = sum((np.array(y_test) == 1) & (np.array(y_pred) == 1))
    tn = sum((np.array(y_test) == 0) & (np.array(y_pred) == 0))
    fp = sum((np.array(y_test) == 0) & (np.array(y_pred) == 1))
    fn = sum((np.array(y_test) == 1) & (np.array(y_pred) == 0))
    acc = (tp + tn) / max(1, len(y_test))
    precision_pos = tp / max(1, tp + fp)
    recall_pos = tp / max(1, tp + fn)
    f1_pos = 2 * precision_pos * recall_pos / max(1e-12, precision_pos + recall_pos)
    precision_neg = tn / max(1, tn + fn)
    recall_neg = tn / max(1, tn + fp)
    f1_neg = 2 * precision_neg * recall_neg / max(1e-12, precision_neg + recall_neg)
    metrics = {
        "accuracy": float(acc),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": {
            "0": {"precision": float(precision_neg), "recall": float(recall_neg), "f1": float(f1_neg)},
            "1": {"precision": float(precision_pos), "recall": float(recall_pos), "f1": float(f1_pos)},
        },
    }
    return model, metrics


def save_artifacts(model: NaiveBayes, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "fake_news_model.json")
    data = {
        "class_priors": model.class_priors,
        "word_counts": model.word_counts,
        "total_words": model.total_words,
        "vocab": list(model.vocab),
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    info_path = os.path.join(out_dir, "model_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Model: NaiveBayes\n")
    return model_path, info_path


def infer(model: NaiveBayes, texts: List[str]) -> List[int]:
    return model.predict(texts)


def load_model_json(model_path: str) -> Optional[NaiveBayes]:
    if not os.path.exists(model_path):
        return None
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nb = NaiveBayes()
    nb.class_priors = {int(k): float(v) for k, v in data.get("class_priors", {}).items()}
    nb.word_counts = {int(k): {str(w): int(c) for w, c in v.items()} for k, v in data.get("word_counts", {}).items()}
    nb.total_words = {int(k): int(v) for k, v in data.get("total_words", {}).items()}
    nb.vocab = set(data.get("vocab", []))
    return nb


def summarize_and_update_csv(summary_csv: str, metrics: dict, sample_text: Optional[str], sample_pred: Optional[int]):
    if not os.path.isabs(summary_csv):
        summary_csv = os.path.join(os.path.dirname(__file__), summary_csv)
    exists = os.path.exists(summary_csv)
    header = [
        "Project Name",
        "Problem Identification",
        "Data Issues / EDA",
        "ML Model features",
        "Results",
        "Business Impact",
    ]
    res = f"accuracy={round(metrics.get('accuracy', 0.0), 4)}; cm={metrics.get('confusion_matrix')}; when={datetime.utcnow().isoformat()}Z"
    if sample_text is not None and sample_pred is not None:
        res += f"; sample_pred={sample_pred}"
    row = [
        "Fake News Classification",
        "Binary classification of news factuality using text",
        "Missing text; label normalization; class imbalance; duplicates",
        "NaiveBayes(word tokens); artifacts saved",
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
    parser.add_argument("--db", type=str, default="classification.db")
    parser.add_argument("--table", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--sample_text", type=str, default=None)
    parser.add_argument("--summary_csv", type=str, default="Project_Summary.csv")
    args = parser.parse_args()

    db_path = args.db
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(__file__), db_path)
    if not os.path.exists(db_path):
        print("Database not found:", db_path)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    try:
        table = args.table or find_candidate_table(conn)
        if table is None:
            print("No table found in database")
            sys.exit(1)
        df = load_df(conn, table)
        df = clean_df(df)
        model, metrics = train_and_eval(df, args.test_size, args.random_state)
        model_path, info_path = save_artifacts(model, os.path.join(os.path.dirname(__file__), args.out_dir))
        print("Accuracy:", round(metrics["accuracy"], 4))
        print("ConfusionMatrix:", metrics["confusion_matrix"])
        print("ModelSaved:", model_path)
        if args.sample_text:
            pred = infer(model, [args.sample_text])[0]
            print("SamplePred:", pred)
            summarize_and_update_csv(args.summary_csv, metrics, args.sample_text, pred)
        else:
            summarize_and_update_csv(args.summary_csv, metrics, None, None)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
