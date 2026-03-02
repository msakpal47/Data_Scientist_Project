import argparse
import json
from pathlib import Path
from typing import List, Union

from joblib import load

from .preprocess import clean_text
import json


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def artifacts_dir() -> Path:
    models = project_root() / "models"
    if (models / "model.pkl").exists() and (models / "vectorizer.pkl").exists():
        return models
    return project_root() / "backend"


def load_artifacts():
    model = load(artifacts_dir() / "model.pkl")
    vectorizer = load(artifacts_dir() / "vectorizer.pkl")
    return model, vectorizer
def label_map() -> dict:
    lm = artifacts_dir() / "label_map.json"
    if lm.exists():
        try:
            return json.loads(lm.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"0": "FAKE", "1": "REAL"}


def predict_texts(texts: List[str]) -> List[dict]:
    model, vectorizer = load_artifacts()
    thresh = 0.5
    tfile = artifacts_dir() / "threshold.txt"
    if tfile.exists():
        pass
    clean_texts = [clean_text(t) for t in texts]
    X = vectorizer.transform(clean_texts)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= thresh).astype(int).tolist()
    lm = label_map()
    return [{"text": t, "pred": int(p), "label": lm.get(str(int(p))), "proba_true": float(s), "threshold": float(thresh)} for t, p, s in zip(texts, preds, probs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="Single text to classify")
    parser.add_argument("--file", type=str, default=None, help="Path to a file with one text per line")
    args = parser.parse_args()

    if args.text:
        res = predict_texts([args.text])
        print(json.dumps(res[0], ensure_ascii=False))
    elif args.file:
        lines: List[str] = []
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    lines.append(s)
        res = predict_texts(lines)
        print(json.dumps(res, ensure_ascii=False))
    else:
        print("Provide --text or --file")


if __name__ == "__main__":
    main()
