import json
from pathlib import Path
from joblib import load
from preprocess import clean_text

def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def artifacts_dir() -> Path:
    m = project_root() / "models"
    if (m / "model.pkl").exists() and (m / "vectorizer.pkl").exists():
        return m
    return Path(__file__).resolve().parent

def main():
    model = load(artifacts_dir() / "model.pkl")
    vectorizer = load(artifacts_dir() / "vectorizer.pkl")
    tfile = artifacts_dir() / "threshold.txt"
    thresh = 0.5
    if tfile.exists():
        try:
            thresh = float(tfile.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    print("classes_", getattr(model, "classes_", None))
    print("threshold", thresh)
    samples = [
        "United Nations member states adopted 17 Sustainable Development Goals as a shared framework for global development through 2030.",
        "NASA successfully landed the Perseverance rover on Mars to search for signs of ancient life.",
        "A viral post claims that chemtrails are sprayed to control weather and people.",
    ]
    X = vectorizer.transform([clean_text(s) for s in samples])
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= thresh).astype(int).tolist()
    out = [{"text": s, "proba_class1": float(p), "pred": int(y)} for s, p, y in zip(samples, probs, preds)]
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
