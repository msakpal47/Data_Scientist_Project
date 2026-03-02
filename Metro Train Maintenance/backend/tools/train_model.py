import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import load_data, train_and_eval, MODEL_PATH, MODEL_DIR, FEATURES
import joblib

def main():
    df = load_data()
    acc, report, pipe, _ = train_and_eval(df)
    joblib.dump({"model": pipe, "features": FEATURES}, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)
    print("Accuracy:", acc)

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    main()
