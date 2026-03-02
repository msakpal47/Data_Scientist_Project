import os
import pickle
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    X = np.arange(0, 100, dtype=float).reshape(-1, 1)
    y = 2 * X.ravel() + 1
    model = LinearRegression().fit(X, y)
    joblib.dump(model, os.path.join(models_dir, "model.pkl"))
    with open(os.path.join(models_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(["f0"], f)


if __name__ == "__main__":
    main()

