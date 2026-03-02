import os
import pickle
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from preprocess import split_features_target, fit_transform_scaler


def main():
    data_path = os.environ.get("DATA_CSV")
    target = os.environ.get("TARGET_COLUMN", "target")
    if not data_path:
        raise RuntimeError("DATA_CSV not set")
    df = pd.read_csv(data_path)
    X, y = split_features_target(df, target)
    Xs, scaler = fit_transform_scaler(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)
    joblib.dump(model, os.path.join(models_dir, "model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))


if __name__ == "__main__":
    main()

