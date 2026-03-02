import os
import joblib
import pandas as pd
from src.data_preprocessing import preprocess_data


def _model_path() -> str:
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    return os.path.join(models_dir, "model.pkl")


def predict_aqi(input_dict: dict) -> float:
    model_file = _model_path()
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            "Model file not found. Train the model first to generate models/model.pkl"
        )

    model = joblib.load(model_file)

    df = pd.DataFrame([input_dict])
    X, _ = preprocess_data(df, training=False)

    prediction = model.predict(X)
    return float(prediction[0])
