import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")


class SignalPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._model = None

    def load(self):
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("Model not trained")
            self._model = joblib.load(self.model_path)
        return self

    def predict_dbm(self, sample: dict) -> float:
        self.load()
        y = self._model.predict([sample])
        return float(y[0])

