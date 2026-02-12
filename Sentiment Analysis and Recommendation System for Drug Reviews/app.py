import os
import joblib
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.joblib")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)

model = joblib.load(MODEL_PATH)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "text is required"}), 400
    pred = int(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba([text])[0][pred])
    else:
        prob = None
    return jsonify({"sentiment": "positive" if pred == 1 else "negative", "label": pred, "confidence": prob})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
