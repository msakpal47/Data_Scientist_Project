from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
from src.predict import predict_aqi
from src.data_preprocessing import load_data

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    try:
        prediction = predict_aqi(data)
        category, advice = get_aqi_category(prediction)
        return jsonify({"predicted_AQI": round(prediction, 2), "category": category, "advice": advice})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/feature_importance")
def feature_importance():
    with open("models/feature_importance.json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/model_confidence")
def model_confidence():
    with open("models/model_confidence.json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/model_comparison")
def model_comparison():
    with open("models/model_comparison.json") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/trend")
def trend():
    df = load_data()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    last_24 = df.sort_values("Date").tail(24)
    return jsonify({"dates": last_24["Date"].astype(str).tolist(), "aqi": last_24["AQI"].tolist()})


def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory."
    elif aqi <= 100:
        return "Moderate", "Acceptable air quality."
    elif aqi <= 150:
        return "Unhealthy (Sensitive)", "Sensitive people should limit exposure."
    elif aqi <= 200:
        return "Unhealthy", "Health effects possible."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert issued."
    else:
        return "Hazardous", "Serious health risk."


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
