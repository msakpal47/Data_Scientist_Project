from flask import Blueprint, render_template, request, jsonify
from .services.prediction_service import get_feature_columns, predict_from_payload
from .services.business_service import business_metrics

bp = Blueprint("routes", __name__)


@bp.route("/", methods=["GET"])
def index():
    columns = get_feature_columns()
    return render_template("index.html", columns=columns)


@bp.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        y = predict_from_payload(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    metrics = business_metrics(y)
    res = {"prediction": round(float(y), 2)}
    res.update(metrics)
    return jsonify(res)
