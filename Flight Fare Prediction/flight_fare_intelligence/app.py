from flask import Flask, render_template, request, jsonify
from flask import Response
import os
import math
from typing import Dict, List, Tuple
import pickle
import numpy as np
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

app = Flask(__name__, template_folder="templates", static_folder="static")


def heuristic_fare(travel_class, booking_lead_days, route, carrier, stops, duration_minutes):
    base = 50.0
    duration_component = 0.12 * float(duration_minutes)
    stops_component = 25.0 * float(stops)
    lead_time_component = -0.08 * float(booking_lead_days)
    route_hash = abs(hash(route)) % 100
    carrier_hash = abs(hash(carrier)) % 80
    demand_component = (route_hash + carrier_hash) * 0.1
    cls_mult = 1.0 if str(travel_class).lower() == "economy" else 1.8
    fare = (base + duration_component + stops_component + lead_time_component + demand_component) * cls_mult
    return max(30.0, round(fare, 2))


def heuristic_contributions(travel_class, booking_lead_days, route, carrier, stops, duration_minutes) -> Dict[str, float]:
    base = 50.0
    duration_component = 0.12 * float(duration_minutes)
    stops_component = 25.0 * float(stops)
    lead_time_component = -0.08 * float(booking_lead_days)
    route_hash = abs(hash(route)) % 100
    carrier_hash = abs(hash(carrier)) % 80
    demand_component = (route_hash + carrier_hash) * 0.1
    cls_mult = 1.0 if str(travel_class).lower() == "economy" else 1.8

    pre_mult_sum = base + duration_component + stops_component + lead_time_component + demand_component
    total = pre_mult_sum * cls_mult
    # Return component impacts approximately scaled by class multiplier
    return {
        "baseline": round(base * cls_mult, 2),
        "duration": round(duration_component * cls_mult, 2),
        "stops": round(stops_component * cls_mult, 2),
        "lead_time": round(lead_time_component * cls_mult, 2),
        "route_demand": round((route_hash * 0.1) * cls_mult, 2),
        "carrier_demand": round((carrier_hash * 0.1) * cls_mult, 2),
        "class_multiplier": round((cls_mult - 1.0) * pre_mult_sum if cls_mult != 1.0 else 0.0, 2),
        "predicted": max(30.0, round(total, 2)),
    }


def estimate_uncertainty(fare: float, travel_class: str, booking_lead_days: int, stops: int) -> Tuple[float, float, int]:
    cls = str(travel_class).lower()
    base_pct = 0.08 if cls == "economy" else 0.12
    if booking_lead_days <= 3:
        base_pct += 0.07
    if stops > 0:
        base_pct += 0.04
    ci_low = max(30.0, round(fare * (1 - base_pct), 2))
    ci_high = round(fare * (1 + base_pct), 2)
    confidence = 95
    if booking_lead_days <= 3:
        confidence -= 10
    if stops > 0:
        confidence -= 5
    confidence = max(60, min(95, confidence))
    return ci_low, ci_high, confidence


feature_order = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "duration",
    "days_left",
    "route",
    "is_weekend_departure",
    "duration_hours",
]

models_loaded = False
encoders = None
economy_model = None
business_model = None
economy_meta = None
business_meta = None

def load_models():
    global models_loaded, encoders, economy_model, business_model, economy_meta, business_meta
    try:
        with open(os.path.join("models", "encoders.pkl"), "rb") as f:
            encoders = pickle.load(f)
        with open(os.path.join("models", "economy_model.pkl"), "rb") as f:
            economy_model = pickle.load(f)
        with open(os.path.join("models", "business_model.pkl"), "rb") as f:
            business_model = pickle.load(f)
        emeta_path = os.path.join("models", "economy_meta.pkl")
        bmeta_path = os.path.join("models", "business_meta.pkl")
        if os.path.exists(emeta_path):
            with open(emeta_path, "rb") as f:
                economy_meta = pickle.load(f)
        if os.path.exists(bmeta_path):
            with open(bmeta_path, "rb") as f:
                business_meta = pickle.load(f)
        models_loaded = True
    except Exception:
        models_loaded = False

load_models()

def safe_transform(col, value):
    le = encoders.get(col)
    if le is None:
        return value
    try:
        return int(le.transform([str(value)])[0])
    except Exception:
        return int(0)

def predict_with_model(payload):
    cls = payload.get("class_type", "Economy")
    if not models_loaded:
        return None
    # Derive engineered features server-side
    source_city = payload.get("source_city")
    destination_city = payload.get("destination_city")
    route = f"{source_city}_{destination_city}"
    dep_time = payload.get("departure_time")
    duration = float(payload.get("duration", payload.get("duration_minutes", 120)))
    is_weekend_departure = 1 if str(dep_time) in ("Evening", "Night") else 0
    duration_hours = duration / 60.0
    enriched = {
        **payload,
        "route": route,
        "is_weekend_departure": is_weekend_departure,
        "duration_hours": duration_hours,
    }
    x_vals = []
    for col in feature_order:
        val = enriched.get(col)
        if isinstance(val, str):
            val = safe_transform(col, val)
        x_vals.append(val)
    X = np.array([x_vals], dtype=float)
    if str(cls).lower() == "economy":
        y = economy_model.predict(X)
        meta = economy_meta or {}
    else:
        y = business_model.predict(X)
        meta = business_meta or {}
    price = float(y[0])
    sigma = float(meta.get("residual_std", 0.0))
    if sigma > 0:
        ci_low = max(0.0, price - 1.96 * sigma)
        ci_high = price + 1.96 * sigma
        confidence = 95
    else:
        ci_low, ci_high, confidence = estimate_uncertainty(price, cls, int(payload.get("days_left", 30)), int(payload.get("stops", 0)))
    return {"predicted": round(price, 2), "ci_low": round(ci_low, 2), "ci_high": round(ci_high, 2), "confidence": confidence}

def shap_contributions(payload):
    if not (models_loaded and HAS_SHAP):
        return None
    cls = payload.get("class_type", "Economy")
    # Build enriched vector and encoded X
    source_city = payload.get("source_city")
    destination_city = payload.get("destination_city")
    route = f"{source_city}_{destination_city}"
    dep_time = payload.get("departure_time")
    duration = float(payload.get("duration", payload.get("duration_minutes", 120)))
    is_weekend_departure = 1 if str(dep_time) in ("Evening", "Night") else 0
    duration_hours = duration / 60.0
    enriched = {
        **payload,
        "route": route,
        "is_weekend_departure": is_weekend_departure,
        "duration_hours": duration_hours,
    }
    x_vals = []
    for col in feature_order:
        val = enriched.get(col)
        if isinstance(val, str):
            val = safe_transform(col, val)
        x_vals.append(val)
    X = np.array([x_vals], dtype=float)
    model = economy_model if str(cls).lower() == "economy" else business_model
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer(X)
        # shap 0.45 returns Explanation with .values
        if hasattr(sv, "values"):
            vals = sv.values
        else:
            vals = sv
        row = vals[0]
        items = [{"feature": feature_order[i], "value": float(row[i])} for i in range(len(feature_order))]
        items_sorted = sorted(items, key=lambda x: abs(x["value"]), reverse=True)
        return items_sorted
    except Exception:
        return None

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict(flat=True)

    if models_loaded and "airline" in data:
        res = predict_with_model(data)
        if res:
            return jsonify({"predicted_fare": res["predicted"], "predicted_price": res["predicted"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "confidence": res["confidence"]})
    travel_class = data.get("travel_class", "Economy")
    booking_lead_days = int(data.get("booking_lead_days", int(data.get("days_left", 30))))
    route = data.get("route") or f"{data.get('source_city','DEL')}-{data.get('destination_city','BOM')}"
    carrier = data.get("carrier") or data.get("airline") or "GenericAir"
    stops = int(data.get("stops", 0))
    duration_minutes = int(data.get("duration_minutes", int(data.get("duration", 120))))
    fare = heuristic_fare(travel_class, booking_lead_days, route, carrier, stops, duration_minutes)
    ci_low, ci_high, confidence = estimate_uncertainty(fare, travel_class, booking_lead_days, stops)
    return jsonify({"predicted_fare": fare, "predicted_price": fare, "ci_low": ci_low, "ci_high": ci_high, "confidence": confidence})


@app.post("/price_trend")
def price_trend():
    data = request.get_json(silent=True) or {}
    travel_class = data.get("travel_class", "Economy")
    route = data.get("route", "DEL-BOM")
    carrier = data.get("carrier", "GenericAir")
    stops = int(data.get("stops", 0))
    duration_minutes = int(data.get("duration_minutes", 120))
    max_days = int(data.get("max_days", 120))
    step = int(data.get("step", 5))

    days = list(range(0, max_days + 1, step))
    if models_loaded and "airline" in data:
        payload = data.copy()
        prices = []
        for d in days:
            payload["days_left"] = d
            r = predict_with_model(payload) or {}
            prices.append(r.get("predicted", 0.0))
    else:
        prices = [heuristic_fare(travel_class, d, route, carrier, stops, duration_minutes) for d in days]
    return jsonify({"days": days, "prices": prices})


@app.post("/recommend")
def recommend():
    data = request.get_json(silent=True) or {}
    travel_class = data.get("travel_class", "Economy")
    booking_lead_days = int(data.get("booking_lead_days", 30))
    route = data.get("route", "DEL-BOM")
    carrier = data.get("carrier", "GenericAir")
    stops = int(data.get("stops", 0))
    duration_minutes = int(data.get("duration_minutes", 120))
    horizon = int(data.get("horizon", 90))
    step = int(data.get("step", 3))

    # Check future window [0..booking_lead_days + horizon] for minimum price
    future_days = list(range(0, booking_lead_days + horizon + 1, step))
    if models_loaded and "airline" in data:
        payload = data.copy()
        prices = []
        for d in future_days:
            payload["days_left"] = d
            r = predict_with_model(payload) or {}
            prices.append(r.get("predicted", 0.0))
        r_now = predict_with_model({**data, "days_left": booking_lead_days}) or {}
        current_price = r_now.get("predicted", 0.0)
    else:
        prices = [
            heuristic_fare(travel_class, d, route, carrier, stops, duration_minutes) for d in future_days
        ]
        current_price = heuristic_fare(travel_class, booking_lead_days, route, carrier, stops, duration_minutes)
    min_price = min(prices)
    min_idx = prices.index(min_price)
    best_day = future_days[min_idx]

    threshold = min_price * 1.005  # 0.5% tolerance
    action = "Book now" if current_price <= threshold else "Wait"
    savings = round(max(0.0, current_price - min_price), 2)

    return jsonify({
        "current_price": current_price,
        "recommended_lead_days": best_day,
        "recommended_price": min_price,
        "action": action,
        "potential_savings": savings
    })


@app.post("/explain")
def explain():
    data = request.get_json(silent=True) or {}
    travel_class = data.get("travel_class", "Economy")
    booking_lead_days = int(data.get("booking_lead_days", 30))
    route = data.get("route", "DEL-BOM")
    carrier = data.get("carrier", "GenericAir")
    stops = int(data.get("stops", 0))
    duration_minutes = int(data.get("duration_minutes", 120))

    if models_loaded and "airline" in data:
        # Try SHAP first
        if HAS_SHAP:
            items_sorted = shap_contributions(data)
            if items_sorted:
                r = predict_with_model(data) or {}
                return jsonify({"method": "shap", "predicted": r.get("predicted", 0.0), "contributions": items_sorted})
        # Fallback to model feature importances
        cls = data.get("class_type", "Economy")
        model = economy_model if str(cls).lower() == "economy" else business_model
        if hasattr(model, "feature_importances_"):
            fi = list(model.feature_importances_)
            items = [{"feature": feature_order[i], "value": round(float(fi[i]), 6)} for i in range(len(feature_order))]
            items_sorted = sorted(items, key=lambda x: x["value"], reverse=True)
            r = predict_with_model(data) or {}
            return jsonify({"method": "model_importances", "predicted": r.get("predicted", 0.0), "contributions": items_sorted})
    contrib = heuristic_contributions(travel_class, booking_lead_days, route, carrier, stops, duration_minutes)
    items = [{"feature": k, "value": v} for k, v in contrib.items() if k not in ("predicted",)]
    items_sorted = sorted(items, key=lambda x: abs(x["value"]), reverse=True)
    return jsonify({"method": "heuristic_fallback", "predicted": contrib["predicted"], "contributions": items_sorted})


def create_app():
    return app


@app.post("/reload_models")
def reload_models():
    load_models()
    return jsonify({"models_loaded": models_loaded})


@app.get("/model_metrics")
def model_metrics():
    if not models_loaded:
        return jsonify({"models_loaded": False})
    e = economy_meta or {}
    b = business_meta or {}
    return jsonify({
        "models_loaded": True,
        "economy": {
            "r2": e.get("r2"),
            "mae": e.get("mae"),
            "cv_r2_mean": e.get("cv_r2_mean"),
            "cv_r2_std": e.get("cv_r2_std"),
            "dataset_rows": e.get("dataset_rows"),
            "train_size": e.get("train_size"),
            "test_size": e.get("test_size"),
            "model_version": e.get("model_version"),
            "trained_at": e.get("trained_at"),
        },
        "business": {
            "r2": b.get("r2"),
            "mae": b.get("mae"),
            "cv_r2_mean": b.get("cv_r2_mean"),
            "cv_r2_std": b.get("cv_r2_std"),
            "dataset_rows": b.get("dataset_rows"),
            "train_size": b.get("train_size"),
            "test_size": b.get("test_size"),
            "model_version": b.get("model_version"),
            "trained_at": b.get("trained_at"),
        }
    })


@app.get("/project_summary.csv")
def project_summary_csv():
    headers = ["section", "key", "value"]
    rows = []
    rows.append(("system", "models_loaded", str(models_loaded)))
    rows.append(("system", "shap_enabled", str(HAS_SHAP)))
    rows.append(("system", "feature_count", str(len(feature_order))))
    rows.append(("system", "features", "|".join(feature_order)))
    if models_loaded:
        econ_cls = type(economy_model).__name__ if economy_model is not None else ""
        bus_cls = type(business_model).__name__ if business_model is not None else ""
        rows.append(("economy_model", "class", econ_cls))
        rows.append(("business_model", "class", bus_cls))
        for seg, meta in (("economy", economy_meta or {}), ("business", business_meta or {})):
            rows.append((seg, "r2", str(meta.get("r2"))))
            rows.append((seg, "mae", str(meta.get("mae"))))
            rows.append((seg, "cv_r2_mean", str(meta.get("cv_r2_mean"))))
            rows.append((seg, "cv_r2_std", str(meta.get("cv_r2_std"))))
            rows.append((seg, "dataset_rows", str(meta.get("dataset_rows"))))
            rows.append((seg, "train_size", str(meta.get("train_size"))))
            rows.append((seg, "test_size", str(meta.get("test_size"))))
            rows.append((seg, "model_version", str(meta.get("model_version"))))
            rows.append((seg, "trained_at", str(meta.get("trained_at"))))
    csv_lines = []
    csv_lines.append(",".join(headers))
    for r in rows:
        csv_lines.append(",".join([str(x).replace(",", ";") for x in r]))
    csv_text = "\n".join(csv_lines)
    return Response(csv_text, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=project_summary.csv"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)


