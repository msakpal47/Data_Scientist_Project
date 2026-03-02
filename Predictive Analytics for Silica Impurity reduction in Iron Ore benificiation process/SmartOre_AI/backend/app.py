import os
import sys
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import predict as predictor


def create_app():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(os.path.dirname(backend_dir), "frontend")
    templates_dir = os.path.join(frontend_dir, "templates")
    static_dir = os.path.join(frontend_dir, "static")
    app = Flask(__name__, template_folder=templates_dir, static_folder=static_dir)
    CORS(app)
    return app


app = create_app()
print("Loading model...")
loaded = predictor.init_cache()
print("Model loaded:", loaded)
if not loaded:
    try:
        from train_model import train_and_save
        print("Training model (cold start)...")
        train_and_save()
        loaded = predictor.init_cache()
        print("Model loaded after training:", loaded)
    except Exception as e:
        print("Auto-train failed:", e)
try:
    predictor.warmup()
except Exception as e:
    print("Warmup failed:", e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(predictor.status())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    print("api_predict: start")
    try:
        t0 = time.time()
        payload = request.get_json(force=True)
        print("api_predict: payload keys:", list(payload.keys()) if isinstance(payload, dict) else type(payload))
        res = predictor.predict_payload(payload)
        res["elapsed_ms"] = int((time.time() - t0) * 1000)
        print("api_predict: ok")
        return jsonify(res)
    except Exception as e:
        print("api_predict: error:", e)
        import traceback
        traceback.print_exc()
        try:
            fb = predictor.fallback_predict(payload if 'payload' in locals() else {})
        except Exception:
            fb = {
                "silica_concentrate": 2.0,
                "risk": "Medium",
                "recommendations": ["Keep current operating conditions; monitor trends"],
                "fallback": True,
            }
        fb["elapsed_ms"] = 0
        return jsonify(fb)

@app.route("/api/ping")
def api_ping():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
