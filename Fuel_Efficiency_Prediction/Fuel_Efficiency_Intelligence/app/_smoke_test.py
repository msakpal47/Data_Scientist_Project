import json
import os
from app import create_app

app = create_app()
client = app.test_client()

payload = {
    "r": 600,
    "m (kg)": 1500.0,
    "Mt": 120.0,
    "Ewltp (g/km)": 110.0,
    "Ft": "petrol",
    "Fm": "E10",
    "ec (cm3)": 1600.0,
    "ep (KW)": 85.0,
    "z (Wh/km)": 180.0,
    "Erwltp (g/km)": 5.0,
    "Electric range (km)": 0.0,
}

resp = client.post("/predict", json=payload)
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(base, "models")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "smoke_test_result.json"), "w") as f:
    f.write(json.dumps({"status": resp.status_code, "body": resp.get_json(silent=True)}, indent=2))
m = client.get("/model-metrics")
fi = client.get("/feature-importance")
docs = client.get("/api-docs")
exp = client.post("/explain", json={"payload": payload})
with open(os.path.join(out_dir, "smoke_model_metrics.json"), "w") as f:
    f.write(json.dumps({"status": m.status_code, "body": m.get_json(silent=True)}, indent=2))
with open(os.path.join(out_dir, "smoke_feature_importance.json"), "w") as f:
    f.write(json.dumps({"status": fi.status_code, "body": fi.get_json(silent=True)}, indent=2))
sim = client.post("/simulate", json={"payload": payload, "ep_delta_pct": 25, "feature_name": "ep (KW)"})
with open(os.path.join(out_dir, "smoke_simulate.json"), "w") as f:
    f.write(json.dumps({"status": sim.status_code, "body": sim.get_json(silent=True)}, indent=2))
with open(os.path.join(out_dir, "smoke_explain.json"), "w") as f:
    f.write(json.dumps({"status": exp.status_code, "body": exp.get_json(silent=True)}, indent=2))
with open(os.path.join(out_dir, "smoke_api_docs.json"), "w") as f:
    f.write(json.dumps({"status": docs.status_code, "body": docs.get_json(silent=True)}, indent=2))
