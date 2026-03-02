import os, sys, pickle
from app import create_app

app = create_app()
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base, "models", "fuel_model.pkl")
print("flask_app_created", isinstance(app.name, str))
print("model_exists", os.path.exists(model_path))
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            _ = pickle.load(f)
        print("model_load", True)
    except Exception as e:
        print("model_load", False, str(e))
print("templates_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
print("static_dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))

