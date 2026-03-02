import os
import json

def project_root() -> str:
    return os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def models_dir() -> str:
    return os.path.join(project_root(), "models")

def load_metrics() -> dict:
    path = os.path.join(models_dir(), "metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    print(json.dumps(load_metrics(), indent=2))
