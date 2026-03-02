import os
import json
import joblib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ecommerce_cluster.pkl")
MODEL_CACHE = None


def get_model():
    global MODEL_CACHE
    if MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            return None
        MODEL_CACHE = joblib.load(MODEL_PATH)
    return MODEL_CACHE


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html", "/templates"):
            self.path = "/templates/index.html"
            return super().do_GET()

        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            m = get_model()
            seg = m.get("segments") if m is not None else None
            self.send_json({"ok": True, "model_loaded": m is not None, "segments": seg})
            return

        if parsed.path == "/api/predict":
            qs = parse_qs(parsed.query)
            try:
                limit = int(qs.get("limit", [200])[0])
                offset = int(qs.get("offset", [0])[0])
            except ValueError:
                limit = 200
                offset = 0
            cluster_filter = qs.get("cluster", [None])[0]
            user_id_filter = qs.get("user_id", [None])[0]
            if user_id_filter is not None:
                try:
                    user_id_filter = int(user_id_filter)
                except ValueError:
                    user_id_filter = None

            model = get_model()
            if model is None:
                self.send_json({"error": "model_not_trained"})
                return

            clusters = model["model"].labels_
            users = model["users"]
            segments = model.get("segments", {})

            n = min(len(clusters), len(users))
            clusters = clusters[:n]
            users = users[:n]

            indices = list(range(n))
            if cluster_filter is not None:
                try:
                    cf = int(cluster_filter)
                    indices = [i for i in indices if clusters[i] == cf]
                except ValueError:
                    pass
            if user_id_filter is not None:
                indices = [i for i in indices if users[i] == user_id_filter]

            total = len(indices)
            end = min(offset + limit, total)
            sel = indices[offset:end]
            users_slice = [users[i] for i in sel]
            clusters_slice = [int(clusters[i]) for i in sel]

            max_label = int(max(clusters)) if n > 0 else -1
            size = max(5, max_label + 1)
            counts = [0] * size
            for c in clusters.tolist():
                ci = int(c)
                if 0 <= ci < size:
                    counts[ci] += 1
            counts_filtered = [0] * size
            for i in indices:
                ci = int(clusters[i])
                if 0 <= ci < size:
                    counts_filtered[ci] += 1
            if len(counts) < 5:
                counts = counts + [0] * (5 - len(counts))
            if len(counts_filtered) < 5:
                counts_filtered = counts_filtered + [0] * (5 - len(counts_filtered))

            self.send_json({
                "users": users_slice,
                "clusters": clusters_slice,
                "total": total,
                "offset": offset,
                "limit": limit,
                "counts": counts,
                "counts_filtered": counts_filtered,
                "active_cluster": int(cluster_filter) if cluster_filter is not None and str(cluster_filter).isdigit() else None,
                "segments": segments,
            })
            return

        if parsed.path == "/api/cluster_counts":
            model = get_model()
            if model is None:
                self.send_json({"error": "model_not_trained"})
                return
            clusters = model["model"].labels_
            n = len(clusters)
            max_label = int(max(clusters)) if n > 0 else -1
            size = max(5, max_label + 1)
            counts = [0] * size
            for c in clusters.tolist():
                ci = int(c)
                if 0 <= ci < size:
                    counts[ci] += 1
            if len(counts) < 5:
                counts = counts + [0] * (5 - len(counts))
            m = get_model()
            seg = m.get("segments") if m is not None else {}
            self.send_json({"counts": counts, "total": len(clusters), "segments": seg})
            return

        if parsed.path == "/api/segments":
            model = get_model()
            if model is None:
                self.send_json({"error": "model_not_trained"})
                return
            self.send_json({"segments": model.get("segments", {})})
            return

        return super().do_GET()

    def send_json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)


def run():
    os.chdir(BASE_DIR)
    for host in ("0.0.0.0", "127.0.0.1"):
        for port in (8000, 8080, 5000):
            try:
                server = ThreadingHTTPServer((host, port), Handler)
                print(f"Serving on http://{host}:{port}/")
                server.serve_forever()
                return
            except OSError:
                continue
    raise RuntimeError("Unable to bind server to any port")


if __name__ == "__main__":
    run()
