import os
import json
import joblib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rfm_cluster.pkl")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/segments":

            if not os.path.exists(MODEL_PATH):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"Model not trained"}')
                return

            data = joblib.load(MODEL_PATH)
            rfm = data["rfm_data"]

            result = rfm.to_dict(orient="records")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        else:
            super().do_GET()


def main():
    os.chdir(BASE_DIR)
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    print("Server running at http://localhost:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
