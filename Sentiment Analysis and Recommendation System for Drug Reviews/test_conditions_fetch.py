import urllib.request

def get(path):
    url = f"http://127.0.0.1:8000{path}"
    try:
        with urllib.request.urlopen(url) as resp:
            body = resp.read().decode()
            print(f"OK {path} STATUS", resp.status, "LEN", len(body))
            print(body[:200])
    except Exception as e:
        print(f"ERR {path}", e)

get("/")
get("/metrics_data")
get("/conditions")
get("/__routes")
