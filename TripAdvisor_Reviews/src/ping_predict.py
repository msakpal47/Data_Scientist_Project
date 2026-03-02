import json
import urllib.request
import urllib.error

url = "http://127.0.0.1:3000/predict"
data = {"review": "great staff and friendly service"}
req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8")
        print(body)
except urllib.error.HTTPError as e:
    print("HTTPError", e.code, e.read().decode("utf-8", errors="ignore"))
except Exception as e:
    print("Error", str(e))

