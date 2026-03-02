import json, sys, urllib.request
url = sys.argv[1]
payload = json.loads(sys.argv[2])
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req) as resp:
    print(resp.read().decode("utf-8"))
