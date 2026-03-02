import sys, urllib.request, urllib.parse, urllib.error
url = sys.argv[1]
data = dict(arg.split("=",1) for arg in sys.argv[2:])
body = urllib.parse.urlencode(data).encode("utf-8")
req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/x-www-form-urlencoded"})
try:
    with urllib.request.urlopen(req) as resp:
        print(resp.read().decode("utf-8"))
except urllib.error.HTTPError as e:
    print(e.read().decode("utf-8"))
    raise
