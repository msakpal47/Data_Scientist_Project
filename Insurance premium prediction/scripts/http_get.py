import sys, urllib.request
if __name__ == "__main__":
    url = sys.argv[1]
    with urllib.request.urlopen(url) as resp:
        print(resp.read().decode("utf-8"))
