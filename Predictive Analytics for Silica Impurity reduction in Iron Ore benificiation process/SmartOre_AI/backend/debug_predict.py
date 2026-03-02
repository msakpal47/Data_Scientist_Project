import sys, os, json, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import predict as p

if __name__ == "__main__":
    ok = p.init_cache()
    print("init_ok:", ok)
    print("model_is_none:", p._CACHE["model"] is None)
    payload = {
        "% Iron Feed": 65,
        "% Silica Feed": 4.2,
        "Starch Flow": 110,
        "Amina Flow": 22,
        "Ore Pulp Flow": 780,
        "Ore Pulp pH": 9.6,
        "Ore Pulp Density": 1.24,
        "Avg Air Flow": 95,
    }
    try:
        t0 = time.time()
        res = p.predict_payload(payload)
        res['elapsed_ms'] = int((time.time() - t0) * 1000)
        print("predict_ok:", True)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print("predict_ok:", False, "err:", e)
