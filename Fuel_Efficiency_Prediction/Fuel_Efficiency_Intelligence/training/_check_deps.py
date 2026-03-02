ok = True
mods = ["numpy", "pandas", "sklearn", "xgboost"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        ok = False
        missing.append(m)
if ok:
    print("ok")
else:
    print("missing:" + ",".join(missing))

