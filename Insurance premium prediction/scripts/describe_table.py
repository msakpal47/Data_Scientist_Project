import sqlite3, sys, json, os
BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(BASE), "regression.db")
table = sys.argv[1]
con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute(f"PRAGMA table_info('{table}')")
cols = [r[1] for r in cur.fetchall()]
print(json.dumps(cols))
con.close()
