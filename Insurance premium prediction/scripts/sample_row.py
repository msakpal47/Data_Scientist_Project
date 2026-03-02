import sqlite3, sys, json, os
BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(os.path.dirname(BASE), "regression.db")
table = sys.argv[1]
con = sqlite3.connect(DB)
df = None
try:
    import pandas as pd
    df = pd.read_sql_query(f"SELECT * FROM [{table}] LIMIT 1", con)
    print(df.to_json(orient="records"))
finally:
    con.close()
