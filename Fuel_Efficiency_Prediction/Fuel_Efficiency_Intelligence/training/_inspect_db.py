import os
import sqlite3
import json
import sys


def main():
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Regression.db"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "regression.db"),
        os.path.join(os.getcwd(), "regression.db"),
    ]
    db_path = None
    for c in candidates:
        if os.path.exists(c):
            db_path = os.path.abspath(c)
            break
    if not db_path:
        print(json.dumps({"error": "db_not_found", "tried": candidates}))
        return
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    out = {"db_path": db_path, "tables": []}
    for t in tables:
        cur.execute(f"PRAGMA table_info('{t}')")
        cols = [{"name": c[1], "type": c[2]} for c in cur.fetchall()]
        cur.execute(f"SELECT COUNT(1) FROM '{t}'")
        n = cur.fetchone()[0]
        out["tables"].append({"name": t, "rows": n, "columns": cols})
    con.close()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

