import sqlite3
import json
import os
import sys

def main(db_path: str) -> None:
    print("DB path:", db_path)
    print("Exists:", os.path.exists(db_path))
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print("Tables:", tables)
    info = {}
    for t in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            row_count = int(cur.fetchone()[0])
        except Exception:
            row_count = None
        cur.execute(f"PRAGMA table_info({t})")
        cols = cur.fetchall()
        info[t] = {
            "rows": row_count,
            "columns": [(c[1], c[2]) for c in cols],
        }
    print("Schema:")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "classification.db",
        )
        db_path = os.path.abspath(db_path)
    main(db_path)
