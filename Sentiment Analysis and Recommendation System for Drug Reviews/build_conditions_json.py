import os
import json
import re
import sqlite3

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
DB_PATH = os.path.join(BASE_DIR, "classification.db")
TABLE = "drug_reviews_sentiment_analysis"
OUT_PATH = os.path.join(BASE_DIR, "static", "conditions.json")

def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(f'SELECT "condition", COUNT(*) FROM {TABLE} WHERE "condition" IS NOT NULL GROUP BY "condition" ORDER BY "condition" ASC')
    rows = cur.fetchall()
    def valid_cond(s: str) -> bool:
        if not s:
            return False
        t = s.strip()
        if "<" in t or ">" in t or "users found this comment helpful" in t:
            return False
        if len(t) < 2:
            return False
        if re.search(r"[A-Za-z]", t) is None:
            return False
        return True
    filtered = [(str(c).strip(), int(n)) for c, n in rows if valid_cond(str(c) if c is not None else "")]
    filtered.sort(key=lambda x: (-x[1], x[0]))
    data = [{"condition": c, "count": n} for c, n in filtered]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"count": len(data), "results": data}, f, ensure_ascii=False, indent=2)
    print("WROTE", OUT_PATH, "COUNT", len(data))

if __name__ == "__main__":
    main()
