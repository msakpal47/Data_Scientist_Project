import sqlite3
import os

DB_PATH = os.path.join(
    r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews",
    "classification.db",
)

def main():
    if not os.path.exists(DB_PATH):
        print("DB_NOT_FOUND", DB_PATH)
        return

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cur.fetchall()]
    print("TABLES", tables)

    target_table = "drug_reviews_sentiment_analysis" if "drug_reviews_sentiment_analysis" in tables else (tables[0] if tables else None)
    print("USING_TABLE", target_table)
    if not target_table:
        return

    cur.execute(f"PRAGMA table_info({target_table})")
    cols = cur.fetchall()
    print("COLUMNS", cols)

    cur.execute(f"SELECT COUNT(*) FROM {target_table}")
    total = cur.fetchone()[0]
    print("ROW_COUNT", total)

    cur.execute(f"SELECT * FROM {target_table} LIMIT 5")
    rows = cur.fetchall()
    print("SAMPLE_ROWS", rows)

    con.close()

if __name__ == "__main__":
    main()
