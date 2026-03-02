import os
import sqlite3

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
DB_PATH = os.path.join(BASE_DIR, "classification.db")
TABLE_NAME = "drug_reviews_sentiment_analysis"

def main():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(f'SELECT DISTINCT "condition" FROM {TABLE_NAME} WHERE "condition" IS NOT NULL ORDER BY "condition" ASC LIMIT 50')
    rows = cur.fetchall()
    print("COUNT_50", len(rows))
    for r in rows[:20]:
        print("COND", r[0])
    con.close()

if __name__ == "__main__":
    main()
