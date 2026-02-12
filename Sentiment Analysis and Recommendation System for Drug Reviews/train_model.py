import os
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
DB_PATH = os.path.join(BASE_DIR, "classification.db")
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.joblib")

def load_data():
    con = sqlite3.connect(DB_PATH)
    query = "SELECT uniqueID, drugName, condition, review, rating, usefulCount FROM drug_reviews_sentiment_analysis"
    df = pd.read_sql_query(query, con)
    con.close()
    return df

def label_sentiment(rating: int) -> int:
    if rating >= 7:
        return 1
    if rating <= 4:
        return 0
    return -1

def main():
    df = load_data()
    df["label"] = df["rating"].astype(int).apply(label_sentiment)
    df = df[df["label"] != -1].copy()
    df.dropna(subset=["review"], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"].values, df["label"].values, test_size=0.2, random_state=42, stratify=df["label"].values
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=1)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(pipeline, MODEL_PATH)
    print("MODEL_SAVED", MODEL_PATH)

if __name__ == "__main__":
    main()
