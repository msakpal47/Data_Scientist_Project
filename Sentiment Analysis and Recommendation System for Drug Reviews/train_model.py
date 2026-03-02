import os
import sqlite3
import html
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pickle

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
DB_PATH = os.path.join(BASE_DIR, "classification.db")
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.joblib")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_JOBLIB = os.path.join(MODELS_DIR, "sentiment_model.joblib")
MODEL_PKL = os.path.join(MODELS_DIR, "sentiment_model.pkl")

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
    def clean_text(t: str) -> str:
        s = html.unescape(str(t))
        s = re.sub(r"\d+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    df["review_clean"] = df["review"].apply(clean_text)
    df = df.sort_values("uniqueID").reset_index(drop=True)
    n_train = 112908
    n_val = 32259
    n_prod = 16130
    total = len(df)
    if total >= (n_train + n_val + n_prod):
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train : n_train + n_val]
        prod_df = df.iloc[n_train + n_val : n_train + n_val + n_prod]
    else:
        split1 = int(total * 0.7)
        split2 = int(total * 0.9)
        train_df = df.iloc[:split1]
        val_df = df.iloc[split1:split2]
        prod_df = df.iloc[split2:]
    X_train = train_df["review_clean"].values
    y_train = train_df["label"].values
    X_val = val_df["review_clean"].values
    y_val = val_df["label"].values
    X_prod = prod_df["review_clean"].values
    y_prod = prod_df["label"].values

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    print("VALIDATION_REPORT")
    print(classification_report(y_val, y_val_pred, digits=4))
    print("VALIDATION_CONFUSION_MATRIX")
    print(confusion_matrix(y_val, y_val_pred))
    y_prod_pred = pipeline.predict(X_prod)
    print("PRODUCTION_REPORT")
    print(classification_report(y_prod, y_prod_pred, digits=4))
    print("PRODUCTION_CONFUSION_MATRIX")
    print(confusion_matrix(y_prod, y_prod_pred))
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        feats = tfidf.get_feature_names_out()
        weights = clf.coef_[0]
        top_pos_idx = weights.argsort()[-20:][::-1]
        top_neg_idx = weights.argsort()[:20]
        print("TOP_POSITIVE_FEATURES")
        print([feats[i] for i in top_pos_idx])
        print("TOP_NEGATIVE_FEATURES")
        print([feats[i] for i in top_neg_idx])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(pipeline, MODEL_JOBLIB)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(pipeline, f)
    print("MODEL_SAVED", MODEL_PATH)
    print("MODEL_SAVED", MODEL_JOBLIB)
    print("MODEL_SAVED", MODEL_PKL)

if __name__ == "__main__":
    main()
