import os
import joblib
import pickle

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
MODELS_DIR = os.path.join(BASE_DIR, "models")
SRC_JOBLIB = os.path.join(BASE_DIR, "sentiment_model.joblib")
DST_JOBLIB = os.path.join(MODELS_DIR, "sentiment_model.joblib")
DST_PKL = os.path.join(MODELS_DIR, "sentiment_model.pkl")

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    model = joblib.load(SRC_JOBLIB)
    joblib.dump(model, DST_JOBLIB)
    with open(DST_PKL, "wb") as f:
        pickle.dump(model, f)
    print("SAVED_TO", MODELS_DIR)

if __name__ == "__main__":
    main()
