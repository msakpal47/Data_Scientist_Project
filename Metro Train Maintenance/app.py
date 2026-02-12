import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import streamlit as st

DB_PATH = os.path.join(os.path.dirname(__file__), "classification.db")
TABLE = "fault_detection_manufacturing"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fault_model.pkl")

FEATURES = [
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE",
    "ROTATIONSPEED",
    "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
    "TTF_FlowCool Pressure Dropped Below Limit",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "TTF_Flowcool leak",
]
TARGET = "fault_occurred"

@st.cache_data
def load_data():
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE}", con)
    con.close()
    # Keep only relevant columns
    cols = FEATURES + [TARGET, "stage", "recipe", "recipe_step"]
    df = df[[c for c in cols if c in df.columns]].copy()
    df.dropna(subset=[TARGET], inplace=True)
    df[TARGET] = df[TARGET].astype(int)
    return df

def build_pipeline(random_state=42):
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    return pipe

def train_and_eval(df):
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": pipe, "features": FEATURES}, MODEL_PATH)
    return acc, report, pipe, (X_test, y_test)

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def predict(model_bundle, input_dict):
    model = model_bundle["model"]
    feats = model_bundle["features"]
    x = np.array([[float(input_dict[f]) for f in feats]])
    pred = model.predict(x)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(x)
        proba = float(probas[0][1])
    return int(pred), proba

def render_metrics(report):
    st.subheader("Classification Metrics")
    st.write(f"Accuracy: {report['accuracy']:.3f}")
    st.write(
        pd.DataFrame(report).transpose()[["precision", "recall", "f1-score", "support"]]
    )

def render_predict_form(df, model_bundle):
    st.subheader("Predict Fault Occurrence")
    st.caption("Adjust sensor values to simulate current operating conditions.")
    ranges = {}
    for f in FEATURES:
        s = df[f].astype(float)
        ranges[f] = (float(np.nanmin(s)), float(np.nanmax(s)), float(np.nanmedian(s)))
    cols = st.columns(2)
    inputs = {}
    for i, f in enumerate(FEATURES):
        col = cols[i % 2]
        mn, mx, md = ranges[f]
        inputs[f] = col.slider(f, value=md, min_value=mn, max_value=mx)
    if st.button("Run Prediction"):
        pred, proba = predict(model_bundle, inputs)
        st.success(f"Predicted fault_occurred: {pred}")
        if proba is not None:
            st.info(f"Estimated fault probability: {proba:.3f}")

def main():
    st.set_page_config(page_title="Metro Train Maintenance", layout="wide")
    st.title("Metro Train Maintenance: Fault Detection")
    st.caption("Trains a model from the manufacturing sensor dataset and provides interactive fault predictions.")

    df = load_data()
    with st.sidebar:
        st.header("Controls")
        if st.button("Train / Retrain Model"):
            acc, report, pipe, test = train_and_eval(df)
            st.session_state["last_report"] = report
            st.session_state["model_bundle"] = {"model": pipe, "features": FEATURES}
            st.success(f"Training complete. Accuracy: {acc:.3f}")
        if st.button("Load Saved Model"):
            bundle = load_model()
            if bundle is not None:
                st.session_state["model_bundle"] = bundle
                st.success("Loaded saved model.")
            else:
                st.error("No saved model found. Train first.")

    bundle = st.session_state.get("model_bundle") or load_model()
    report = st.session_state.get("last_report")
    if bundle is None:
        st.warning("No model loaded. Train or load a saved model from the sidebar.")
    else:
        if report:
            render_metrics(report)
        render_predict_form(df, bundle)

if __name__ == "__main__":
    main()
