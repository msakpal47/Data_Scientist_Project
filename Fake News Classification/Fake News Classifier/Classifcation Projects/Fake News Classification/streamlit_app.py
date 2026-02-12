import os
import pickle
import sqlite3
import streamlit as st
import numpy as np
import pandas as pd
from fake_news_classifier import (
    find_candidate_table,
    load_df,
    clean_df,
    train_and_eval,
    save_artifacts,
    infer,
    summarize_and_update_csv,
    load_model_json,
    NaiveBayes,
)


def get_paths():
    base = os.path.dirname(__file__)
    db_path = os.path.join(base, "classification.db")
    artifacts_dir = os.path.join(base, "artifacts")
    model_path = os.path.join(artifacts_dir, "fake_news_model.json")
    summary_csv = os.path.join(base, "Project_Summary.csv")
    return db_path, artifacts_dir, model_path, summary_csv


def load_model(model_path: str):
    return load_model_json(model_path)


def ensure_model_loaded():
    _, _, model_path, _ = get_paths()
    if "model" not in st.session_state or st.session_state["model"] is None:
        st.session_state["model"] = load_model(model_path)


st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")
st.markdown(
    """
    <style>
    .app-header {
        font-size: 2.2rem;
        font-weight: 800;
        padding: 8px 16px;
        border-radius: 12px;
        background: linear-gradient(90deg, #1f6feb 0%, #6e40c9 50%, #e36209 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .soft-card {padding: 1rem; border-radius: 12px; background: #f7f9fc; border: 1px solid #eef2f7;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="app-header">📰 Fake News Classifier</div>', unsafe_allow_html=True)
st.caption("Type text and classify as factual or fake. Train and track metrics.")

db_path, artifacts_dir, model_path, summary_csv = get_paths()
ensure_model_loaded()
with st.sidebar:
    st.subheader("Quick Start")
    st.write("1) Click Train Model")
    st.write("2) Type text and click Classify")
    st.write("3) Or Load Saved Model if already trained")
    st.write("Database: classification.db")
    st.write("Artifacts: artifacts/")

train_tab, classify_tab = st.tabs(["Train", "Classify"])
with train_tab:
    st.markdown('<div class="soft-card">Train the model from the database and save artifacts.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train Model", use_container_width=True):
            if not os.path.exists(db_path):
                st.error(f"Database not found: {db_path}")
            else:
                conn = sqlite3.connect(db_path)
                try:
                    table = find_candidate_table(conn)
                    if table is None:
                        st.error("No suitable table found.")
                    else:
                        df = load_df(conn, table)
                        df = clean_df(df)
                        model, metrics = train_and_eval(df, test_size=0.2, random_state=42)
                        save_artifacts(model, artifacts_dir)
                        st.session_state["model"] = model
                        st.session_state["metrics"] = metrics
                        summarize_and_update_csv(summary_csv, metrics, None, None)
                        acc = round(metrics.get("accuracy", 0.0), 4)
                        st.success(f"Training complete. Accuracy: {acc}")
                finally:
                    conn.close()
    with col2:
        if st.button("Load Saved Model", use_container_width=True):
            st.session_state["model"] = load_model(model_path)
            if st.session_state["model"] is None:
                st.error("No saved model found.")
            else:
                st.success("Model loaded.")
                conn = sqlite3.connect(db_path)
                try:
                    table = find_candidate_table(conn)
                    if table:
                        df = load_df(conn, table)
                        df = clean_df(df)
                        y_true = df["label"].astype(int).tolist()
                        y_pred = st.session_state["model"].predict(df["text"].astype(str).tolist())
                        arr_t = np.array(y_true)
                        arr_p = np.array(y_pred)
                        tp = int(np.sum((arr_t == 1) & (arr_p == 1)))
                        tn = int(np.sum((arr_t == 0) & (arr_p == 0)))
                        fp = int(np.sum((arr_t == 0) & (arr_p == 1)))
                        fn = int(np.sum((arr_t == 1) & (arr_p == 0)))
                        acc = float((tp + tn) / max(1, len(arr_t)))
                        st.session_state["metrics"] = {
                            "accuracy": acc,
                            "confusion_matrix": [[tn, fp], [fn, tp]],
                        }
                        st.info(f"Evaluation on full dataset. Accuracy: {round(acc,4)}")
                finally:
                    conn.close()
    if "metrics" in st.session_state:
        st.subheader("Latest Metrics")
        m = st.session_state["metrics"]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Accuracy", value=round(m.get("accuracy", 0.0), 4))
        with c2:
            st.write("Confusion Matrix")
            st.write(m.get("confusion_matrix"))

with classify_tab:
    text = st.text_area("Enter news text", height=160)
    if st.button("Classify", use_container_width=True):
        if not text.strip():
            st.warning("Enter some text to classify.")
        elif "model" not in st.session_state or st.session_state["model"] is None:
            st.warning("Load a saved model or train a new one first.")
        else:
            pred = infer(st.session_state["model"], [text])[0]
            if isinstance(st.session_state["model"], NaiveBayes):
                score_factual = st.session_state["model"].predict_proba([text])[0]
            else:
                score_factual = None
            label = "Factual ✅" if int(pred) == 1 else "Fake ❌"
            st.subheader(label)
            if score_factual is not None:
                if int(pred) == 1:
                    st.metric("Score (probability of factual)", value=round(float(score_factual), 4))
                else:
                    st.metric("Score (probability of fake)", value=round(float(1.0 - score_factual), 4))
            metrics = st.session_state.get("metrics", {})
            summarize_and_update_csv(summary_csv, metrics, text, int(pred))
