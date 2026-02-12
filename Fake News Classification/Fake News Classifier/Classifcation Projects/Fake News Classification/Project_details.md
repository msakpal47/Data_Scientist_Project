# Fake News Classification Project - Interview Guide

## 1. Project Overview (Start to End)
**Goal:** The objective of this project is to build a Machine Learning system capable of automatically classifying news articles as "Factual" or "Fake" based on their textual content.

**Workflow:**
1.  **Data Ingestion:** We read raw data from a SQLite database (`classification.db`), specifically looking for tables containing text and labels.
2.  **Data Cleaning & EDA:** The raw data is cleaned to remove rows with missing values, normalize labels (handling "True"/"False" strings and 0/1 integers), and handle duplicates.
3.  **Model Training:** We implemented a **Naive Bayes Classifier from scratch** (probabilistic model) to predict the likelihood of a news piece being fake or real based on word frequencies.
4.  **Evaluation:** The model is evaluated using Accuracy and Confusion Matrix metrics.
5.  **Deployment (UI):** A user-friendly web interface was built using **Streamlit** to allow users to interact with the model, train it on the fly, and classify new text inputs.
6.  **Monitoring:** Every training run and inference result is logged into a CSV file (`Project_Summary.csv`) to track performance and business impact over time.

## 2. How to Build and Run
### Prerequisites
-   Python 3.x
-   Libraries: `pandas`, `numpy`, `streamlit`, `sqlite3` (built-in), `pickle` (built-in).

### Steps
1.  **Setup Environment:** Ensure all dependencies are installed.
2.  **Prepare Data:** Place the `classification.db` file in the project root.
3.  **Run Application:** Execute the command:
    ```bash
    streamlit run streamlit_app.py
    ```
4.  **Interact:**
    -   Click **Train Model** to load data from the DB, clean it, and train the Naive Bayes model.
    -   Click **Load Saved Model** to use a previously trained model saved in the `artifacts/` folder.
    -   Enter text in the text area and click **Classify** to see if it's Factual or Fake.

## 3. Challenges Faced
During development, several challenges were encountered and resolved:
-   **Data Quality:** The raw data contained mixed types in the label column (strings like "True", "False" and integers 1, 0) and missing text fields.
    -   *Solution:* Implemented a robust `clean_df` function to normalize labels to binary integers (0/1) and drop invalid rows.
-   **Class Imbalance:** Real-world datasets often have uneven distributions of fake vs. real news.
    -   *Solution:* We monitored the Confusion Matrix to ensure the model wasn't just predicting the majority class.
-   **Text Processing:** Raw text contains noise (punctuation, case differences).
    -   *Solution:* Implemented a custom tokenizer using Regex (`re` library) to convert text to lowercase and extract words.
-   **Model Persistence:** Need to save the trained model so we don't have to retrain every time.
    -   *Solution:* Used `pickle` to serialize the Python object and save it to disk.

## 4. Improvements & Key Features
-   **Custom Implementation:** Instead of relying solely on `scikit-learn`, I implemented the **Naive Bayes algorithm from scratch**. This demonstrates a deep understanding of the underlying probability theory (Bayes' Theorem, log-probabilities to prevent underflow).
-   **Interactive UI:** Built a dashboard using **Streamlit**. This makes the project accessible to non-technical stakeholders who can test the model immediately.
-   **Experiment Tracking:** Automated logging to `Project_Summary.csv`. This acts as a "lab notebook," recording the accuracy and confusion matrix for every model version, which is crucial for reproducibility.
-   **Error Handling:** Added checks for missing database files, empty models, or invalid user inputs to prevent crashes.

## 5. Libraries Used & Justification
-   **Streamlit:**
    -   *Why:* It allows for the rapid creation of data apps completely in Python without needing HTML/CSS/JS knowledge. Perfect for demos and prototypes.
-   **Pandas:**
    -   *Why:* The industry standard for data manipulation. Essential for cleaning the dataframe and handling tabular data from the SQL database.
-   **NumPy:**
    -   *Why:* Used for efficient numerical operations, particularly for calculating log-probabilities in the Naive Bayes algorithm.
-   **SQLite3:**
    -   *Why:* A lightweight, serverless database engine. It avoids the overhead of setting up a full SQL server for a standalone project.
-   **Pickle:**
    -   *Why:* Python's native object serialization protocol. It's the standard way to save trained ML models to disk.

## 6. How to Use (User Guide)
1.  **Training:**
    -   On the left side of the app, click "Train Model".
    -   Wait for the success message showing the Accuracy (e.g., ~88%).
2.  **Inference:**
    -   Type or paste a news article into the "Enter news text" box.
    -   Click "Classify".
    -   The system will output **"Factual ✅"** or **"Fake ❌"**.
3.  **Review:**
    -   Scroll down to see the "Latest Metrics" including the Confusion Matrix to understand how well the model is performing.

---
*This document provides a comprehensive summary for interview discussions, highlighting both the technical implementation and the business value of the solution.*
