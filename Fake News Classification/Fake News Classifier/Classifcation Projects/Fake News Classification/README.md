# Fake News Classification — Quick Manual

## Directory
- Project root: `e:\Data_Scientist_Project\Classifcation Projects\Fake News Classification`
- Data: `classification.db` (must contain `text` and `label`)
- App: `streamlit_app.py`
- Script: `fake_news_classifier.py`
- Artifacts: `artifacts\fake_news_model.json` (created after training)
- Logs: `Project_Summary.csv`

## How to Run (Script)
1. Open Terminal in the project folder
2. Train and log metrics:
   ```
   python fake_news_classifier.py --db classification.db --test_size 0.2
   ```
3. Optional one-off prediction:
   ```
   python fake_news_classifier.py --db classification.db --test_size 0.2 --sample_text "Your news text here"
   ```

## How to Run (Streamlit UI)
1. Install Streamlit if needed:
   ```
   python -m pip install streamlit
   ```
2. Start the app from the project folder:
   ```
   python -m streamlit run streamlit_app.py
   ```
3. In the browser:
   - Go to `http://localhost:8501`
   - Tab “Train”: Click “Train Model” (or “Load Saved Model” if trained before)
   - Tab “Classify”: Type your text and click “Classify”

## How it Works
- The database is scanned to find a table with `text` and `label`
- Labels are normalized to 0 (fake) and 1 (factual)
- A Naive Bayes model trains on tokenized words
- Metrics appear in the UI and the model is saved as JSON
- `Project_Summary.csv` is appended with metrics and timestamps

## Troubleshooting
- If “Load a saved model or train a new one first.” appears, train once or click “Load Saved Model”
- Ensure `classification.db` exists and has `text` and `label` columns
- Re-run training after changing data
