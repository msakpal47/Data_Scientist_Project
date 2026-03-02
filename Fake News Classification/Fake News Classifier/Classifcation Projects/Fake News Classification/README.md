# Fake News Classification — Quick Manual

## Directory
- Project root: `e:\Data_Scientist_Project\Classifcation Projects\Fake News Classification\Fake News Classifier\Classifcation Projects\Fake News Classification`
- Data: `classification.db` (table `fake_news_classification` with `text`, `label`)
- Backend: `backend/` (Flask app and ML code)
- Frontend: `templates/`, `static/`
- Artifacts: `backend/model.pkl`, `backend/vectorizer.pkl`
- Logs: `Project_Summary.csv`

## Install
```
python -m pip install -r requirements.txt
```

## Train
```
python backend/train_model.py
```
- Uses 55,031 rows for training, 15,723 for validation, remainder for production holdout.
- Saves `backend/model.pkl` and `backend/vectorizer.pkl`.
- Appends a row to `Project_Summary.csv` with accuracy and sizes.

## Run Server
```
python -m flask --app backend.app run --host 0.0.0.0 --port 5000
```
Open http://127.0.0.1:5000 and classify text.

## CLI Prediction
```
python backend/predict.py --text "Your news text here"
```

## Troubleshooting
- If the UI shows “Model artifacts not found”, run `python backend/train_model.py`.
- Ensure `classification.db` exists and table `fake_news_classification` has columns `text` and `label`.
- Re-run training after changing data.
