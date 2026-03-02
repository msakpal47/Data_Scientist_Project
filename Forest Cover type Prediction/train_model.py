import sqlite3
import json
import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        if 'Horizontal_Distance_To_Hydrology' in X.columns and 'Vertical_Distance_To_Hydrology' in X.columns:
            X['Hydrology_Distance'] = (X['Horizontal_Distance_To_Hydrology']**2 + X['Vertical_Distance_To_Hydrology']**2) ** 0.5
        if 'Horizontal_Distance_To_Roadways' in X.columns and 'Horizontal_Distance_To_Fire_Points' in X.columns:
            X['Road_Fire_Distance_Ratio'] = X['Horizontal_Distance_To_Roadways'] / (X['Horizontal_Distance_To_Fire_Points'] + 1e-3)
        if 'Elevation' in X.columns and 'Slope' in X.columns:
            X['Elevation_Slope_Interaction'] = X['Elevation'] * X['Slope']
        if {'Hillshade_9am','Hillshade_Noon','Hillshade_3pm'}.issubset(set(X.columns)):
            X['Mean_Hillshade'] = (X['Hillshade_9am'] + X['Hillshade_Noon'] + X['Hillshade_3pm']) / 3.0
        if 'Aspect' in X.columns:
            rad = np.deg2rad(X['Aspect'] % 360)
            X['Aspect_sin'] = np.sin(rad)
            X['Aspect_cos'] = np.cos(rad)
            X = X.drop(columns=['Aspect'])
        return X

def load_data(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    chosen = None
    for t in tables:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({t})")]
        if 'Elevation' in cols and ('target' in cols or 'Cover_Type' in cols):
            chosen = t
            break
    if chosen is None:
        chosen = tables[0] if tables else None
    if chosen is None:
        conn.close()
        raise RuntimeError("No suitable table found in classification.db")
    df = pd.read_sql_query(f'SELECT * FROM {chosen}', conn)
    conn.close()
    if 'target' not in df.columns and 'Cover_Type' in df.columns:
        df['target'] = df['Cover_Type']
    return df

def get_base_columns():
    cont = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
    wilderness = [f'Wilderness_Area_{i}' for i in range(4)]
    soil = [f'Soil_Type_{i}' for i in range(40)]
    return cont + wilderness + soil

def main():
    df = load_data('classification.db')
    sample_n = min(50000, len(df))
    df = df.sample(sample_n, random_state=42)
    y = df['target'].astype(int)
    X = df.drop(columns=['target'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    models = []
    rf = Pipeline(steps=[('features', FeatureEngineer()), ('clf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    models.append(('RandomForest', rf))
    try:
        from xgboost import XGBClassifier
        xgb = Pipeline(steps=[('features', FeatureEngineer()), ('clf', XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, objective='multi:softmax', num_class=7, n_jobs=-1, random_state=42))])
        models.append(('XGBoost', xgb))
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        lgb = Pipeline(steps=[('features', FeatureEngineer()), ('clf', LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=31, random_state=42))])
        models.append(('LightGBM', lgb))
    except Exception:
        pass
    results = []
    best = None
    for name, m in models:
        if name == 'XGBoost':
            m.fit(X_train, (y_train.values if hasattr(y_train, 'values') else y_train) - 1)
            y_pred_raw = m.predict(X_test)
            y_pred = (y_pred_raw + 1)
        else:
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append({'model': name, 'accuracy': acc, 'macro_f1': f1})
        if best is None or f1 > best['macro_f1']:
            best = {'name': name, 'pipe': m, 'y_pred': y_pred, 'accuracy': acc, 'macro_f1': f1}
    acc = best['accuracy']
    f1 = best['macro_f1']
    models_dir = 'models'
    import os
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best['pipe'], os.path.join(models_dir, 'model.pkl'))
    meta = {
        'accuracy': acc,
        'macro_f1': f1,
        'base_columns': get_base_columns(),
        'class_labels': sorted(df['target'].unique().tolist()),
        'comparison': results
    }
    with open(os.path.join(models_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f)
    try:
        conn_i = sqlite3.connect('insights.db')
        cur = conn_i.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics(
                accuracy REAL,
                macro_f1 REAL,
                model_name TEXT,
                sample_n INTEGER,
                trained_at TEXT
            )
        """)
        cur.execute(
            "INSERT INTO model_metrics(accuracy, macro_f1, model_name, sample_n, trained_at) VALUES(?,?,?,?,?)",
            (acc, f1, best['name'], int(sample_n), datetime.datetime.utcnow().isoformat())
        )
        conn_i.commit()
        conn_i.close()
    except Exception:
        pass
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as _np
        plots_dir = os.path.join('static', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        cm = _np.zeros((len(sorted(df['target'].unique())), len(sorted(df['target'].unique()))), dtype=int)
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, best['y_pred'])
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        plt.close()
        cor = df[get_base_columns()].corr(numeric_only=True)
        plt.figure(figsize=(10,8))
        sns.heatmap(cor, cmap='coolwarm', center=0)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation.png'))
        plt.close()
        plt.figure(figsize=(8,4))
        sns.histplot(df['Elevation'], bins=50)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'elevation_distribution.png'))
        plt.close()
        try:
            fi = None
            clf = best['pipe']['clf']
            X_trf = best['pipe']['features'].transform(X_train)
            cols = list(X_trf.columns)
            if hasattr(clf, 'feature_importances_'):
                fi = clf.feature_importances_
                idx = _np.argsort(fi)[::-1][:15]
                top_cols = [cols[i] for i in idx]
                top_vals = [fi[i] for i in idx]
                plt.figure(figsize=(9,6))
                sns.barplot(x=top_vals, y=top_cols, orient='h')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
                plt.close()
        except Exception:
            pass
        try:
            conn_i = sqlite3.connect('insights.db')
            cur = conn_i.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_reports(
                    model_name TEXT,
                    report TEXT,
                    trained_at TEXT
                )
            """)
            rep = classification_report(y_test, best['y_pred'])
            cur.execute("INSERT INTO model_reports(model_name, report, trained_at) VALUES(?,?,?)", (best['name'], rep, datetime.datetime.utcnow().isoformat()))
            conn_i.commit()
            conn_i.close()
        except Exception:
            pass
    except Exception:
        pass
    print({'accuracy': acc, 'macro_f1': f1, 'model': best['name'], 'comparison': results})

if __name__ == '__main__':
    main()
