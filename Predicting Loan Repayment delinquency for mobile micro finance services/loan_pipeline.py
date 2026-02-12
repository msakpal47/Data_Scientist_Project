import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TABLE_NAME = "Telecom_microservices_loan"
TARGET_COLUMN = "label"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_rows_from_sqlite(
    db_path: str,
    table_name: str,
    limit: int | None,
    offset: int,
) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    if limit is None:
        query = f'SELECT * FROM "{table_name}" ORDER BY rowid'
        params: tuple[object, ...] = ()
    else:
        query = f'SELECT * FROM "{table_name}" ORDER BY rowid LIMIT ? OFFSET ?'
        params = (int(limit), int(offset))

    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(query, con, params=params)


def iter_rows_from_sqlite(
    db_path: str,
    table_name: str,
    chunk_size: int,
    offset: int,
) -> Iterator[pd.DataFrame]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    query = f'SELECT * FROM "{table_name}" ORDER BY rowid LIMIT -1 OFFSET ?'
    with sqlite3.connect(db_path) as con:
        for chunk in pd.read_sql_query(query, con, params=(int(offset),), chunksize=int(chunk_size)):
            if chunk is None or chunk.empty:
                break
            yield chunk


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_identifier: bool = True):
        self.drop_identifier = drop_identifier

    def fit(self, X, y=None):
        X_df = self._ensure_dataframe(X)
        if TARGET_COLUMN in X_df.columns:
            X_df = X_df.drop(columns=[TARGET_COLUMN])

        if self.drop_identifier and "msisdn" in X_df.columns:
            X_df = X_df.drop(columns=["msisdn"])

        transformed = self._transform_raw(X_df)
        self.input_columns_ = list(X_df.columns)
        self.output_columns_ = list(transformed.columns)
        return self

    def transform(self, X):
        if not hasattr(self, "input_columns_") or not hasattr(self, "output_columns_"):
            raise RuntimeError("FeatureEngineer must be fit before transform.")

        X_df = self._ensure_dataframe(X)
        if TARGET_COLUMN in X_df.columns:
            X_df = X_df.drop(columns=[TARGET_COLUMN])

        if self.drop_identifier and "msisdn" in X_df.columns:
            X_df = X_df.drop(columns=["msisdn"])

        for col in self.input_columns_:
            if col not in X_df.columns:
                X_df[col] = np.nan
        extra_cols = [c for c in X_df.columns if c not in self.input_columns_]
        if extra_cols:
            X_df = X_df.drop(columns=extra_cols)

        transformed = self._transform_raw(X_df)
        for col in self.output_columns_:
            if col not in transformed.columns:
                transformed[col] = np.nan
        transformed = transformed[self.output_columns_]
        return transformed

    def _ensure_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _safe_div(self, num: pd.Series, den: pd.Series) -> pd.Series:
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        return np.where((den.isna()) | (den == 0), np.nan, num / den)

    def _transform_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        if "pcircle" in out.columns:
            out["pcircle"] = out["pcircle"].astype("string").fillna(pd.NA).astype(object)

        for col in out.columns:
            if col == "pcircle":
                continue
            out[col] = pd.to_numeric(out[col], errors="coerce")

        if "pdate" in out.columns:
            parsed = pd.to_datetime(out["pdate"], errors="coerce")
            if parsed.notna().any():
                out["pdate_year"] = parsed.dt.year
                out["pdate_month"] = parsed.dt.month
                out["pdate_dayofweek"] = parsed.dt.dayofweek
            out = out.drop(columns=["pdate"])

        if "last_rech_date_ma" in out.columns and "last_rech_date_da" in out.columns:
            month = pd.to_numeric(out["last_rech_date_ma"], errors="coerce")
            day = pd.to_numeric(out["last_rech_date_da"], errors="coerce")
            year = pd.Series(np.full(len(out), 2000), index=out.index)
            parsed = pd.to_datetime(
                {"year": year, "month": month, "day": day},
                errors="coerce",
            )
            if parsed.notna().any():
                out["last_rech_dayofyear"] = parsed.dt.dayofyear

        if {"payback30", "cnt_loans30"}.issubset(out.columns):
            out["payback_rate30"] = self._safe_div(out["payback30"], out["cnt_loans30"])
        if {"payback90", "cnt_loans90"}.issubset(out.columns):
            out["payback_rate90"] = self._safe_div(out["payback90"], out["cnt_loans90"])

        if {"amnt_loans30", "cnt_loans30"}.issubset(out.columns):
            out["avg_loan_amt30"] = self._safe_div(out["amnt_loans30"], out["cnt_loans30"])
        if {"amnt_loans90", "cnt_loans90"}.issubset(out.columns):
            out["avg_loan_amt90"] = self._safe_div(out["amnt_loans90"], out["cnt_loans90"])

        if {"sumamnt_ma_rech30", "cnt_ma_rech30"}.issubset(out.columns):
            out["avg_rech_amt30"] = self._safe_div(out["sumamnt_ma_rech30"], out["cnt_ma_rech30"])
        if {"sumamnt_ma_rech90", "cnt_ma_rech90"}.issubset(out.columns):
            out["avg_rech_amt90"] = self._safe_div(out["sumamnt_ma_rech90"], out["cnt_ma_rech90"])

        return out


def build_pipeline(random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                make_column_selector(dtype_include=object),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = SGDClassifier(
        loss="log_loss",
        alpha=0.0001,
        penalty="l2",
        max_iter=2000,
        tol=1e-3,
        random_state=random_state,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("features", FeatureEngineer(drop_identifier=True)),
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )


@dataclass(frozen=True)
class TrainResult:
    metrics: dict
    model_path: str


def evaluate_binary_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    metrics: dict[str, object] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    return metrics


def save_artifacts(
    model: Pipeline,
    model_path: str,
    metadata_path: str,
    metrics: dict,
    extra_metadata: dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    dump(model, model_path)

    metadata = {
        "trained_at": utc_now_iso(),
        "table_name": TABLE_NAME,
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_model(model_path: str) -> Pipeline:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load(model_path)

