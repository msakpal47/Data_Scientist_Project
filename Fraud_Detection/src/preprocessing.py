import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame, expected_cols: list[str] | None = None) -> pd.DataFrame:
    x = df.copy()
    cols = x.columns.tolist()
    if "oldbalanceOrg" in cols and "newbalanceOrig" in cols:
        x["balance_diff_orig"] = x["oldbalanceOrg"] - x["newbalanceOrig"]
        x["is_zero_balance_after"] = (x["newbalanceOrig"] == 0).astype(int)
    if "newbalanceDest" in cols and "oldbalanceDest" in cols:
        x["balance_diff_dest"] = x["newbalanceDest"] - x["oldbalanceDest"]
    if "amount" in cols:
        x["high_amount_flag"] = (x["amount"] > 200000).astype(int)
    drop_cols = []
    for c in ["nameOrig", "nameDest"]:
        if c in x.columns:
            drop_cols.append(c)
    if drop_cols:
        x = x.drop(columns=drop_cols)
    if "type" in x.columns:
        x = pd.get_dummies(x, columns=["type"], prefix="type")
    x = x.replace([np.inf, -np.inf], np.nan)
    numeric_x = x.select_dtypes(include=["number"]).copy()
    if expected_cols is not None:
        for c in expected_cols:
            if c not in numeric_x.columns:
                numeric_x[c] = 0.0
        extra = [c for c in numeric_x.columns if c not in expected_cols]
        if extra:
            numeric_x = numeric_x.drop(columns=extra)
        numeric_x = numeric_x.reindex(columns=expected_cols)
    return numeric_x
