import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "oldbalanceOrg" in x.columns and "newbalanceOrig" in x.columns:
        x["balance_diff_orig"] = x["oldbalanceOrg"] - x["newbalanceOrig"]
        x["is_zero_balance_after"] = (x["newbalanceOrig"] == 0).astype(int)
    if "newbalanceDest" in x.columns and "oldbalanceDest" in x.columns:
        x["balance_diff_dest"] = x["newbalanceDest"] - x["oldbalanceDest"]
    if "amount" in x.columns:
        x["high_amount_flag"] = (x["amount"] > 200000).astype(int)
    return x
