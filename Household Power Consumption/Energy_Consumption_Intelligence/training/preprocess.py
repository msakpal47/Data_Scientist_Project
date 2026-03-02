import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_features_target(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def fit_transform_scaler(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = pd.DataFrame(Xs, columns=X.columns, index=X.index)
    return Xs, scaler


def transform_scaler(X, scaler):
    Xs = scaler.transform(X)
    Xs = pd.DataFrame(Xs, columns=X.columns, index=X.index)
    return Xs

