import pandas as pd
import sqlite3
from datetime import timedelta


def load_and_create_rfm(db_path):
    conn = sqlite3.connect(db_path)

    df = pd.read_sql("SELECT * FROM online_retail_customers", conn)
    conn.close()

    df = df[df['CustomerID'].notna()]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    reference_date = df['InvoiceDate'].max() + timedelta(days=1)

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    return rfm
