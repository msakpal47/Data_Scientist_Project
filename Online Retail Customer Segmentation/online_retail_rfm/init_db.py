import os
import random
import sqlite3
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "clustering.db")


def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS online_retail_customers (
            InvoiceNo TEXT,
            StockCode TEXT,
            Description TEXT,
            Quantity INTEGER,
            InvoiceDate TEXT,
            UnitPrice REAL,
            CustomerID INTEGER,
            Country TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def populate_synthetic(n_customers=300, min_invoices=3, max_invoices=40):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("DELETE FROM online_retail_customers")
    conn.commit()

    start_date = datetime(2010, 12, 1)
    end_date = datetime(2011, 12, 9)
    date_range_days = (end_date - start_date).days

    invoice_counter = 100000
    countries = ["United Kingdom", "Germany", "France", "Spain", "Netherlands", "Belgium"]

    for cust_idx in range(n_customers):
        customer_id = 10000 + cust_idx
        invoices = random.randint(min_invoices, max_invoices)
        for _ in range(invoices):
            is_cancel = random.random() < 0.05
            invoice_no = f"{'C' if is_cancel else ''}{invoice_counter}"
            invoice_counter += 1

            date_offset = random.randint(0, date_range_days)
            inv_date = start_date + timedelta(days=date_offset)

            line_items = random.randint(1, 5)
            for _li in range(line_items):
                qty = random.randint(1, 10)
                unit_price = round(random.uniform(2.0, 150.0), 2)
                stock_code = f"STK{random.randint(100, 999)}"
                desc = f"Product {stock_code}"
                country = random.choice(countries)
                cur.execute(
                    """
                    INSERT INTO online_retail_customers
                    (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        invoice_no,
                        stock_code,
                        desc,
                        qty,
                        inv_date.strftime("%Y-%m-%d %H:%M:%S"),
                        unit_price,
                        customer_id,
                        country,
                    ),
                )

    conn.commit()
    conn.close()


def main():
    ensure_db()
    populate_synthetic()
    print(f"Database initialized at {DB_PATH} with table 'online_retail_customers'")


if __name__ == "__main__":
    main()
