import math
import numpy as np
import re


def parse_helpfulness(val):
    if val is None:
        return 0.0
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(val))
    if not m:
        return 0.0
    a, b = int(m.group(1)), int(m.group(2))
    return a / b if b > 0 else 0.0


def build_features(rows, vectorizer, scaler, svd, fit=False):
    texts = []
    nums = []

    for r in rows:
        title = str(r.get("Title", "") or "")
        summary = str(r.get("review/summary", "") or "")
        text = str(r.get("review/text", "") or "")
        full_text = f"{title} {summary} {text}".strip()

        score = float(r.get("review/score", 0) or 0)
        helpful = parse_helpfulness(r.get("review/helpfulness"))
        price = r.get("Price")

        try:
            price = float(price)
            price_log = math.log(price + 1)
        except:
            price_log = 0.0

        length = len(full_text.split())

        texts.append(full_text)
        nums.append([score, helpful, price_log, length])

    if fit:
        X_text = vectorizer.fit_transform(texts)
        X_num = scaler.fit_transform(nums)
        X_text_red = svd.fit_transform(X_text)
    else:
        X_text = vectorizer.transform(texts)
        X_num = scaler.transform(nums)
        X_text_red = svd.transform(X_text)

    return np.hstack([X_text_red, X_num])
