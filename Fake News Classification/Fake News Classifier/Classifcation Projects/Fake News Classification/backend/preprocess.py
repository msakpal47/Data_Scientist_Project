import re
from typing import Optional


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_label(val) -> int:
    if val is None:
        return -1
    v = str(val).strip().lower()
    if v in {"true", "1", "yes", "real", "factual"}:
        return 1
    if v in {"false", "0", "no", "fake", "misleading"}:
        return 0
    try:
        n = int(v)
        return 1 if n == 1 else 0
    except Exception:
        return -1
