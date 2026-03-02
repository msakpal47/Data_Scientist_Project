import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.data import find


def ensure_nltk_data():
    try:
        find("corpora/stopwords")
    except LookupError:
        pass
    try:
        find("corpora/wordnet")
    except LookupError:
        pass


ensure_nltk_data()
NEGATION_WORDS = {"not", "no", "never", "nor", "n't", "hardly", "barely", "scarcely"}
try:
    _stop_words = set(stopwords.words("english"))
    _stop_words = {w for w in _stop_words if w not in NEGATION_WORDS}
except Exception:
    _stop_words = {
        "the", "and", "a", "an", "is", "it", "to", "in", "of", "that", "this", "for",
        "on", "with", "as", "at", "by", "be", "are", "was", "were", "or", "from",
        "but", "so", "if", "then", "than", "too", "very", "can", "could",
    }
_lemmatizer = WordNetLemmatizer()

def _safe_lemmatize(word: str) -> str:
    try:
        return _lemmatizer.lemmatize(word)
    except Exception:
        return word


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    words = [_safe_lemmatize(w) for w in words if w not in _stop_words]
    return " ".join(words)

