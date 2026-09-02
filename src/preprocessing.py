"""
Text preprocessing utilities for the Emotion Detection pipeline.

Each function is small, pure, and independently testable. `clean_text`
chains them together in the exact order used during model training, so
training and inference always apply identical preprocessing.
"""

import string

_STOPWORDS = None


def get_stopwords() -> set:
    """Lazily load and cache the NLTK English stopword set."""
    global _STOPWORDS
    if _STOPWORDS is None:
        import nltk
        from nltk.corpus import stopwords

        try:
            _STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _STOPWORDS = set(stopwords.words("english"))
    return _STOPWORDS


def lowercase(text: str) -> str:
    """Lowercase all characters."""
    return text.lower()


def remove_punctuation(text: str) -> str:
    """Strip standard punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove digit characters."""
    return "".join(ch for ch in text if not ch.isdigit())


def remove_emojis(text: str) -> str:
    """Remove non-ASCII characters (emojis, accented symbols, etc.)."""
    return "".join(ch for ch in text if ch.isascii())


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords."""
    stop_words = get_stopwords()
    return " ".join(w for w in text.split() if w not in stop_words)


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline, applied in this order:
    lowercase -> remove punctuation -> remove numbers -> remove emojis
    -> remove stopwords.

    Safe against non-string / NaN input (returns empty string).
    """
    if not isinstance(text, str):
        return ""
    text = lowercase(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text
