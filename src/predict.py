"""
Inference utilities. Loads the trained vectorizer + model once (cached)
and exposes predict(text) / predict_batch(texts) for use by the
Streamlit app or anything else.
"""

import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


class ArtifactsNotFoundError(FileNotFoundError):
    """Raised when trained model artifacts are missing."""


@lru_cache(maxsize=1)
def _load_artifacts():
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    model_path = MODELS_DIR / "logistic_model.pkl"
    label_map_path = MODELS_DIR / "emotion_number.json"

    if not (vectorizer_path.exists() and model_path.exists() and label_map_path.exists()):
        raise ArtifactsNotFoundError(
            "Model artifacts not found in models/. Run `python src/train.py` first."
        )

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    number_to_emotion = {int(k): v for k, v in label_map["number_to_emotion"].items()}
    return vectorizer, model, number_to_emotion, label_map.get("accuracy")


def get_model_accuracy():
    """Return the held-out test accuracy recorded at training time."""
    _, _, _, accuracy = _load_artifacts()
    return accuracy


def get_emotion_classes():
    """Return the list of emotion labels the model was trained on."""
    _, _, number_to_emotion, _ = _load_artifacts()
    return [number_to_emotion[i] for i in sorted(number_to_emotion)]


def predict(text: str) -> dict:
    """
    Predict the emotion of a piece of text.

    Returns:
        {
            "emotion": str | None,        # top predicted label, None if input was empty after cleaning
            "probabilities": dict,        # {emotion_label: probability}
            "cleaned_text": str,          # text after preprocessing
        }
    """
    vectorizer, model, number_to_emotion, _ = _load_artifacts()

    cleaned = clean_text(text)
    if not cleaned.strip():
        return {"emotion": None, "probabilities": {}, "cleaned_text": cleaned}

    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    pred_idx = int(proba.argmax())

    probabilities = {number_to_emotion[i]: float(p) for i, p in enumerate(proba)}
    return {
        "emotion": number_to_emotion[pred_idx],
        "probabilities": probabilities,
        "cleaned_text": cleaned,
    }


def predict_batch(texts) -> list:
    """Run predict() over an iterable of texts."""
    return [predict(t) for t in texts]
