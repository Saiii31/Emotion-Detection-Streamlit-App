"""
Trains the emotion-detection model (TF-IDF + Logistic Regression, the
best-performing combination from the original experimentation notebook)
and saves the vectorizer, model, and label map to models/.

Usage:
    python src/train.py
"""

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.txt"
MODELS_DIR = BASE_DIR / "models"


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, sep=";", header=None, names=["text", "emotion"])
    print(f"Loaded {len(df)} rows across {df['emotion'].nunique()} emotion classes.")

    unique_emotions = sorted(df["emotion"].unique())
    emotion_to_number = {emo: i for i, emo in enumerate(unique_emotions)}
    number_to_emotion = {i: emo for emo, i in emotion_to_number.items()}

    df["label"] = df["emotion"].map(emotion_to_number)
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=[number_to_emotion[i] for i in sorted(number_to_emotion)]
    )

    print(f"\nAccuracy: {accuracy:.4f}\n")
    print(report)

    with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / "logistic_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "emotion_number.json", "w") as f:
        json.dump(
            {
                "emotion_to_number": emotion_to_number,
                "number_to_emotion": number_to_emotion,
                "accuracy": accuracy,
            },
            f,
            indent=2,
        )

    print(f"\nSaved artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
