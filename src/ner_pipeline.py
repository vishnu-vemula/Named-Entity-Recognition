from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    import nltk

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
except Exception:  # pragma: no cover - optional if NLTK is unavailable
    pass

STOP_WORDS = set(stopwords.words("english"))


def ensure_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download

        download(model_name)
        return spacy.load(model_name)


NLP = ensure_spacy_model()


def load_dataset(data_dir: str | Path) -> pd.DataFrame:
    base_path = Path(data_dir)
    files = [
        ("gossipcop_fake.csv", "fake", "gossip"),
        ("gossipcop_real.csv", "real", "gossip"),
        ("politifact_fake.csv", "fake", "politifact"),
        ("politifact_real.csv", "real", "politifact"),
    ]

    frames = []
    for filename, label, domain in files:
        file_path = base_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {file_path}")

        frame = pd.read_csv(file_path)
        frame["label"] = label
        frame["domain"] = domain
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return combined


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [token for token in word_tokenize(text) if token not in STOP_WORDS]
    if not tokens:
        return ""

    doc = NLP(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    return " ".join(lemmatized)


def prepare_features(data_dir: str | Path) -> pd.DataFrame:
    df = load_dataset(data_dir)
    df = df.dropna(subset=["title", "news_url"]).copy()

    df["title_clean"] = df["title"].map(clean_text)
    df["url_clean"] = df["news_url"].map(clean_text)
    df["tweet_count"] = df["tweet_ids"].fillna("").map(lambda value: len(str(value).split("\t")))

    return df


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, strip_accents="unicode")),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]
    )


def train_and_evaluate(data_dir: str | Path) -> tuple[Pipeline, dict]:
    df = prepare_features(data_dir)
    features = df["title_clean"] + " " + df["url_clean"]
    target = df["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cleaned NLP pipeline for the fake-vs-real news dataset.")
    parser.add_argument("--data-dir", default="data", help="Directory containing the CSV files used for training.")
    args = parser.parse_args()

    model, metrics = train_and_evaluate(args.data_dir)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification report:\n")
    print(metrics["classification_report"])
    print("\nConfusion matrix:\n")
    print(metrics["confusion_matrix"])

    print(f"\nModel ready: {type(model).__name__}")


if __name__ == "__main__":
    main()
