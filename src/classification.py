from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import normalize_text


def train_text_classifier(
    frame,
    text_col="normalized_title",
    label_col="category_label",
    test_size=0.2,
    random_state=42,
    max_rows=None,
):
    dataset = frame[[text_col, label_col]].dropna().copy()
    dataset = dataset[dataset[text_col].astype(str).str.strip().ne("")]
    counts = dataset[label_col].value_counts()
    dataset = dataset[dataset[label_col].isin(counts[counts >= 2].index)]
    if dataset[label_col].nunique() < 2:
        raise ValueError("At least two categories with two records each are required.")
    if max_rows and len(dataset) > max_rows:
        per_class = max(2, max_rows // dataset[label_col].nunique())
        dataset = pd.concat(
            [
                group.sample(min(len(group), per_class), random_state=random_state)
                for _, group in dataset.groupby(label_col, sort=True)
            ],
            ignore_index=True,
        )

    X = dataset[text_col].astype(str)
    y = dataset[label_col].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.97,
                    max_features=100_000,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    labels = sorted(y.unique())
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, labels=labels, average="macro", zero_division=0
    )
    probabilities = pipeline.predict_proba(X_test)
    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels),
        "labels": labels,
        "report": report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_size": test_size,
        "random_state": random_state,
        "training_rows": len(dataset),
    }


def predict_category(model, texts):
    if isinstance(texts, str):
        texts = [texts]
    clean_texts = [normalize_text(text) for text in texts]
    if any(not text for text in clean_texts):
        raise ValueError("Enter a non-empty product description for each prediction.")
    predicted = model.predict(clean_texts)
    probabilities = model.predict_proba(clean_texts) if hasattr(model, "predict_proba") else None
    return predicted, probabilities


def save_model_artifact(model_result, fingerprint, artifact_path):
    artifact = {
        "version": 1,
        "dataset_fingerprint": fingerprint,
        "model_result": model_result,
    }
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_model_artifact(fingerprint, artifact_path):
    path = Path(artifact_path)
    if not path.is_file():
        return None
    artifact = joblib.load(path)
    if artifact.get("version") != 1 or artifact.get("dataset_fingerprint") != fingerprint:
        return None
    return artifact.get("model_result")
