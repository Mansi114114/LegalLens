#!/usr/bin/env python3
# coding: utf-8
# Train a legal question-type classifier using TF‑IDF + LogisticRegression

import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score


def get_paths():
    cur = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(cur, "data", "question_train.csv")
    model_dir = os.path.join(cur, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "question_text.model")
    return data_path, model_path


def load_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training file not found: {csv_path}\n"
            f"Please create data/question_train.csv with columns: label,text"
        )

    df = pd.read_csv(csv_path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("question_train.csv must have columns: label,text")

    df = df.dropna(subset=["label", "text"])
    print("First 5 labels from CSV:", df["label"].head().tolist())
    print("First 5 texts from CSV:", df["text"].head().tolist())
    return df["text"].tolist(), df["label"].tolist()


def build_pipeline():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=8000,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
        ))
    ])
    return pipeline


def train_and_evaluate():
    data_path, model_path = get_paths()
    print(f"Loading training data from: {data_path}")
    texts, labels = load_data(data_path)

    print(f"Samples: {len(texts)}")

    # The dataset is small, so a single train/test split wastes data and
    # gives a noisy estimate. Stratified k-fold cross-validation gives a
    # more honest picture of accuracy while still using every example.
    n_splits = min(5, min(pd.Series(labels).value_counts()))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print(f"Running {n_splits}-fold cross-validation...")
    y_pred = cross_val_predict(build_pipeline(), texts, labels, cv=cv)
    acc = accuracy_score(labels, y_pred)
    print(f"Cross-validated accuracy: {acc:.4f}")
    print("\nClassification report (cross-validated):")
    print(classification_report(labels, y_pred, zero_division=0))

    # Fit the final model on ALL available data — every example counts when
    # the dataset is this small, and there's no separate holdout to protect.
    model = build_pipeline()
    print("Fitting final model on the full dataset...")
    model.fit(texts, labels)

    print(f"Saving model to: {model_path}")
    joblib.dump(model, model_path)
    print("Done.")


if __name__ == "__main__":
    train_and_evaluate()