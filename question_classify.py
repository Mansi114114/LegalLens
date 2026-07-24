#!/usr/bin/env python3
# coding: utf-8
# Legal question-type classifier using TF‑IDF + scikit‑learn

import os
import joblib


class QuestionClassify(object):
    """Thin wrapper around the trained classifier pipeline.

    The pipeline itself (TF-IDF vectorizer + LogisticRegression) is trained
    by question_train.py and saved to model/question_text.model. This class
    just loads that pickle and exposes a simple predict() call so the rest
    of the app doesn't need to know how the model is built.
    """

    def __init__(self):
        cur = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(cur, "model", "question_text.model")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please run question_train.py first to train and save the model."
            )

        # joblib.load restores the whole scikit-learn Pipeline object,
        # vectorizer and classifier included — no separate vocabulary file
        # needed.
        self.model = joblib.load(self.model_path)

    def predict(self, text: str) -> str:
        """Return the single best-guess category label for `text`.

        Note: qa_assistant.py mostly uses predict_proba directly (via
        self.model) instead of this method, since it needs the confidence
        score too, not just the top label.
        """
        if not text.strip():
            return "Empty question"
        pred = self.model.predict([text])[0]
        return pred


def main():
    handler = QuestionClassify()
    print("Legal Question Type Classifier (blank line to exit)\n")
    while True:
        q = input("Enter legal question: ").strip()
        if not q:
            print("Exiting.")
            break
        label = handler.predict(q)
        print("Predicted question type:", label)
        print("-" * 60)


if __name__ == "__main__":
    main()