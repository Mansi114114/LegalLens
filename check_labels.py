"""
Quick sanity check: prints how many training examples exist per category
in data/question_train.csv. Handy after editing the CSV, before retraining
with question_train.py.

The classes are not expected to be exactly equal — relabelling the misfiled
inheritance questions deliberately left "Property dispute" smaller than the
rest, and question_train.py compensates with class_weight="balanced". Watch
for a category collapsing to a handful of rows, not for small differences.

Needs pandas: pip install -r requirements-dev.txt
"""
import os

import pandas as pd

CUR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(CUR, "data", "question_train.csv"))
print(df["label"].value_counts().to_string())
print(f"\ntotal: {len(df)} rows, {df['label'].nunique()} categories")
