"""
Quick sanity check: prints how many training examples exist per category
in data/question_train.csv. Handy after editing the CSV, to confirm the
classes are still balanced before retraining with question_train.py.
"""
import pandas as pd

df = pd.read_csv("data/question_train.csv")
print(df["label"].value_counts())