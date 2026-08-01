# create_embeddings.py
from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/qa_pairs.json", "r", encoding="utf-8") as f:
    qa = json.load(f)

texts = [x["question"] for x in qa]

emb = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save("data/question_embeddings.npy", emb)