from fastembed import TextEmbedding
import numpy as np
import json

model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

with open("data/qa_pairs.json", "r", encoding="utf-8") as f:
    qa = json.load(f)

texts = [x["question"] for x in qa]
emb = np.array(list(model.embed(texts)))
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

np.save("data/question_embeddings.npy", emb)