#!/usr/bin/env python3
# coding: utf-8
"""
Pre-compute MiniLM embeddings for every question in the knowledge base.

Run this whenever data/qa_pairs.json changes:

    python build_embeddings.py

QAAssistant refuses to start if the saved vectors and qa_pairs.json have
drifted out of sync, so a forgotten rebuild fails loudly instead of quietly
pairing questions with the wrong vectors.

Doubles as the deploy warm-up step: it populates the FastEmbed model cache,
so the running server doesn't have to download ~87 MB on its first request.
"""

import os
import json
import numpy as np

from qa_assistant import get_embedding_model, EMBEDDING_CACHE_DIR

CUR = os.path.dirname(os.path.abspath(__file__))


def main():
    print(f"Model cache: {EMBEDDING_CACHE_DIR}")
    model = get_embedding_model()

    qa_path = os.path.join(CUR, "data", "qa_pairs.json")
    with open(qa_path, "r", encoding="utf-8") as f:
        qa = json.load(f)

    texts = [x["question"] for x in qa]
    print(f"Embedding {len(texts)} questions...")

    emb = np.array(list(model.embed(texts)))

    # Store L2-normalised, so similarity at query time is a plain dot product.
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    out_path = os.path.join(CUR, "data", "question_embeddings.npy")
    np.save(out_path, emb)
    print(f"Saved {emb.shape} -> {out_path}")


if __name__ == "__main__":
    main()
