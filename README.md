# ⚖️ LegalLens

**AI-powered legal guidance assistant for everyday legal questions.**

Describe a legal issue in plain English — LegalLens classifies it, searches a curated legal knowledge base using hybrid (TF-IDF + semantic) retrieval, and returns the closest matching guidance, with a similarity score to show how confident the match is.

🔗 **Live demo:** [legallens-il4m.onrender.com](https://legallens-il4m.onrender.com/)

---

## Features

- 🧠 **Hybrid retrieval** — TF-IDF keyword matching blended with semantic search (`all-MiniLM-L6-v2` via FastEmbed) for results that hold up even when the wording doesn't match exactly
- 🎯 **Automatic category prediction** across 13 legal categories, from a Logistic Regression classifier
- 🛑 **Knows what it doesn't cover** — small talk and off-topic questions get a direct response instead of a forced legal match
- 🤝 **Shows both sides of a close call** — when two answers fit about equally well, you pick which one applies
- 💬 **Chat history**, saved locally in your browser — search, rename, and revisit past conversations
- 📚 223 curated Q&A entries, each with practical next-step guidance
- ⚡ Lightweight ONNX-based inference, built for fast cold starts on free-tier hosting

---

## Tech Stack

**Backend:** Python, Flask, scikit-learn, FastEmbed, NumPy
**Frontend:** HTML, CSS, JavaScript
**ML:** TF-IDF vectorizer + MiniLM sentence embeddings, cosine similarity, Logistic Regression classifier
**Deployment:** Render, gunicorn

---

## How it works

1. **Classify** — predict the most likely legal category for the question.
2. **Retrieve** — score the entire knowledge base with blended TF-IDF + semantic similarity.
3. **Decide** — if nothing is a close enough match, say so; if it's genuinely off-topic, say that too.
4. **Respond** — return the best-matching question, its guidance, and how confident the match is. If two answers are neck-and-neck, show both.

---

## Getting Started

```bash
git clone https://github.com/Mansi114114/LegalLens.git
cd LegalLens
pip install -r requirements.txt

python build_embeddings.py   # builds the semantic search index
python app.py                 # http://127.0.0.1:5000
```

To retrain the category classifier after editing the training data:

```bash
pip install -r requirements-dev.txt
python question_train.py
```

---

## Project Structure

```
LegalLens/
├── app.py                 # Flask app & routes
├── qa_assistant.py         # Retrieval engine
├── question_classify.py    # Category classifier
├── build_embeddings.py     # Precomputes semantic embeddings
├── question_train.py       # Trains the classifier
├── data/                   # Knowledge base & training data
├── model/                  # Trained classifier
├── templates/, static/     # Frontend
└── render.yaml              # Deployment config
```

---

## Disclaimer

LegalLens provides general legal information only — it is **not legal advice**. Always consult a qualified lawyer for matters specific to your situation.