# ⚖️ CrimeKG Legal QA Assistant

A legal Q&A tool: type a question in plain English, it classifies the legal
category (labour, tenancy, cybercrime, etc.), retrieves the closest matching
question from a curated knowledge base, and returns the stored guidance —
through a proper web chat UI, not just a terminal loop.

This project builds on an earlier prototype's basic idea (classify a legal
question, then look up a canned answer) but has been substantially rewritten:
a fragile, dead-end retrieval pipeline replaced with a resilient one that
recovers gracefully instead of returning "no answer found", a small but
richer knowledge base, and a full web UI added on top of what was
originally a bare terminal script. Every file below is actively used —
no unused legacy code is included.

> **Disclaimer:** This tool gives general information, not legal advice.
> The UI states this, and that framing should stay in place if you extend it.

## Features

- 🧠 TF-IDF + Logistic Regression classifier across 11 legal categories
- 🔍 Resilient retrieval that widens scope instead of dead-ending on "no answer found"
- 💬 Web chat UI (Flask + vanilla JS) alongside a terminal CLI
- 📚 A small, curated, extensible knowledge base (`data/qa_pairs.json`)
- 🔁 One-command retraining when you edit the training data

## Requirements

- Python 3.10+
- pip

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

Prefer the terminal? `python qa_assistant.py` still works the same way.

## How it works

1. **Classify** — `question_classify.py` loads a trained TF‑IDF +
   LogisticRegression pipeline (`model/question_text.model`) that predicts
   one of 11 legal categories, along with a confidence score. It's trained
   on `data/question_train.csv` (380 example phrasings — formal, casual,
   and typo'd — across the 11 categories).
2. **Retrieve** — `qa_assistant.py` vectorizes your question and the
   knowledge base (`data/qa_pairs.json`, 154 question/answer entries) with
   a stemmed TF‑IDF, then ranks matches by cosine similarity.
   - If the classifier is confident, the search is scoped to that category
     first.
   - If nothing scores highly enough there (or the classifier wasn't
     confident), the search automatically **widens to the whole knowledge
     base** instead of giving up.
   - Widened results still get a small boost toward the classifier's top
     guess (even when it wasn't confident enough to hard-filter on) — this
     stops a coincidental keyword overlap in the wrong category from
     narrowly beating the actually-relevant answer.
   - Widened results are then filtered to stay within one coherent category
     (the best match's own category), so you don't get an unrelated answer
     tacked on just because it also cleared the similarity bar.
   - If still nothing matches closely, you get general guidance for the
     predicted category rather than a dead end — and if the question
     doesn't look like it's about any covered legal topic at all, it says
     so plainly instead of guessing.
3. **Serve** — `app.py` is a small Flask app exposing `POST /api/ask` and a
   chat UI (`templates/index.html`, `static/`).

## Project structure

```
CrimeAssistant/
├── app.py                    # Flask web server
├── qa_assistant.py           # retrieval engine (+ terminal CLI)
├── question_classify.py      # loads and queries the trained classifier
├── question_train.py         # trains/retrains the classifier
├── check_labels.py           # quick label-count sanity check
├── templates/index.html      # chat UI
├── static/style.css          # UI styling
├── static/script.js          # UI logic
├── data/
│   ├── qa_pairs.json         # knowledge base (question → type → answers)
│   └── question_train.csv    # labeled training data for the classifier
├── model/question_text.model # trained classifier (joblib)
└── requirements.txt
```

## API

`POST /api/ask` accepts JSON and returns the assistant's answer.

**Request**

```json
{ "question": "My landlord won't return my security deposit" }
```

**Response**

```json
{
  "category": "Housing / tenancy",
  "confidence": 0.87,
  "answers": [
    "Review your rental agreement and send a written notice.",
    "Seek legal advice if the deposit is withheld without justification."
  ]
}
```

Exact response fields depend on the implementation in `qa_assistant.py` —
treat the above as a representative shape, not a guaranteed schema.

## Retraining the classifier

If you edit `data/question_train.csv`, retrain with:

```bash
python question_train.py
```

This re-fits the pipeline against your currently installed scikit-learn
version and overwrites `model/question_text.model`, and prints a
cross-validated accuracy report so you can see how well it's generalizing.
Retraining after any Python/library upgrade also avoids scikit-learn
version-mismatch warnings (and occasional bad predictions) from loading a
model pickled by a different version.

## Extending the knowledge base

Add entries to `data/qa_pairs.json` in this shape:

```json
{
  "question": "My landlord is refusing to return my security deposit.",
  "type": "Housing / tenancy",
  "answers": [
    "Review your rental agreement and send a written notice.",
    "Seek legal advice if the deposit is withheld without justification."
  ]
}
```

`type` must be one of the categories the classifier knows about — run
`python check_labels.py` to see the current list and how many training
examples back each one. No retraining needed for knowledge-base changes
only — only changes to `question_train.csv` require re-running
`question_train.py`.

## Troubleshooting

- **`ModuleNotFoundError` on startup** — make sure you've run
  `pip install -r requirements.txt` inside the environment you're using to
  run the app.
- **scikit-learn version-mismatch warning when loading the model** —
  harmless in most cases, but re-run `python question_train.py` to
  regenerate `model/question_text.model` against your installed version if
  you also notice degraded predictions.
- **NLTK resource errors** (e.g. missing stopwords/punkt data) — run the
  relevant `nltk.download(...)` call once in a Python shell; which
  resource is needed will be named in the error message.
- **Port 5000 already in use** — either free the port or run with a
  different one, e.g. `flask run --port 5001` or by editing the
  `app.run(...)` call in `app.py`.

## Notes

- This tool gives general information, not legal advice — the UI says so,
  and that framing should stay in place if you extend it.
- `check_labels.py` is a convenience script, not required for the app to
  run — safe to remove if you don't need it.

## License

No license file is currently included. Add one (e.g. MIT, Apache-2.0) if
you plan to share or open-source this project — until then, all rights are
reserved by default.