# ⚖️ LegalLens – Intelligent Legal Question Answering using Hybrid Retrieval

LegalLens is an AI-powered legal guidance assistant that helps users describe legal issues in plain English, automatically identifies the legal category, retrieves the most relevant cases using **Hybrid Retrieval (TF-IDF + Sentence Transformers)**, and provides practical legal guidance through a modern web interface.

> **Disclaimer:** LegalLens provides general legal information only. It is **not legal advice**. Always consult a qualified legal professional for important legal decisions.

---

## ✨ Features

- 🤖 Automatic legal category classification
- 🔍 Hybrid Retrieval (TF-IDF + Sentence Transformers)
- 📊 Normalized similarity scores for fair ranking
- 📚 Curated legal knowledge base
- 💬 Modern chat-based web interface
- 🎯 Category-first retrieval with intelligent fallback
- ⚡ Fast Flask backend with REST API
- 📱 Responsive dark-themed UI

---

## 🛠 Tech Stack

### Backend
- Python
- Flask
- Scikit-learn
- Sentence Transformers
- NumPy
- Joblib

### Frontend
- HTML5
- CSS3
- Vanilla JavaScript

### Machine Learning
- Logistic Regression
- TF-IDF Vectorizer
- Sentence Transformers
- Cosine Similarity

---

## 🚀 How It Works

### 1. User Question

The user types a legal question in plain English.

Example:

> My landlord won't return my security deposit.

---

### 2. Legal Classification

A Logistic Regression classifier predicts the legal category.

Examples:

- Housing / Tenancy
- Consumer Complaint
- Cybercrime
- Medical Negligence
- Labour Dispute
- Property Dispute

---

### 3. Hybrid Retrieval

LegalLens performs retrieval using **two different methods**.

#### TF-IDF Retrieval

Finds keyword-based similarity.

#### Semantic Retrieval

Sentence Transformers understand the meaning of the sentence.

Example:

> "Doctor operated on the wrong body part."

matches

> "The doctor gave the wrong treatment."

even without identical words.

---

### 4. Score Normalization

Since TF-IDF and semantic similarities use different value ranges, both scores are normalized before combining.

```
Final Score =
0.4 × Normalized TF-IDF
+
0.6 × Normalized Semantic Score
```

This prevents one retrieval method from dominating the final ranking.

---

### 5. Intelligent Ranking

Results are ranked using the combined hybrid similarity score.

If no strong match exists within the predicted legal category, the search automatically expands across all legal categories.

---

### 6. Response Generation

LegalLens returns

- Predicted legal category
- Most similar legal question
- Similarity score
- Legal guidance
- Additional relevant matches (optional)

---

## 📂 Project Structure

```
CrimeAssistant/
│
├── app.py                     # Flask application
├── qa_assistant.py            # Hybrid retrieval engine
├── question_classify.py       # Category classifier
├── question_train.py          # Train classifier
├── check_labels.py            # Dataset validation
│
├── data/
│   ├── qa_pairs.json
│   └── question_train.csv
│
├── model/
│   └── question_text.model
│
├── templates/
│   ├── landing.html
│   └── index.html
│
├── static/
│   ├── css/
│   │   ├── landing.css
│   │   └── style.css
│   │
│   ├── script.js
│   └── favicon.svg
│
├── requirements.txt
└── README.md
```

---

## 📸 Application Flow

Landing Page

↓

User enters legal question

↓

Legal category prediction

↓

Hybrid Retrieval

- TF-IDF
- Sentence Transformers

↓

Similarity score normalization

↓

Ranking

↓

Most relevant legal guidance returned

---

## 📡 API

### POST `/api/ask`

### Request

```json
{
  "question": "My landlord won't return my security deposit."
}
```

### Sample Response

```json
{
  "type": "Housing / Tenancy",
  "similarity_score": 0.91,
  "results": [
    {
      "question": "My landlord is refusing to return my security deposit.",
      "score": 0.91,
      "answers": [
        "Review your rental agreement.",
        "Send a written notice.",
        "Consult a lawyer if necessary."
      ]
    }
  ]
}
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

---

## 📚 Knowledge Base

Legal questions and guidance are stored in

```
data/qa_pairs.json
```

New legal entries can be added without retraining the retrieval system.

---

## 🧠 Retraining the Classifier

If you update

```
data/question_train.csv
```

retrain the classifier using

```bash
python question_train.py
```

This generates a new trained model in

```
model/question_text.model
```

---

## 🔮 Future Improvements

- Voice-based legal queries
- OCR for legal documents
- LLM-generated legal explanations
- Multilingual support
- Case document search
- User authentication
- Legal document summarization

---

## ⚠️ Disclaimer

LegalLens provides general legal information and educational guidance only.

It is **not a substitute for professional legal advice**. Always consult a qualified legal practitioner before making legal decisions.
