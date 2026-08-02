# ⚖️ LegalLens – AI-Powered Legal Question Answering using Hybrid Retrieval

LegalLens is an AI-powered legal guidance assistant that helps users describe legal issues in plain English, automatically predicts the most relevant legal category, retrieves the best matching legal guidance using **Hybrid Retrieval (TF-IDF + FastEmbed)**, and presents practical legal information through a modern, responsive web interface.

> **Disclaimer:** LegalLens provides general legal information only. It is **not legal advice**. Always consult a qualified legal professional before making legal decisions.

---

# 🌐 Live Demo

🚀 **Try LegalLens here**

**https://legallens-il4m.onrender.com/**

---

# ✨ Features

- 🤖 Automatic legal category prediction
- 🔍 Hybrid Retrieval (TF-IDF + FastEmbed)
- 🧠 Semantic search using **all-MiniLM-L6-v2**
- 📊 Weighted similarity scoring
- 📚 Curated legal knowledge base
- 🎯 Category-first retrieval with intelligent fallback
- ⚡ Memory-optimized inference using ONNX Runtime
- 💬 Interactive chatbot interface
- 📱 Responsive dark-themed UI
- 🚀 Deployed on Render

---

# 🛠 Tech Stack

## Backend

- Python
- Flask
- Scikit-learn
- FastEmbed
- NumPy
- Joblib

## Frontend

- HTML5
- CSS3
- JavaScript

## Machine Learning

- Logistic Regression
- TF-IDF Vectorizer
- FastEmbed (all-MiniLM-L6-v2)
- Cosine Similarity

---

# 🚀 How It Works

## Step 1 – User Question

The user enters a legal question in plain English.

Example:

> My landlord won't return my security deposit.

---

## Step 2 – Legal Category Prediction

A Logistic Regression classifier predicts the most relevant legal category.

Supported categories include:

- Marriage and Family
- Labour Dispute
- Traffic Accident
- Debt Dispute
- Criminal Defence
- Property Dispute
- Consumer Complaint
- Cybercrime
- Medical Negligence
- Housing / Tenancy
- Education Dispute
- Insurance Claims
- Inheritance & Succession

---

## Step 3 – Hybrid Retrieval

LegalLens retrieves the most relevant legal guidance using **two complementary retrieval methods.**

### 🔹 TF-IDF Retrieval

Captures keyword similarity.

Example

```
salary not paid
```

matches

```
salary payment delayed
```

because important keywords overlap.

---

### 🔹 Semantic Retrieval

FastEmbed generates semantic embeddings using the **all-MiniLM-L6-v2** embedding model.

Instead of relying on identical words, it understands sentence meaning.

Example

```
Doctor operated on the wrong body part.
```

matches

```
The doctor gave the wrong treatment.
```

even though the wording is different.

---

## Step 4 – Score Normalization

TF-IDF and semantic similarities produce scores on different scales.

Both scores are normalized before combining.

```
Final Score

=

0.4 × TF-IDF Score

+

0.6 × Semantic Score
```

This prevents one retrieval method from dominating the ranking.

---

## Step 5 – Intelligent Ranking

The highest-ranked legal guidance is returned.

If there is no strong match inside the predicted category, LegalLens automatically expands the search across every legal category.

---

## Step 6 – Response

The assistant returns

- Predicted legal category
- Most similar legal question
- Similarity score
- Legal guidance
- Additional relevant results

---

# 📂 Project Structure

```text
LegalLens/
│
├── app.py
├── qa_assistant.py
├── question_classify.py
├── question_train.py
├── build_embeddings.py
├── check_labels.py
│
├── data/
│   ├── qa_pairs.json
│   ├── question_train.csv
│   └── question_embeddings.npy
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

# 📸 Application Flow

```
Landing Page

        │

        ▼

User enters legal question

        │

        ▼

Legal Category Prediction

        │

        ▼

Hybrid Retrieval

   ├── TF-IDF
   └── FastEmbed

        │

        ▼

Similarity Score Normalization

        │

        ▼

Weighted Ranking

        │

        ▼

Most Relevant Legal Guidance Returned
```

---

# 📡 REST API

## POST `/api/ask`

### Request

```json
{
    "question": "My landlord won't return my security deposit."
}
```

---

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

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Mansi114114/CrimeAssistant.git
```

Move into the project

```bash
cd CrimeAssistant
```

Install dependencies

```bash
pip install -r requirements.txt
```

Generate semantic embeddings

```bash
python build_embeddings.py
```

Train the classifier

```bash
python question_train.py
```

Run the application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# 📚 Knowledge Base

Legal questions and guidance are stored in

```
data/qa_pairs.json
```

Each record contains

- Legal question
- Legal category
- Practical legal guidance

New entries can be added easily without changing the application code.

---

# 🧠 Semantic Embeddings

Semantic embeddings are generated using

```
build_embeddings.py
```

Run

```bash
python build_embeddings.py
```

This creates

```
data/question_embeddings.npy
```

The application loads these precomputed embeddings during startup instead of generating them every time.

This provides:

- ⚡ Faster startup
- 💾 Lower memory usage
- 🚀 Faster semantic retrieval
- ☁️ Easier deployment on cloud platforms like Render

---

# 🎯 Retraining the Classifier

If you update

```
data/question_train.csv
```

retrain the classifier using

```bash
python question_train.py
```

This generates

```
model/question_text.model
```

---

# 📈 Future Improvements

- 🎙 Voice-based legal queries
- 🌍 Multilingual support
- 📄 OCR for legal documents
- 🤖 LLM-generated legal explanations
- 📑 Legal document summarization
- 📂 PDF upload support
- 🔐 User authentication
- 💬 Conversation history
- ⚖️ Case law search

---

# ⚠️ Disclaimer

LegalLens provides general legal information and educational guidance only.

The information returned by this application should **not** be considered professional legal advice.

Always consult a qualified legal practitioner before making legal decisions.

---

⭐ If you found this project useful, consider giving it a star on GitHub!