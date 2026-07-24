#!/usr/bin/env python3
# coding: utf-8
"""
Web UI for the Legal QA Assistant.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request, jsonify

from qa_assistant import QAAssistant

app = Flask(__name__)

# Load the assistant once at startup (loading the model/data on every
# request would be slow and is unnecessary).
assistant = QAAssistant()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please type a question."}), 400

    payload = assistant.answer(question)
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
