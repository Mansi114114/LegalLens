#!/usr/bin/env python3
# coding: utf-8
"""
Legal QA Assistant — retrieval engine.

Pipeline:
  1. Classify the question into a legal category (with confidence).
  2. Search the knowledge base for the closest matching question(s)
     using TF-IDF + cosine similarity.
  3. If the category-filtered search comes up empty (or the classifier
     wasn't confident), fall back to searching the *entire* knowledge
     base instead of giving up.
  4. De-duplicate near-identical "(Scenario N)" variants so the same
     answer isn't shown three times.
"""

import os
import re
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

from question_classify import QuestionClassify

_stemmer = PorterStemmer()
_TOKEN_RE = re.compile(r"[a-zA-Z]+")

# A small, hand-picked list of genuine function words (articles, pronouns,
# prepositions, auxiliary verbs, conjunctions). Deliberately NOT using
# sklearn's built-in "english" stop word list — it's an old, quirky list
# that includes ordinary content words like "fire", "call", "give", "put",
# and "found", which are exactly the kind of words that matter in a legal
# question (e.g. "fired", "given notice"). Silently dropping those broke
# matching on real questions like "my boss fire me from job".
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "so", "as", "of", "in",
    "on", "at", "by", "for", "with", "about", "against", "to", "from",
    "up", "down", "into", "over", "under", "again", "further", "then",
    "once", "here", "there", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "than", "too", "very", "s", "t", "can", "will", "just",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "until", "while", "when", "where", "why", "how",
}
_STEMMED_STOP_WORDS = sorted({_stemmer.stem(w) for w in _STOP_WORDS})

# Minimum cosine similarity to accept a match at all.
MIN_SCORE = 0.12

# Minimum classifier confidence before we trust the predicted category
# enough to filter the search space with it. Below this we just search
# everything, which is safer than filtering on a shaky guess.
MIN_TYPE_CONFIDENCE = 0.22

# Even when the classifier isn't confident enough to hard-filter the search
# (above), its top guess is still informative — used as a small tiebreaker
# bonus when ranking widened/fallback search results, so a near-tied but
# coincidental keyword overlap in the wrong category doesn't win outright.
CLASSIFIER_TIEBREAK_BOOST = 0.05


SCENARIO_SUFFIX_RE = re.compile(r"\s*\(Scenario\s*\d+\)\s*$", re.IGNORECASE)

# Generic closing clauses that show up across many categories ("What legal
# action can I take?", "What should I do?", ...). Left in, these can
# dominate TF-IDF similarity through rare shared trigrams even when the
# substantive content doesn't match at all, so we strip them before
# vectorizing (display text is untouched).
BOILERPLATE_RE = re.compile(
    r"\b(what (legal action|should|are|is|can)\b.*?\?|"
    r"how (can|do|should)\b.*?\?)",
    re.IGNORECASE,
)


def _strip_boilerplate(text: str) -> str:
    stripped = BOILERPLATE_RE.sub(" ", text)
    stripped = stripped.strip()
    # Don't return an empty string if the whole question was boilerplate
    # (shouldn't normally happen, but be defensive).
    return stripped if stripped else text


def _stem_tokenize(text: str):
    """Lowercase, split into words, and reduce each to its word stem.

    Without this, "fired" and "fire", or "forced"/"force"/"forcefully",
    are completely different tokens to TF-IDF and share no similarity at
    all — even though they mean the same thing. Stemming collapses them
    to a common root ("fire", "forc") so questions phrased with different
    verb tenses/forms still match.
    """
    return [_stemmer.stem(t) for t in _TOKEN_RE.findall(text.lower())]


def _clean_display_question(q: str) -> str:
    """Strip the internal '(Scenario N)' suffix used for training variety."""
    return SCENARIO_SUFFIX_RE.sub("", q).strip()


# Generic fallback advice per legal category, used only when we truly
# can't find a close match, so the user still gets something useful
# instead of a dead end.
GENERIC_ADVICE = {
    "Marriage and family": [
        "Consult a family lawyer to understand your rights regarding marriage, custody, or divorce.",
        "If there is abuse or threat involved, you can also file a police complaint or seek a protection order.",
    ],
    "Labour dispute": [
        "Keep written evidence (emails, messages, payslips) related to the dispute.",
        "Consult a labour lawyer or approach your local labour authority/tribunal.",
    ],
    "Traffic accident": [
        "File a First Information Report (FIR) at the nearest police station if injury or damage occurred.",
        "Collect evidence (photos, witness details) and consult a lawyer about insurance claims or compensation.",
    ],
    "Debt dispute": [
        "Gather all loan or payment records related to the debt.",
        "Consult a lawyer about sending a legal notice or pursuing recovery through the appropriate court.",
    ],
    "Criminal defence": [
        "Do not make statements to the police without legal counsel present.",
        "Consult a criminal defence lawyer immediately to understand the charges and your rights.",
    ],
    "Property dispute": [
        "Gather all property documents (title deed, sale agreement, tax receipts).",
        "Consult a property lawyer about mediation or filing a civil suit.",
    ],
    "Consumer complaint": [
        "Keep receipts, warranties, and any communication with the seller/service provider.",
        "File a complaint with the Consumer Disputes Redressal Commission or consult a consumer lawyer.",
    ],
    "Cybercrime": [
        "Preserve evidence such as screenshots, messages, and transaction IDs.",
        "Report the incident on the national cybercrime portal or your local cybercrime cell.",
    ],
    "Medical negligence": [
        "Collect all medical records, prescriptions, and bills related to the treatment.",
        "Consult a lawyer about filing a complaint with the medical council or consumer forum.",
    ],
    "Housing / tenancy": [
        "Review your rental agreement for the relevant clauses.",
        "Consult a lawyer or local rent authority about your rights as a tenant/landlord.",
    ],
    "Education dispute": [
        "Keep copies of admission forms, fee receipts, and correspondence with the institution.",
        "Consult a lawyer or approach the relevant education regulatory body.",
    ],
}


class QAAssistant:
    def __init__(self):
        cur = os.path.dirname(os.path.abspath(__file__))
        qa_path = os.path.join(cur, "data", "qa_pairs.json")

        if not os.path.exists(qa_path):
            raise FileNotFoundError(
                f"QA data file not found: {qa_path}\n"
                f"Please create data/qa_pairs.json."
            )

        with open(qa_path, "r", encoding="utf-8") as f:
            self.qa_pairs = json.load(f)

        self.questions = [item["question"] for item in self.qa_pairs]
        self.types = [item.get("type", "") for item in self.qa_pairs]

        # Text actually fed to the vectorizer: boilerplate closing clauses
        # stripped out so they can't dominate similarity over the
        # substantive part of the question.
        self._retrieval_texts = [_strip_boilerplate(q) for q in self.questions]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            tokenizer=_stem_tokenize,
            token_pattern=None,
            stop_words=_STEMMED_STOP_WORDS,
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
        )
        self.question_matrix = self.vectorizer.fit_transform(self._retrieval_texts)

        self.type_classifier = QuestionClassify()
        self.known_types = sorted(set(self.types))

    # -- internal helpers -------------------------------------------------

    def _predict_type(self, user_question: str):
        """Return (predicted_type, confidence) using predict_proba when available."""
        model = self.type_classifier.model
        try:
            proba = model.predict_proba([user_question])[0]
            classes = model.classes_
            best_idx = proba.argmax()
            return classes[best_idx], float(proba[best_idx])
        except AttributeError:
            # Classifier has no predict_proba; fall back to a plain predict.
            return model.predict([user_question])[0], 1.0

    def _rank(self, user_vec, indices, top_k):
        if not indices:
            return []
        sub_matrix = self.question_matrix[indices]
        sims = cosine_similarity(user_vec, sub_matrix)[0]
        order = sims.argsort()[::-1][:top_k]
        return [(indices[i], float(sims[i])) for i in order]

    def _dedupe(self, ranked):
        """Collapse results that share the same answer text (scenario variants)."""
        seen = set()
        results = []
        for idx, score in ranked:
            item = self.qa_pairs[idx]
            key = tuple(item.get("answers", []))
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "question": _clean_display_question(item["question"]),
                "type": item.get("type", ""),
                "answers": item.get("answers", []),
                "score": score,
            })
        return results

    # -- public API ---------------------------------------------------------

    def search(self, user_question: str, top_k: int = 3):
        user_question = (user_question or "").strip()
        if not user_question:
            return "Empty question", [], False

        predicted_type, confidence = self._predict_type(user_question)
        user_vec = self.vectorizer.transform([_strip_boilerplate(user_question)])

        # 1) Try within the predicted category, but only trust the category
        #    filter if the classifier is reasonably confident.
        ranked = []
        if confidence >= MIN_TYPE_CONFIDENCE:
            type_indices = [i for i, t in enumerate(self.types) if t == predicted_type]
            ranked = [(i, s) for i, s in self._rank(user_vec, type_indices, top_k * 2) if s >= MIN_SCORE]

        # 2) If that found nothing, widen the search to the whole knowledge base.
        used_fallback = False
        if not ranked:
            all_indices = list(range(len(self.questions)))
            ranked = [(i, s) for i, s in self._rank(user_vec, all_indices, top_k * 2) if s >= MIN_SCORE]
            used_fallback = True

            if ranked:
                # The classifier's top guess is still useful signal here even
                # though it wasn't confident enough to hard-filter on above —
                # use it as a small tiebreaker so a near-tied but off-topic
                # keyword overlap doesn't win purely by coincidence (e.g. a
                # scholarship question narrowly losing to an unrelated
                # traffic-accident question that happens to share "cancelled").
                def _ranking_key(pair):
                    i, s = pair
                    boost = CLASSIFIER_TIEBREAK_BOOST if self.types[i] == predicted_type else 0.0
                    return s + boost

                ranked.sort(key=_ranking_key, reverse=True)

                # Don't pad the result out with unrelated categories just
                # because they cleared the bar too — keep only results that
                # share the best match's category, or are nearly as strong a
                # match (using the same boosted comparison for consistency).
                top_type = self.types[ranked[0][0]]
                top_boosted = _ranking_key(ranked[0])
                ranked = [
                    (i, s) for i, s in ranked
                    if self.types[i] == top_type or _ranking_key((i, s)) >= top_boosted - 0.05
                ]

        results = self._dedupe(ranked)[:top_k]
        return predicted_type, results, used_fallback, confidence

    def answer(self, user_question: str, top_k: int = 3):
        """High level convenience method returning a ready-to-display payload."""
        predicted_type, results, used_fallback, confidence = self.search(user_question, top_k=top_k)

        if results:
            # If we had to widen the search past the classifier's guess, the
            # matched result's own category is more trustworthy to display
            # than the classifier's shaky first guess.
            display_type = results[0]["type"] if used_fallback else predicted_type
            return {
                "type": display_type,
                "confident": not used_fallback,
                "results": results,
                "note": None,
            }

        # Nothing matched closely enough anywhere. If the classifier itself
        # wasn't even meaningfully more confident than a random guess, the
        # question is likely outside what this knowledge base covers at all —
        # don't dress that up as category-specific legal advice.
        if confidence < MIN_TYPE_CONFIDENCE:
            return {
                "type": None,
                "confident": False,
                "results": [],
                "note": (
                    "I couldn't match this to a legal topic in my knowledge base "
                    "(marriage & family, labour, traffic accidents, debt, criminal defence, "
                    "property, consumer complaints, cybercrime, medical negligence, "
                    "housing/tenancy, or education disputes). Try rephrasing your question "
                    "with more detail."
                ),
                "generic_advice": [],
            }

        fallback_advice = GENERIC_ADVICE.get(predicted_type)
        return {
            "type": predicted_type,
            "confident": False,
            "results": [],
            "note": (
                "I couldn't find a closely matching question in the knowledge base, "
                "but here is general guidance for this category."
                if fallback_advice else
                "I couldn't find a matching question in the knowledge base. "
                "Try rephrasing, or consult a lawyer for advice specific to your situation."
            ),
            "generic_advice": fallback_advice or [],
        }


def main():
    assistant = QAAssistant()

    print("Legal QA Assistant (blank line to exit)\n")

    while True:
        q = input("Your question: ").strip()

        if not q:
            print("Exiting.")
            break

        payload = assistant.answer(q)

        print("\nPredicted type:", payload["type"])

        if payload["results"]:
            label = "Top answers:" if payload["confident"] else "Closest matches found (broadened search):"
            print(label)
            for i, item in enumerate(payload["results"], 1):
                print(f"\nCandidate {i} (score={item['score']:.3f}, type={item['type']})")
                print(f"Matched question: {item['question']}")
                for ans in item["answers"]:
                    print("-", ans)
        else:
            print(payload["note"])
            for ans in payload.get("generic_advice", []):
                print("-", ans)

        print("-" * 60)


if __name__ == "__main__":
    main()
