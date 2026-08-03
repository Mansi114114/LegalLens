#!/usr/bin/env python3
# coding: utf-8
"""
Legal QA Assistant — retrieval engine.

Pipeline:
  1. Reject small talk / non-legal input up front, so "hello" gets a
     greeting instead of a fabricated case match.
  2. Score the whole knowledge base in one pass: TF-IDF cosine +
     FastEmbed (MiniLM) cosine, blended on their *raw* scales.
  3. Decide whether the question is in scope at all, using absolute
     similarity thresholds. Out of scope => say so, retrieve nothing.
  4. Rank in-scope questions, nudging results in the classifier's
     predicted category upward (a soft preference, not a hard filter).
  5. De-duplicate near-identical "(Scenario N)" variants so the same
     answer isn't shown three times.
"""

import os
import re
import json
import threading

import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from fastembed import TextEmbedding


from question_classify import QuestionClassify

_stemmer = PorterStemmer()
_TOKEN_RE = re.compile(r"[a-zA-Z]+")

# --- Embedding model -----------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# FastEmbed defaults to /tmp, which is ephemeral on most PaaS hosts — so the
# ~87 MB ONNX model gets re-downloaded on every cold start, before the first
# user can get an answer. Cache it inside the project instead, so a build
# step can warm it once and the running container just reads it off disk.
EMBEDDING_CACHE_DIR = os.environ.get(
    "FASTEMBED_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fastembed_cache"),
)


def get_embedding_model():
    """Build the FastEmbed model, downloading to the shared cache if needed.

    Used by both the server and build_embeddings.py so the offline embedding
    build and the runtime query encoder can never drift onto different
    models (which would silently make every cosine score meaningless).
    """
    return TextEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_dir=EMBEDDING_CACHE_DIR,
    )

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

# --- Scoring -------------------------------------------------------------
#
# IMPORTANT: these thresholds only mean anything because the two similarity
# signals are blended on their RAW cosine scales. An earlier version min-max
# normalised each signal before blending, which rescales the best candidate
# in *every* search to exactly 1.0 no matter how bad it actually is. That
# made the accept threshold unreachable-by-construction, so literally any
# input — "hello", "pizza recipe" — came back as a confident case match.
# Never re-introduce per-query normalisation here.
TFIDF_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6

# Minimum blended (raw) score to show a match at all.
MIN_SCORE = 0.40

# Scope gate. A question is only answered if it is genuinely close to
# something in the knowledge base. The numbers below are measured, not
# guessed: over a probe set of off-topic input ("hello", "pizza recipe",
# "how to learn python") the best semantic cosine peaked at 0.36, while
# real legal questions bottomed out at 0.66. 0.45 sits in that gap.
IN_SCOPE_SEMANTIC = 0.45

# TF-IDF alone can rescue an oddly-phrased question the embedding misses,
# but only on strong literal overlap. Short off-topic queries pick up
# surprisingly high TF-IDF cosine from a single shared rare word ("who won
# the world cup" scored 0.29), so this bar is set above that noise floor.
IN_SCOPE_TFIDF = 0.55

# Above this, the top match is close enough to present without hedging.
CONFIDENT_SEMANTIC = 0.60

# When the top two candidates land within this much of each other, the gap
# between them isn't meaningful signal — they're competing readings of the
# same question ("family member sold my property" vs "someone forged
# documents to sell my property"). Ranking one above the other implies a
# confidence the scores don't support, so the user is asked to pick instead.
AMBIGUITY_MARGIN = 0.08

# Minimum classifier confidence before its predicted category is allowed to
# influence ranking at all. With 13 categories a uniform guess is ~0.077, so
# anything near that carries no information.
MIN_TYPE_CONFIDENCE = 0.22

# The predicted category is a soft preference, not a filter: matches in it
# get a small ranking bonus so a coincidental keyword overlap in the wrong
# category doesn't win a near-tie outright.
CATEGORY_BOOST = 0.05

# Small talk and pleasantries. These deserve a real reply rather than being
# pushed through retrieval and dressed up as legal guidance.
GREETING_RE = re.compile(
    r"^[\s!.,?]*("
    r"hi|hii+|hey|hello+|yo|hola|namaste|greetings|sup|wass?up|"
    r"good\s+(morning|afternoon|evening|night)|"
    r"how\s+(are|r)\s+(you|u)|how'?s\s+it\s+going|what'?s\s+up|"
    r"thanks?|thank\s+you|thx|ty|nice|cool|ok(ay)?|yes|no|yep|nope|"
    r"bye|goodbye|see\s+(you|ya)|good\s?bye|"
    r"who\s+are\s+you|what\s+(can\s+you\s+do|are\s+you)|help"
    r")[\s!.,?]*$",
    re.IGNORECASE,
)

GREETING_REPLY = (
    "Hello. I'm LegalLens — I look up general legal information for everyday "
    "situations in India. Describe what happened in a sentence or two and I'll "
    "find the closest guidance I have. For example: \"my landlord is refusing to "
    "return my security deposit\" or \"I was fired without notice\"."
)


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


def _known_topics_sentence(types) -> str:
    """Render the covered categories as a readable 'a, b and c.' list.

    Derived from the knowledge base rather than hard-coded, so adding a
    category to qa_pairs.json can't leave this message out of date.
    """
    topics = [t.lower() for t in types if t]
    if not topics:
        return "a range of everyday legal topics."
    if len(topics) == 1:
        return f"{topics[0]}."
    return ", ".join(topics[:-1]) + f" and {topics[-1]}."


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
    "Inheritance & succession": [
        "Gather the will (if any), death certificate, and property/title documents.",
        "Consult a lawyer about obtaining a succession or legal heir certificate, or probate of the will.",
    ],
    "Insurance claims": [
        "Keep the policy document, claim forms, and all correspondence with the insurer.",
        "If the claim is wrongly rejected, escalate to the insurer's grievance officer, then the Insurance Ombudsman.",
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

        # Pre-computed (already L2-normalised) MiniLM embeddings, one row per
        # entry in qa_pairs.json. Built offline by build_embeddings.py so the
        # server never has to embed the corpus at startup.
        emb_path = os.path.join(cur, "data", "question_embeddings.npy")
        if not os.path.exists(emb_path):
            raise FileNotFoundError(
                f"Embeddings file not found: {emb_path}\n"
                f"Please run build_embeddings.py first."
            )
        self.question_embeddings = np.load(emb_path)

        # A stale .npy (qa_pairs.json edited without rebuilding) would silently
        # pair questions with the wrong vectors and quietly poison every
        # result, so fail loudly instead.
        if len(self.question_embeddings) != len(self.qa_pairs):
            raise ValueError(
                f"Embeddings are stale: {len(self.question_embeddings)} vectors "
                f"for {len(self.qa_pairs)} QA pairs. Re-run build_embeddings.py."
            )

        self.embedding_model = None
        self._embedding_lock = threading.Lock()
        self.type_classifier = QuestionClassify()
        self.known_types = sorted(set(self.types))

        # Boolean mask per category, used for the soft ranking preference.
        self._types_arr = np.array(self.types)

    def load_embeddings(self):
        """Lazily construct the ONNX embedding model on first use.

        Deferred rather than done in __init__ so the web process boots (and
        starts serving the landing page) without paying the model load, which
        matters on small containers with a startup timeout.

        Locked because gunicorn is commonly run with threaded workers: two
        concurrent first-requests would otherwise both build a model, briefly
        holding two copies in memory — enough to OOM a 512 MB instance.
        """
        if self.embedding_model is None:
            with self._embedding_lock:
                # Re-check inside the lock: another thread may have finished
                # building it while this one was waiting.
                if self.embedding_model is None:
                    self.embedding_model = get_embedding_model()
        return self.embedding_model

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

    def _score_all(self, user_question, user_vec):
        """Score every knowledge-base entry against the question.

        Returns (tfidf_scores, semantic_scores, blended_scores) as raw
        cosine similarities — deliberately NOT rescaled per query, so the
        thresholds above stay comparable across different questions.

        The corpus is a couple of hundred rows, so scoring all of it costs
        one small mat-vec; there's no reason to pre-filter by category and
        then need a second pass when the filter comes up empty.
        """
        tfidf_scores = cosine_similarity(user_vec, self.question_matrix)[0]

        model = self.load_embeddings()
        query_embedding = next(model.embed([user_question]))
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        semantic_scores = self.question_embeddings @ query_embedding

        # Cosine can go negative for genuinely opposed text; clamp so a
        # negative semantic score can't drag a blend below zero.
        semantic_scores = np.clip(semantic_scores, 0.0, None)

        blended = TFIDF_WEIGHT * tfidf_scores + SEMANTIC_WEIGHT * semantic_scores
        return tfidf_scores, semantic_scores, blended

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
                # Optional structured citations on a QA pair, e.g.
                #   "acts": ["Consumer Protection Act, 2019"],
                #   "sections": ["S. 35"]
                # Passed straight through when present so the knowledge base
                # can grow citations without touching the retrieval code.
                "acts": item.get("acts", []),
                "sections": item.get("sections", []),
                "score": score,
            })
        return results

    # -- public API ---------------------------------------------------------

    def search(self, user_question: str, top_k: int = 3):
        """Retrieve matches for a question already known to be in scope.

        Returns (predicted_type, results, in_scope, confidence, best_semantic).
        """
        predicted_type, confidence = self._predict_type(user_question)
        user_vec = self.vectorizer.transform([_strip_boilerplate(user_question)])

        tfidf_scores, semantic_scores, blended = self._score_all(user_question, user_vec)

        best_semantic = float(semantic_scores.max())
        best_tfidf = float(tfidf_scores.max())

        # Scope gate. Nothing in the knowledge base is even loosely related,
        # so there is no honest match to return — and inventing one is how
        # "hello" used to come back as a consumer complaint.
        if best_semantic < IN_SCOPE_SEMANTIC and best_tfidf < IN_SCOPE_TFIDF:
            return predicted_type, [], False, confidence, best_semantic

        # Soft category preference: nudge, don't filter. If the classifier is
        # no better than a coin flip across 13 classes, ignore it entirely.
        ranking = blended.copy()
        if confidence >= MIN_TYPE_CONFIDENCE:
            ranking = ranking + CATEGORY_BOOST * (self._types_arr == predicted_type)

        order = np.argsort(ranking)[::-1][: top_k * 3]
        ranked = [(int(i), float(blended[i])) for i in order if blended[i] >= MIN_SCORE]

        results = self._dedupe(ranked)[:top_k]

        # `ranked` is ordered by the boosted score, but each result carries its
        # raw score for display. Those two can disagree, which would render a
        # list whose visible numbers aren't descending. Keep the boost's pick
        # of the primary result, and order the remainder by what's shown.
        #
        # The primary itself can't be "wrong" this way: overtaking on a 0.05
        # boost requires a raw gap under 0.05, and anything inside
        # AMBIGUITY_MARGIN is handed to the user as a choice rather than
        # ranked at all.
        if len(results) > 2:
            results[1:] = sorted(results[1:], key=lambda r: r["score"], reverse=True)

        return predicted_type, results, True, confidence, best_semantic

    def answer(self, user_question: str, top_k: int = 3):
        """High level convenience method returning a ready-to-display payload."""
        user_question = (user_question or "").strip()

        if not user_question:
            return {
                "kind": "empty",
                "type": None,
                "confident": False,
                "results": [],
                "note": "Please describe your legal issue.",
                "generic_advice": [],
            }

        # Small talk never reaches retrieval.
        if GREETING_RE.match(user_question):
            return {
                "kind": "greeting",
                "type": None,
                "confident": False,
                "results": [],
                "note": GREETING_REPLY,
                "generic_advice": [],
            }

        predicted_type, results, in_scope, confidence, best_semantic = self.search(
            user_question, top_k=top_k
        )

        if not in_scope:
            return {
                "kind": "out_of_scope",
                "type": None,
                "confident": False,
                "results": [],
                "note": (
                    "That doesn't look like a legal question I can help with. I cover "
                    + _known_topics_sentence(self.known_types)
                    + " Describe your situation in a sentence or two and I'll look for "
                    "the closest guidance I have."
                ),
                "generic_advice": [],
            }

        if len(results) >= 2 and (results[0]["score"] - results[1]["score"]) <= AMBIGUITY_MARGIN:
            # Too close to call. Hand the decision to the user, who knows
            # which reading fits their situation, instead of presenting an
            # arbitrary winner as though the ranking were decisive.
            options = results[:2]
            shared_type = (
                options[0]["type"] if options[0]["type"] == options[1]["type"] else None
            )
            return {
                "kind": "choice",
                "type": shared_type,
                "confident": False,
                "results": options,
                "note": (
                    "Two entries match your question about equally well. Pick whichever "
                    "describes your situation more closely."
                ),
                "generic_advice": [],
            }

        if results:
            # Display the matched entry's own category rather than the
            # classifier's guess — the match is the stronger evidence.
            return {
                "kind": "match",
                "type": results[0]["type"] or predicted_type,
                "confident": best_semantic >= CONFIDENT_SEMANTIC,
                "results": results,
                "note": (
                    None if best_semantic >= CONFIDENT_SEMANTIC else
                    "No close match — these are the nearest topics I have, so treat "
                    "them as background rather than an answer to your exact situation."
                ),
                "generic_advice": [],
            }

        # In scope (related to something we cover) but nothing cleared the bar
        # for showing a specific answer. Offer category-level guidance instead.
        fallback_advice = GENERIC_ADVICE.get(predicted_type) if confidence >= MIN_TYPE_CONFIDENCE else None
        return {
            "kind": "no_match",
            "type": predicted_type if fallback_advice else None,
            "confident": False,
            "results": [],
            "note": (
                "I couldn't find a closely matching question in the knowledge base, "
                "but here is general guidance for this category."
                if fallback_advice else
                "I couldn't find a matching question in the knowledge base. "
                "Try rephrasing with more detail about what happened, or consult a "
                "lawyer for advice specific to your situation."
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

        print(f"\n[{payload['kind']}] type:", payload["type"])

        if payload["results"]:
            label = "Top answers:" if payload["confident"] else "Nearest topics (no close match):"
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