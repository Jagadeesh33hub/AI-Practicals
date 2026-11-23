"""
Practical 8: Mini AI Application - Sentiment Classifier with Explainability & Ethics Report
Aim: Build a customer feedback sentiment classifier, generate word-level explainability
     attributions for every prediction, and run an automated bias/fairness check.
"""

import math
import re
from collections import Counter


TRAINING_DATA = [
    ("The product quality is amazing and delivery was fast", "Positive"),
    ("Excellent customer support and super easy setup", "Positive"),
    ("Love the battery life and sleek modern design", "Positive"),
    ("Great value for money works flawlessly every day", "Positive"),
    ("Very happy with this purchase highly recommended", "Positive"),
    ("Terrible build quality broke after two days", "Negative"),
    ("Worst customer service ever complete waste of money", "Negative"),
    ("Very slow performance and frequent crashes", "Negative"),
    ("Disappointed with the late shipping and damaged box", "Negative"),
    ("Poor battery life and frustrating user interface", "Negative"),
]


class ExplainableNaiveBayesClassifier:
    """Interpretable Multinomial Naive Bayes with log-odds word attribution."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_counts: Counter = Counter()
        self.word_counts: dict[str, Counter] = {"Positive": Counter(), "Negative": Counter()}
        self.vocab: set[str] = set()

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]+", text.lower())

    def fit(self, dataset: list[tuple[str, str]]) -> None:
        for text, label in dataset:
            self.class_counts[label] += 1
            tokens = self.tokenize(text)
            self.word_counts[label].update(tokens)
            self.vocab.update(tokens)

    def predict_with_explanation(self, text: str) -> dict:
        tokens = self.tokenize(text)
        vocab_size = len(self.vocab)
        total_docs = sum(self.class_counts.values())

        log_probs = {}
        for label in ("Positive", "Negative"):
            prior = math.log(self.class_counts[label] / total_docs)
            total_words = sum(self.word_counts[label].values())
            score = prior
            for tok in tokens:
                likelihood = (self.word_counts[label][tok] + self.alpha) / (total_words + self.alpha * vocab_size)
                score += math.log(likelihood)
            log_probs[label] = score

        max_lp = max(log_probs.values())
        exp_pos = math.exp(log_probs["Positive"] - max_lp)
        exp_neg = math.exp(log_probs["Negative"] - max_lp)
        prob_pos = exp_pos / (exp_pos + exp_neg)

        total_pos_words = sum(self.word_counts["Positive"].values())
        total_neg_words = sum(self.word_counts["Negative"].values())
        attributions = []
        for tok in dict.fromkeys(tokens):
            p_w_pos = (self.word_counts["Positive"][tok] + self.alpha) / (total_pos_words + self.alpha * vocab_size)
            p_w_neg = (self.word_counts["Negative"][tok] + self.alpha) / (total_neg_words + self.alpha * vocab_size)
            weight = math.log(p_w_pos / p_w_neg)
            attributions.append((tok, round(weight, 3)))

        attributions.sort(key=lambda x: abs(x[1]), reverse=True)
        prediction = "Positive" if prob_pos >= 0.5 else "Negative"
        confidence = prob_pos if prediction == "Positive" else (1.0 - prob_pos)

        return {
            "text": text,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "top_features": attributions[:4],
        }


def run_ethics_and_fairness_audit(model: ExplainableNaiveBayesClassifier) -> None:
    print("\n" + "=" * 74)
    print("ETHICS, FAIRNESS & EXPLAINABILITY AUDIT REPORT")
    print("=" * 74)

    template_a = "Customer Rahul said the delivery was fast and amazing"
    template_b = "Customer Maria said the delivery was fast and amazing"
    res_a = model.predict_with_explanation(template_a)
    res_b = model.predict_with_explanation(template_b)
    diff = abs(res_a["confidence"] - res_b["confidence"])

    print("1. Counterfactual Fairness Test (Name Perturbation):")
    print(f"   Prompt A: '{template_a}' -> {res_a['prediction']} ({res_a['confidence']}%)")
    print(f"   Prompt B: '{template_b}' -> {res_b['prediction']} ({res_b['confidence']}%)")
    print(f"   Confidence Drift: {diff:.2f}% (Status: {'PASS - Invariant' if diff < 0.01 else 'REVIEW'})")
    print("2. Transparency & Explainability:")
    print("   Every decision exposes exact token log-odds weights so human reviewers")
    print("   can verify decisions and detect spurious correlations.")
    print("3. Data Privacy & Limitations:")
    print("   - Small training corpus may misclassify sarcasm or negation ('not good').")
    print("   - PII (names/emails) should be anonymized before logging user feedback.")


if __name__ == "__main__":
    clf = ExplainableNaiveBayesClassifier()
    clf.fit(TRAINING_DATA)

    test_reviews = [
        "Amazing battery life and super fast delivery",
        "Terrible support and slow crashes every day",
        "Great design but shipping was damaged",
    ]

    print("=" * 74)
    print("MINI AI APPLICATION: EXPLAINABLE SENTIMENT CLASSIFIER")
    print("=" * 74)
    for review in test_reviews:
        out = clf.predict_with_explanation(review)
        print(f"\nReview     : \"{out['text']}\"")
        print(f"Prediction : {out['prediction']} (Confidence: {out['confidence']}%)")
        print(f"Top Drivers: {out['top_features']} (+ favors Positive, - favors Negative)")

    run_ethics_and_fairness_audit(clf)
