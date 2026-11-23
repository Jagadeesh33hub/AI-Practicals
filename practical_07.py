"""
Practical 7: Bayesian Network & Probabilistic Inference
Aim: Model a diagnostic Bayesian Network (Flu, Allergies -> Fever, Cough) with
     Conditional Probability Tables (CPTs), perform exact inference by enumeration,
     and compare with scikit-learn's CategoricalNB probabilistic classifier.
"""

from itertools import product


class DiscreteBayesianNetwork:
    """
    Causal DAG:
        Flu (F)        Allergies (A)
          \   \          /
           v   v        v
         Fever(V)     Cough(C)
    """

    def __init__(self) -> None:
        self.p_flu = {True: 0.15, False: 0.85}
        self.p_allergy = {True: 0.25, False: 0.75}

        self.p_fever_given_flu = {
            True: {True: 0.85, False: 0.15},
            False: {True: 0.05, False: 0.95},
        }

        self.p_cough_given_flu_allergy = {
            (True, True): {True: 0.92, False: 0.08},
            (True, False): {True: 0.75, False: 0.25},
            (False, True): {True: 0.65, False: 0.35},
            (False, False): {True: 0.08, False: 0.92},
        }

    def joint_probability(self, flu: bool, allergy: bool, fever: bool, cough: bool) -> float:
        return (
            self.p_flu[flu]
            * self.p_allergy[allergy]
            * self.p_fever_given_flu[flu][fever]
            * self.p_cough_given_flu_allergy[(flu, allergy)][cough]
        )

    def infer_posterior(self, evidence: dict[str, bool]) -> dict[str, float]:
        """Exact probabilistic inference via enumeration over all world states."""
        matching_states = []
        total_evidence_prob = 0.0

        for flu, allergy, fever, cough in product([True, False], repeat=4):
            state = {"Flu": flu, "Allergy": allergy, "Fever": fever, "Cough": cough}
            if all(state[k] == v for k, v in evidence.items()):
                prob = self.joint_probability(flu, allergy, fever, cough)
                matching_states.append((state, prob))
                total_evidence_prob += prob

        p_flu_given_e = sum(p for s, p in matching_states if s["Flu"]) / total_evidence_prob
        p_allergy_given_e = sum(p for s, p in matching_states if s["Allergy"]) / total_evidence_prob
        return {
            "P(Flu=True | Evidence)": round(p_flu_given_e, 4),
            "P(Allergy=True | Evidence)": round(p_allergy_given_e, 4),
        }


def run_sklearn_probabilistic_classifier() -> None:
    try:
        import numpy as np
        from sklearn.naive_bayes import CategoricalNB
    except ImportError:
        print("\n[Note] scikit-learn not installed; skipping CategoricalNB comparison.")
        return

    X = np.array([
        [1, 1], [1, 1], [1, 0], [1, 1], [0, 1],
        [0, 0], [0, 0], [0, 1], [0, 0], [1, 0],
        [0, 0], [0, 1], [1, 1], [0, 0], [0, 0],
    ])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0])

    clf = CategoricalNB()
    clf.fit(X, y)

    sample = np.array([[1, 1]])
    probs = clf.predict_proba(sample)[0]
    print("\n" + "-" * 72)
    print("SCIKIT-LEARN CATEGORICAL NAIVE BAYES CLASSIFIER CHECK")
    print("-" * 72)
    print("Input Evidence: Fever=1, Cough=1")
    print(f"Predicted P(Flu=False) = {probs[0]:.4f} | P(Flu=True) = {probs[1]:.4f}")


if __name__ == "__main__":
    bn = DiscreteBayesianNetwork()

    print("=" * 72)
    print("BAYESIAN NETWORK DIAGNOSTIC INFERENCE (EXACT ENUMERATION)")
    print("=" * 72)

    queries = [
        {"Fever": True, "Cough": True},
        {"Fever": False, "Cough": True},
        {"Fever": True, "Cough": False},
        {"Fever": False, "Cough": False},
    ]

    for ev in queries:
        posteriors = bn.infer_posterior(ev)
        print(f"Evidence: {str(ev):<32} -> {posteriors}")

    run_sklearn_probabilistic_classifier()
