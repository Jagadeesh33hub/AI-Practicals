"""
Practical 12: Comparison of Supervised AI Learning Models on a Toy Dataset
Aim: Train and evaluate k-Nearest Neighbors (k-NN), Decision Tree, Gaussian Naive Bayes,
     and Perceptron classifiers on the Iris dataset and compare accuracy, precision,
     recall, and F1-score.
"""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier


def compare_classifiers() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    models = {
        "k-NN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Perceptron": make_pipeline(StandardScaler(), Perceptron(max_iter=1000, random_state=42)),
    }

    print("=" * 78)
    print("SUPERVISED AI MODELS COMPARISON ON IRIS TOY DATASET")
    print("=" * 78)
    print(f"Dataset Samples : {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})")
    print(f"Features        : {iris.feature_names}")
    print(f"Target Classes  : {list(iris.target_names)}\n")

    print(f"{'Model':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score'}")
    print("-" * 78)

    best_model_name = ""
    best_preds = None
    best_acc = -1.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        print(f"{name:<22} | {acc*100:>8.2f}% | {prec*100:>8.2f}% | {rec*100:>8.2f}% | {f1*100:>8.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_preds = y_pred

    print("-" * 78)
    print(f"\nDetailed Classification Report for Top Performing Model ({best_model_name}):")
    print(classification_report(y_test, best_preds, target_names=iris.target_names))


if __name__ == "__main__":
    compare_classifiers()
