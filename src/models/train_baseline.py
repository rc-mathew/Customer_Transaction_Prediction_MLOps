import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline

from src.data.load_data import load_training_data
from src.data.split_data import split_dataset
from src.features.preprocessing import build_linear_preprocessor


def evaluate_model(name, model, X_train, y_train, X_valid, y_valid):
    """Train and evaluate a binary classification model."""

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_valid, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_valid, predictions
        ),
        "Precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_valid, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_valid, probabilities
        ),
        "PR-AUC": average_precision_score(
            y_valid, probabilities
        ),
    }

    return results


def main():

    df = load_training_data()

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        id_train,
        id_valid,
        id_test,
    ) = split_dataset(df)

    dummy_model = DummyClassifier(
        strategy="prior"
    )

    logistic_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_linear_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    results = []

    results.append(
        evaluate_model(
            "Dummy Classifier",
            dummy_model,
            X_train,
            y_train,
            X_valid,
            y_valid,
        )
    )

    results.append(
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_train,
            y_train,
            X_valid,
            y_valid,
        )
    )

    results_df = pd.DataFrame(results)

    print("\nBASELINE MODEL RESULTS")
    print("=" * 90)

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )

    print("\nBest model by ROC-AUC:")

    best_model = results_df.loc[
        results_df["ROC-AUC"].idxmax()
    ]

    print(
        f"{best_model['Model']} "
        f"(ROC-AUC = {best_model['ROC-AUC']:.4f})"
    )


if __name__ == "__main__":
    main()