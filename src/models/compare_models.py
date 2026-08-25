import time
from pathlib import Path

import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src.data.load_data import load_training_data
from src.data.split_data import split_dataset
from src.features.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
)


RANDOM_STATE = 42


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
):
    print(f"\nTraining {name}...")

    start = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    start = time.perf_counter()

    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    inference_time = time.perf_counter() - start

    return {
        "Model": name,
        "Accuracy": accuracy_score(
            y_valid,
            predictions,
        ),
        "Balanced Accuracy": balanced_accuracy_score(
            y_valid,
            predictions,
        ),
        "Precision": precision_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_valid,
            probabilities,
        ),
        "PR-AUC": average_precision_score(
            y_valid,
            probabilities,
        ),
        "Training Time (s)": training_time,
        "Inference Time (s)": inference_time,
    }


def main():

    print("\nLoading training data...")

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

    models = {
        "Logistic Regression": Pipeline(
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
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Gaussian Naive Bayes": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_linear_preprocessor(),
                ),
                (
                    "classifier",
                    GaussianNB(),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_tree_preprocessor(),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "HistGradientBoosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_tree_preprocessor(),
                ),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=2.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    results = []

    for name, model in models.items():
        results.append(
            evaluate_model(
                name,
                model,
                X_train,
                y_train,
                X_valid,
                y_valid,
            )
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC",
        ascending=False,
    )

    print("\nMODEL COMPARISON")
    print("=" * 120)

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )

    reports_dir = Path("reports/metrics")

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        reports_dir
        / "model_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nModel comparison saved to: "
        f"{output_path}"
    )

    best_model = results_df.iloc[0]

    print("\nBEST MODEL BY ROC-AUC")
    print("=" * 50)

    print(
        f"{best_model['Model']} "
        f"(ROC-AUC = "
        f"{best_model['ROC-AUC']:.4f}, "
        f"PR-AUC = "
        f"{best_model['PR-AUC']:.4f})"
    )


if __name__ == "__main__":
    main()