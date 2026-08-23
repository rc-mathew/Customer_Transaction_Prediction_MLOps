import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data.load_data import load_training_data
from src.data.split_data import split_dataset
from src.features.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
)


def main():

    print("\nLoading dataset...")

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

    # -----------------------------------------------------
    # MODELS
    # -----------------------------------------------------

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

    hist_gradient_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_tree_preprocessor(),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=200,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )

    models = {
        "Logistic Regression": logistic_model,
        "HistGradientBoosting": hist_gradient_model,
    }

    # -----------------------------------------------------
    # CROSS VALIDATION
    # -----------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
    }

    results = []

    for model_name, model in models.items():

        print(
            f"\nRunning 5-fold cross-validation for "
            f"{model_name}..."
        )

        scores = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
        )

        model_result = {
            "Model": model_name,
            "ROC-AUC Mean":
                scores["test_roc_auc"].mean(),
            "ROC-AUC Std":
                scores["test_roc_auc"].std(),
            "PR-AUC Mean":
                scores["test_pr_auc"].mean(),
            "PR-AUC Std":
                scores["test_pr_auc"].std(),
            "Accuracy Mean":
                scores["test_accuracy"].mean(),
            "Balanced Accuracy Mean":
                scores["test_balanced_accuracy"].mean(),
            "F1 Mean":
                scores["test_f1"].mean(),
        }

        results.append(model_result)

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC-AUC Mean",
        ascending=False,
    )

    print("\nCROSS-VALIDATION MODEL COMPARISON")
    print("=" * 110)

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )

    best_model = results_df.iloc[0]

    print("\nBest cross-validated model:")

    print(
        f"{best_model['Model']} | "
        f"ROC-AUC = "
        f"{best_model['ROC-AUC Mean']:.4f} "
        f"+/- {best_model['ROC-AUC Std']:.4f} | "
        f"PR-AUC = "
        f"{best_model['PR-AUC Mean']:.4f}"
    )


if __name__ == "__main__":
    main()