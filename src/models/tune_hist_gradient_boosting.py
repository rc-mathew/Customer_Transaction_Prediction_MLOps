import json
from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

from src.data.load_data import load_training_data
from src.data.split_data import split_dataset
from src.features.preprocessing import build_tree_preprocessor


RANDOM_STATE = 42


def evaluate_model(model, X_valid, y_valid):
    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)[:, 1]

    return {
        "accuracy": accuracy_score(y_valid, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_valid, predictions
        ),
        "precision": precision_score(
            y_valid, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_valid, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_valid, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_valid, probabilities
        ),
        "pr_auc": average_precision_score(
            y_valid, probabilities
        ),
    }


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
    # PIPELINE
    # -----------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_tree_preprocessor(),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # -----------------------------------------------------
    # CONTROLLED SEARCH SPACE
    # -----------------------------------------------------

    param_distributions = {
        "classifier__learning_rate": [
            0.03,
            0.05,
            0.08,
            0.10,
        ],
        "classifier__max_iter": [
            100,
            150,
            200,
            250,
        ],
        "classifier__max_leaf_nodes": [
            15,
            31,
            63,
        ],
        "classifier__min_samples_leaf": [
            20,
            40,
            60,
        ],
        "classifier__l2_regularization": [
            0.0,
            0.5,
            1.0,
            2.0,
        ],
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=12,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        refit=True,
        return_train_score=False,
    )

    print("\nStarting hyperparameter tuning...")
    print("Optimization metric: ROC-AUC")
    print("Search iterations: 12")
    print("Cross-validation folds: 3")

    search.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # BEST PARAMETERS
    # -----------------------------------------------------

    print("\nBEST HYPERPARAMETERS")
    print("=" * 60)

    for parameter, value in search.best_params_.items():
        print(f"{parameter}: {value}")

    print(
        f"\nBest CV ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    # -----------------------------------------------------
    # VALIDATION EVALUATION
    # -----------------------------------------------------

    best_model = search.best_estimator_

    validation_metrics = evaluate_model(
        best_model,
        X_valid,
        y_valid,
    )
        # -----------------------------------------------------
    # SAVE PRODUCTION MODEL
    # -----------------------------------------------------

    models_dir = Path("models")
    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = models_dir / "customer_transaction_model.joblib"

    joblib.dump(
        best_model,
        model_path,
    )

    print(
        f"\nTrained model saved to: "
        f"{model_path}"
    )

    # -----------------------------------------------------
    # VALIDATION RESULTS
    # -----------------------------------------------------

    print("\nTUNED MODEL VALIDATION RESULTS")
    print("=" * 60)

    for metric, value in validation_metrics.items():
        print(
            f"{metric:22s}: {value:.4f}"
        )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    reports_dir = Path("reports/metrics")
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "best_cv_roc_auc": float(
            search.best_score_
        ),
        "best_params": search.best_params_,
        "validation_metrics": {
            key: float(value)
            for key, value
            in validation_metrics.items()
        },
    }

    output_path = (
        reports_dir
        / "hist_gradient_tuning_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nTuning results saved to: "
        f"{output_path}"
    )

    # -----------------------------------------------------
    # TOP SEARCH RESULTS
    # -----------------------------------------------------

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    top_results = (
        cv_results[
            [
                "rank_test_score",
                "mean_test_score",
                "std_test_score",
                "params",
            ]
        ]
        .sort_values(
            "rank_test_score"
        )
        .head(5)
    )

    print("\nTOP 5 PARAMETER CONFIGURATIONS")
    print("=" * 80)

    print(
        top_results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()