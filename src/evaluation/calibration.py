"""
Probability calibration for the Customer Transaction Prediction project.

Compares:
1. Uncalibrated tuned HistGradientBoosting model
2. Sigmoid calibration
3. Isotonic calibration

Outputs:
- reports/metrics/calibration_results.csv
- reports/figures/calibration_curve.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.load_data import load_training_data
from src.data.split_data import split_dataset
from src.features.preprocessing import build_tree_preprocessor


RANDOM_STATE = 42


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUNING_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "hist_gradient_tuning_results.json"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

CALIBRATION_RESULTS_PATH = (
    METRICS_DIR
    / "calibration_results.csv"
)

CALIBRATION_FIGURE_PATH = (
    FIGURES_DIR
    / "calibration_curve.png"
)


# ---------------------------------------------------------
# LOAD TUNED PARAMETERS
# ---------------------------------------------------------

def load_best_parameters():
    """
    Load best HistGradientBoosting parameters
    generated during hyperparameter tuning.
    """

    if not TUNING_RESULTS_PATH.exists():

        raise FileNotFoundError(
            "Hyperparameter tuning results were not found at: "
            f"{TUNING_RESULTS_PATH}\n"
            "Run the tuning script before calibration."
        )

    with open(
        TUNING_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        tuning_results = json.load(file)

    return tuning_results["best_params"]


# ---------------------------------------------------------
# BUILD TUNED PIPELINE
# ---------------------------------------------------------

def build_tuned_model(best_params):
    """
    Build the tuned HistGradientBoosting pipeline.
    """

    model = Pipeline(
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

    # best_params contains names such as:
    # classifier__learning_rate
    # classifier__max_iter
    # classifier__max_leaf_nodes
    model.set_params(
        **best_params
    )

    return model


# ---------------------------------------------------------
# PROBABILITY METRICS
# ---------------------------------------------------------

def evaluate_probabilities(
    model_name,
    y_true,
    probabilities,
):
    """
    Calculate probability-quality metrics.
    """

    return {
        "Model": model_name,

        "ROC-AUC":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "PR-AUC":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "Brier Score":
            brier_score_loss(
                y_true,
                probabilities,
            ),

        "Log Loss":
            log_loss(
                y_true,
                probabilities,
            ),
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("PROBABILITY CALIBRATION")
    print("=" * 70)

    # Create output folders
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    print("\nLoading dataset...")

    df = load_training_data()

    # -----------------------------------------------------
    # SPLIT DATA
    # -----------------------------------------------------

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

    print(
        "\nTraining data:",
        X_train.shape,
    )

    print(
        "Validation data:",
        X_valid.shape,
    )

    print(
        "Test data:",
        X_test.shape,
    )

    # -----------------------------------------------------
    # LOAD BEST PARAMETERS
    # -----------------------------------------------------

    best_params = load_best_parameters()

    print("\nLoaded tuned parameters:")

    for key, value in best_params.items():
        print(
            f"{key}: {value}"
        )

    # -----------------------------------------------------
    # UNCALIBRATED MODEL
    # -----------------------------------------------------

    print(
        "\nTraining uncalibrated tuned model..."
    )

    uncalibrated_model = (
        build_tuned_model(
            best_params
        )
    )

    uncalibrated_model.fit(
        X_train,
        y_train,
    )

    uncalibrated_probabilities = (
        uncalibrated_model.predict_proba(
            X_valid
        )[:, 1]
    )

    # -----------------------------------------------------
    # SIGMOID CALIBRATION
    # -----------------------------------------------------

    print(
        "Training sigmoid-calibrated model..."
    )

    sigmoid_base_model = (
        build_tuned_model(
            best_params
        )
    )

    sigmoid_model = (
        CalibratedClassifierCV(
            estimator=sigmoid_base_model,
            method="sigmoid",
            cv=3,
        )
    )

    sigmoid_model.fit(
        X_train,
        y_train,
    )

    sigmoid_probabilities = (
        sigmoid_model.predict_proba(
            X_valid
        )[:, 1]
    )

    # -----------------------------------------------------
    # ISOTONIC CALIBRATION
    # -----------------------------------------------------

    print(
        "Training isotonic-calibrated model..."
    )

    isotonic_base_model = (
        build_tuned_model(
            best_params
        )
    )

    isotonic_model = (
        CalibratedClassifierCV(
            estimator=isotonic_base_model,
            method="isotonic",
            cv=3,
        )
    )

    isotonic_model.fit(
        X_train,
        y_train,
    )

    isotonic_probabilities = (
        isotonic_model.predict_proba(
            X_valid
        )[:, 1]
    )

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    results = []

    results.append(
        evaluate_probabilities(
            "Uncalibrated",
            y_valid,
            uncalibrated_probabilities,
        )
    )

    results.append(
        evaluate_probabilities(
            "Sigmoid",
            y_valid,
            sigmoid_probabilities,
        )
    )

    results.append(
        evaluate_probabilities(
            "Isotonic",
            y_valid,
            isotonic_probabilities,
        )
    )

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 70)
    print("CALIBRATION COMPARISON")
    print("=" * 70)

    print(
        results_df
        .round(6)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # SELECT BEST CALIBRATION
    # -----------------------------------------------------

    calibrated_models = (
        results_df[
            results_df[
                "Model"
            ].isin(
                [
                    "Sigmoid",
                    "Isotonic",
                ]
            )
        ]
    )

    best_calibration = (
        calibrated_models.loc[
            calibrated_models[
                "Brier Score"
            ].idxmin()
        ]
    )

    print("\n" + "=" * 70)
    print("SELECTED CALIBRATION METHOD")
    print("=" * 70)

    print(
        f"Method: "
        f"{best_calibration['Model']}"
    )

    print(
        f"ROC-AUC: "
        f"{best_calibration['ROC-AUC']:.6f}"
    )

    print(
        f"PR-AUC: "
        f"{best_calibration['PR-AUC']:.6f}"
    )

    print(
        f"Brier Score: "
        f"{best_calibration['Brier Score']:.6f}"
    )

    print(
        f"Log Loss: "
        f"{best_calibration['Log Loss']:.6f}"
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    results_df.to_csv(
        CALIBRATION_RESULTS_PATH,
        index=False,
    )

    print(
        "\nCalibration metrics saved to:"
    )

    print(
        CALIBRATION_RESULTS_PATH
    )

    # -----------------------------------------------------
    # CALIBRATION CURVE
    # -----------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    probability_sets = {
        "Uncalibrated":
            uncalibrated_probabilities,

        "Sigmoid":
            sigmoid_probabilities,

        "Isotonic":
            isotonic_probabilities,
    }

    for (
        model_name,
        probabilities,
    ) in probability_sets.items():

        (
            fraction_positive,
            mean_predicted,
        ) = calibration_curve(
            y_valid,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=model_name,
        )

    plt.xlabel(
        "Mean Predicted Probability"
    )

    plt.ylabel(
        "Observed Positive Rate"
    )

    plt.title(
        "Probability Calibration Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        CALIBRATION_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nCalibration curve saved to:"
    )

    print(
        CALIBRATION_FIGURE_PATH
    )

    print(
        "\nIMPORTANT:"
        " Test set was not used for calibration."
    )

    print(
        "\nCalibration completed successfully."
    )


if __name__ == "__main__":
    main()