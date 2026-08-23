"""
Decision-threshold optimization for the calibrated
Customer Transaction Prediction model.

Uses the validation set only.

Outputs:
- reports/metrics/threshold_results.csv
- reports/metrics/threshold_metadata.json
- reports/figures/threshold_analysis.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.data.load_data import load_training_data
from src.data.split_data import split_dataset
from src.features.preprocessing import build_tree_preprocessor


RANDOM_STATE = 42


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TUNING_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "hist_gradient_tuning_results.json"
)

CALIBRATION_RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
    / "calibration_results.csv"
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


def load_best_parameters():
    """
    Load tuned HistGradientBoosting parameters.
    """

    if not TUNING_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Tuning results not found at: "
            f"{TUNING_RESULTS_PATH}"
        )

    with open(
        TUNING_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    return results["best_params"]


def get_best_calibration_method():
    """
    Select sigmoid or isotonic based on lowest Brier score.
    """

    if not CALIBRATION_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Calibration results not found at: "
            f"{CALIBRATION_RESULTS_PATH}"
        )

    calibration_results = pd.read_csv(
        CALIBRATION_RESULTS_PATH
    )

    calibrated_only = calibration_results[
        calibration_results["Model"].isin(
            ["Sigmoid", "Isotonic"]
        )
    ]

    best_row = calibrated_only.loc[
        calibrated_only[
            "Brier Score"
        ].idxmin()
    ]

    return best_row["Model"].lower()


def build_calibrated_model(
    best_params,
    calibration_method,
):
    """
    Build tuned HistGradientBoosting with selected calibration.
    """

    base_model = Pipeline(
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

    base_model.set_params(
        **best_params
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=calibration_method,
        cv=3,
    )

    return calibrated_model


def main():

    print("\n" + "=" * 70)
    print("DECISION THRESHOLD OPTIMIZATION")
    print("=" * 70)

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # DATA
    # -----------------------------------------------------

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
    # CONFIGURATION
    # -----------------------------------------------------

    best_params = load_best_parameters()

    calibration_method = (
        get_best_calibration_method()
    )

    print(
        f"\nSelected calibration method: "
        f"{calibration_method}"
    )

    # -----------------------------------------------------
    # TRAIN CALIBRATED MODEL
    # -----------------------------------------------------

    model = build_calibrated_model(
        best_params,
        calibration_method,
    )

    print(
        "\nTraining calibrated tuned model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    # -----------------------------------------------------
    # THRESHOLD SEARCH
    # -----------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.81,
        0.01,
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": threshold,

                "precision":
                    precision_score(
                        y_valid,
                        predictions,
                        zero_division=0,
                    ),

                "recall":
                    recall_score(
                        y_valid,
                        predictions,
                        zero_division=0,
                    ),

                "f1":
                    f1_score(
                        y_valid,
                        predictions,
                        zero_division=0,
                    ),

                "balanced_accuracy":
                    balanced_accuracy_score(
                        y_valid,
                        predictions,
                    ),
            }
        )

    results_df = pd.DataFrame(
        rows
    )

    # -----------------------------------------------------
    # BEST THRESHOLD BY F1
    # -----------------------------------------------------

    best_row = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold:          "
        f"{best_threshold:.2f}"
    )

    print(
        f"Precision:          "
        f"{best_row['precision']:.4f}"
    )

    print(
        f"Recall:             "
        f"{best_row['recall']:.4f}"
    )

    print(
        f"F1 Score:           "
        f"{best_row['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{best_row['balanced_accuracy']:.4f}"
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    threshold_results_path = (
        METRICS_DIR
        / "threshold_results.csv"
    )

    results_df.to_csv(
        threshold_results_path,
        index=False,
    )

    metadata = {
        "calibration_method":
            calibration_method,

        "optimization_metric":
            "f1",

        "selected_threshold":
            best_threshold,

        "precision":
            float(
                best_row["precision"]
            ),

        "recall":
            float(
                best_row["recall"]
            ),

        "f1":
            float(
                best_row["f1"]
            ),

        "balanced_accuracy":
            float(
                best_row[
                    "balanced_accuracy"
                ]
            ),
    }

    threshold_metadata_path = (
        METRICS_DIR
        / "threshold_metadata.json"
    )

    with open(
        threshold_metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        results_df["threshold"],
        results_df["precision"],
        label="Precision",
    )

    plt.plot(
        results_df["threshold"],
        results_df["recall"],
        label="Recall",
    )

    plt.plot(
        results_df["threshold"],
        results_df["f1"],
        label="F1",
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        label=(
            f"Selected threshold "
            f"= {best_threshold:.2f}"
        ),
    )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "Metric Score"
    )

    plt.title(
        "Decision Threshold Analysis"
    )

    plt.legend()

    plt.tight_layout()

    threshold_figure_path = (
        FIGURES_DIR
        / "threshold_analysis.png"
    )

    plt.savefig(
        threshold_figure_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nThreshold results saved to:"
    )

    print(
        threshold_results_path
    )

    print(
        "\nThreshold metadata saved to:"
    )

    print(
        threshold_metadata_path
    )

    print(
        "\nThreshold plot saved to:"
    )

    print(
        threshold_figure_path
    )

    print(
        "\nIMPORTANT: Test set remains untouched."
    )

    print(
        "\nThreshold optimization completed successfully."
    )


if __name__ == "__main__":
    main()