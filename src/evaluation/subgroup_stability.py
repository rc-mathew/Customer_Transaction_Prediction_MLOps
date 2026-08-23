"""
Subgroup and model-stability evaluation.

The Customer Transaction dataset contains anonymized numerical
features rather than genuine protected demographic attributes.

Therefore this module DOES NOT claim demographic fairness testing.

Instead, it:
1. Selects informative numerical features using TRAINING data only.
2. Builds quantile-based subgroups.
3. Evaluates model performance within each subgroup.
4. Detects slices where precision, recall, F1 or ROC-AUC degrade.

Outputs:
- reports/metrics/subgroup_stability_results.csv
- reports/metrics/subgroup_stability_summary.json
- reports/figures/subgroup_stability.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
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

TUNING_RESULTS_PATH = (
    METRICS_DIR
    / "hist_gradient_tuning_results.json"
)

CALIBRATION_RESULTS_PATH = (
    METRICS_DIR
    / "calibration_results.csv"
)

THRESHOLD_METADATA_PATH = (
    METRICS_DIR
    / "threshold_metadata.json"
)


# ---------------------------------------------------------
# LOAD CONFIGURATION
# ---------------------------------------------------------

def load_best_parameters():

    if not TUNING_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Tuning results not found: "
            f"{TUNING_RESULTS_PATH}"
        )

    with open(
        TUNING_RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(file)

    return results["best_params"]


def load_selected_threshold():

    if not THRESHOLD_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Threshold metadata not found: "
            f"{THRESHOLD_METADATA_PATH}"
        )

    with open(
        THRESHOLD_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    return float(
        metadata["selected_threshold"]
    )


def load_calibration_method():

    if not CALIBRATION_RESULTS_PATH.exists():

        raise FileNotFoundError(
            f"Calibration results not found: "
            f"{CALIBRATION_RESULTS_PATH}"
        )

    calibration_results = pd.read_csv(
        CALIBRATION_RESULTS_PATH
    )

    calibrated_only = (
        calibration_results[
            calibration_results[
                "Model"
            ].isin(
                [
                    "Sigmoid",
                    "Isotonic",
                ]
            )
        ]
    )

    best_row = calibrated_only.loc[
        calibrated_only[
            "Brier Score"
        ].idxmin()
    ]

    return (
        best_row["Model"]
        .lower()
    )


# ---------------------------------------------------------
# BUILD PRODUCTION-CANDIDATE MODEL
# ---------------------------------------------------------

def build_model(
    best_params,
    calibration_method,
):

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

    calibrated_model = (
        CalibratedClassifierCV(
            estimator=base_model,
            method=calibration_method,
            cv=3,
        )
    )

    return calibrated_model


# ---------------------------------------------------------
# FEATURE SELECTION FOR SLICE ANALYSIS
# ---------------------------------------------------------

def select_stability_features(
    X_train,
    y_train,
    number_of_features=3,
):
    """
    Select numerical features having the strongest
    absolute correlation with the target.

    IMPORTANT:
    Selection is performed using TRAINING DATA ONLY.
    """

    correlations = {}

    for column in X_train.columns:

        feature = X_train[column]

        if feature.nunique() <= 1:
            continue

        correlation = np.corrcoef(
            feature,
            y_train
        )[0, 1]

        if not np.isnan(correlation):

            correlations[column] = abs(
                correlation
            )

    ranked_features = sorted(
        correlations,
        key=correlations.get,
        reverse=True,
    )

    return ranked_features[
        :number_of_features
    ]


# ---------------------------------------------------------
# SUBGROUP METRICS
# ---------------------------------------------------------

def calculate_slice_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "sample_count":
            len(y_true),

        "actual_positive_rate":
            float(
                np.mean(y_true)
            ),

        "predicted_positive_rate":
            float(
                np.mean(predictions)
            ),

        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),
    }

    # ROC-AUC requires both classes
    if len(
        np.unique(
            y_true
        )
    ) == 2:

        metrics["roc_auc"] = (
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

    else:

        metrics["roc_auc"] = np.nan

    return metrics


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SUBGROUP / MODEL STABILITY EVALUATION"
    )

    print(
        "=" * 75
    )

    print(
        "\nNOTE:"
        " This dataset does not contain genuine"
        " demographic protected attributes."
    )

    print(
        "The analysis below evaluates model stability"
        " across data-derived feature slices."
    )

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

    print(
        "\nLoading dataset..."
    )

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
    # LOAD PRODUCTION CONFIGURATION
    # -----------------------------------------------------

    best_params = (
        load_best_parameters()
    )

    calibration_method = (
        load_calibration_method()
    )

    selected_threshold = (
        load_selected_threshold()
    )

    print(
        f"\nCalibration method: "
        f"{calibration_method}"
    )

    print(
        f"Decision threshold: "
        f"{selected_threshold:.4f}"
    )

    # -----------------------------------------------------
    # TRAIN MODEL
    # -----------------------------------------------------

    model = build_model(
        best_params,
        calibration_method,
    )

    print(
        "\nTraining calibrated model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    # -----------------------------------------------------
    # SELECT FEATURES USING TRAINING DATA ONLY
    # -----------------------------------------------------

    stability_features = (
        select_stability_features(
            X_train,
            y_train,
            number_of_features=3,
        )
    )

    print(
        "\nFeatures selected for"
        " stability analysis:"
    )

    for feature in stability_features:
        print(
            f"- {feature}"
        )

    # -----------------------------------------------------
    # BUILD QUANTILE SEGMENTS
    # -----------------------------------------------------

    results = []

    for feature in stability_features:

        print(
            f"\nEvaluating slices for "
            f"{feature}..."
        )

        # Training-data quartile boundaries
        quantile_edges = (
            X_train[
                feature
            ]
            .quantile(
                [
                    0.00,
                    0.25,
                    0.50,
                    0.75,
                    1.00,
                ]
            )
            .values
        )

        # Ensure unique edges
        quantile_edges = np.unique(
            quantile_edges
        )

        if len(
            quantile_edges
        ) < 3:

            print(
                f"Skipping {feature}: "
                "insufficient unique "
                "quantile boundaries."
            )

            continue

        labels = [
            f"Q{i + 1}"
            for i in range(
                len(
                    quantile_edges
                ) - 1
            )
        ]

        validation_groups = pd.cut(
            X_valid[
                feature
            ],
            bins=quantile_edges,
            labels=labels,
            include_lowest=True,
            duplicates="drop",
        )

        for group in labels:

            mask = (
                validation_groups
                == group
            )

            if mask.sum() == 0:
                continue

            group_y = (
                y_valid.loc[
                    mask
                ]
            )

            group_probabilities = (
                validation_probabilities[
                    mask.values
                ]
            )

            metrics = (
                calculate_slice_metrics(
                    group_y,
                    group_probabilities,
                    selected_threshold,
                )
            )

            metrics[
                "feature"
            ] = feature

            metrics[
                "subgroup"
            ] = group

            results.append(
                metrics
            )

    # -----------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    column_order = [
        "feature",
        "subgroup",
        "sample_count",
        "actual_positive_rate",
        "predicted_positive_rate",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    results_df = results_df[
        column_order
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SUBGROUP PERFORMANCE RESULTS"
    )

    print(
        "=" * 75
    )

    print(
        results_df
        .round(4)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # STABILITY SUMMARY
    # -----------------------------------------------------

    metric_ranges = {}

    for metric in [
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]:

        valid_values = (
            results_df[
                metric
            ]
            .dropna()
        )

        if len(
            valid_values
        ) == 0:
            continue

        metric_ranges[
            metric
        ] = {
            "minimum":
                float(
                    valid_values.min()
                ),

            "maximum":
                float(
                    valid_values.max()
                ),

            "range":
                float(
                    valid_values.max()
                    - valid_values.min()
                ),
        }

    summary = {
        "analysis_type":
            "data-derived subgroup stability",

        "protected_attribute_fairness":
            False,

        "reason":
            (
                "Dataset contains anonymized "
                "numerical features and no "
                "verified protected demographic "
                "attributes."
            ),

        "features_evaluated":
            stability_features,

        "selected_threshold":
            selected_threshold,

        "calibration_method":
            calibration_method,

        "metric_ranges":
            metric_ranges,
    }

    # -----------------------------------------------------
    # SAVE OUTPUTS
    # -----------------------------------------------------

    results_path = (
        METRICS_DIR
        / "subgroup_stability_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary_path = (
        METRICS_DIR
        / "subgroup_stability_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # PLOT F1 BY SUBGROUP
    # -----------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    for feature in stability_features:

        subset = results_df[
            results_df[
                "feature"
            ]
            == feature
        ]

        if subset.empty:
            continue

        plt.plot(
            subset[
                "subgroup"
            ],
            subset[
                "f1"
            ],
            marker="o",
            label=feature,
        )

    plt.xlabel(
        "Feature Quantile Segment"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        "Model Stability Across Data-Derived Subgroups"
    )

    plt.legend()

    plt.tight_layout()

    figure_path = (
        FIGURES_DIR
        / "subgroup_stability.png"
    )

    plt.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nResults saved to:"
    )

    print(
        results_path
    )

    print(
        "\nSummary saved to:"
    )

    print(
        summary_path
    )

    print(
        "\nFigure saved to:"
    )

    print(
        figure_path
    )

    print(
        "\nIMPORTANT:"
        " This analysis should not be presented"
        " as demographic fairness testing."
    )

    print(
        "\nIMPORTANT:"
        " Test set remains untouched."
    )

    print(
        "\nSubgroup stability analysis"
        " completed successfully."
    )


if __name__ == "__main__":
    main()