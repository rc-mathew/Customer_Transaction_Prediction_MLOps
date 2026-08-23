"""
Data drift monitoring for Customer Transaction Prediction.

Reference population:
    Training data

Current population:
    Validation data or a simulated production batch

Drift metrics:
1. Population Stability Index (PSI)
2. Kolmogorov-Smirnov statistic and p-value

PSI interpretation:
    < 0.10       Stable / little drift
    0.10 - 0.25  Moderate drift
    > 0.25       Significant drift

Outputs:
- reports/metrics/drift_results.csv
- reports/metrics/drift_summary.json
- reports/figures/drift_analysis.png

IMPORTANT:
The untouched test set is not used.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data.load_data import load_training_data
from src.data.split_data import split_dataset


RANDOM_STATE = 42


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

DRIFT_RESULTS_PATH = (
    METRICS_DIR
    / "drift_results.csv"
)

DRIFT_SUMMARY_PATH = (
    METRICS_DIR
    / "drift_summary.json"
)

DRIFT_FIGURE_PATH = (
    FIGURES_DIR
    / "drift_analysis.png"
)


# ---------------------------------------------------------
# PSI
# ---------------------------------------------------------

def calculate_psi(
    reference,
    current,
    bins=10,
):
    """
    Calculate Population Stability Index.

    Bin boundaries are determined exclusively from
    the reference distribution.
    """

    reference = np.asarray(
        reference,
        dtype=float,
    )

    current = np.asarray(
        current,
        dtype=float,
    )

    reference = reference[
        np.isfinite(reference)
    ]

    current = current[
        np.isfinite(current)
    ]

    if (
        len(reference) == 0
        or len(current) == 0
    ):
        return np.nan

    # Reference-based quantile boundaries
    boundaries = np.quantile(
        reference,
        np.linspace(
            0,
            1,
            bins + 1,
        ),
    )

    boundaries = np.unique(
        boundaries
    )

    if len(boundaries) < 3:
        return 0.0

    # Ensure values outside the reference range
    # are still assigned to bins.
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    reference_counts, _ = np.histogram(
        reference,
        bins=boundaries,
    )

    current_counts, _ = np.histogram(
        current,
        bins=boundaries,
    )

    reference_pct = (
        reference_counts
        / len(reference)
    )

    current_pct = (
        current_counts
        / len(current)
    )

    epsilon = 1e-6

    reference_pct = np.where(
        reference_pct == 0,
        epsilon,
        reference_pct,
    )

    current_pct = np.where(
        current_pct == 0,
        epsilon,
        current_pct,
    )

    psi = np.sum(
        (
            current_pct
            - reference_pct
        )
        * np.log(
            current_pct
            / reference_pct
        )
    )

    return float(psi)


# ---------------------------------------------------------
# DRIFT CLASSIFICATION
# ---------------------------------------------------------

def classify_psi(psi):
    """
    Convert PSI into an interpretable drift category.
    """

    if pd.isna(psi):
        return "unknown"

    if psi < 0.10:
        return "stable"

    if psi < 0.25:
        return "moderate"

    return "significant"


# ---------------------------------------------------------
# SIMULATED PRODUCTION DRIFT
# ---------------------------------------------------------

def simulate_production_drift(
    current_data,
):
    """
    Introduce controlled shifts into a copy of the
    validation data.

    This is only for demonstrating that the drift
    detector works.

    In production, replace this with an actual
    incoming prediction batch.
    """

    simulated = current_data.copy()

    columns = list(
        simulated.columns
    )

    # Mean shift on first five features
    for column in columns[:5]:

        standard_deviation = (
            simulated[column].std()
        )

        simulated[column] = (
            simulated[column]
            + (
                0.50
                * standard_deviation
            )
        )

    # Scale shift on next five features
    for column in columns[5:10]:

        simulated[column] = (
            simulated[column]
            * 1.15
        )

    return simulated


# ---------------------------------------------------------
# FEATURE DRIFT
# ---------------------------------------------------------

def evaluate_feature_drift(
    reference_data,
    current_data,
):
    """
    Calculate PSI and KS statistics for every feature.
    """

    rows = []

    common_columns = [
        column
        for column in reference_data.columns
        if column in current_data.columns
    ]

    for column in common_columns:

        reference = (
            reference_data[column]
            .dropna()
            .values
        )

        current = (
            current_data[column]
            .dropna()
            .values
        )

        psi = calculate_psi(
            reference,
            current,
        )

        ks_result = ks_2samp(
            reference,
            current,
        )

        rows.append(
            {
                "feature":
                    column,

                "psi":
                    psi,

                "psi_status":
                    classify_psi(
                        psi
                    ),

                "ks_statistic":
                    float(
                        ks_result.statistic
                    ),

                "ks_p_value":
                    float(
                        ks_result.pvalue
                    ),

                "ks_drift":
                    bool(
                        ks_result.pvalue
                        < 0.05
                    ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    results = results.sort_values(
        "psi",
        ascending=False,
    )

    return results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Customer transaction "
            "data drift monitoring"
        )
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Introduce controlled synthetic drift "
            "into the validation batch."
        ),
    )

    args = parser.parse_args()

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DATA DRIFT MONITORING"
    )

    print(
        "=" * 75
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
    # DATA
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

    reference_data = (
        X_train.copy()
    )

    current_data = (
        X_valid.copy()
    )

    if args.simulate:

        print(
            "\nSimulation mode enabled."
        )

        print(
            "Controlled drift will be introduced "
            "into selected features."
        )

        current_data = (
            simulate_production_drift(
                current_data
            )
        )

    else:

        print(
            "\nNatural validation distribution "
            "will be used as the current batch."
        )

    print(
        "\nReference shape:",
        reference_data.shape,
    )

    print(
        "Current shape:",
        current_data.shape,
    )

    # -----------------------------------------------------
    # DRIFT ANALYSIS
    # -----------------------------------------------------

    print(
        "\nCalculating feature drift..."
    )

    drift_results = (
        evaluate_feature_drift(
            reference_data,
            current_data,
        )
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    stable_count = int(
        (
            drift_results[
                "psi_status"
            ]
            == "stable"
        ).sum()
    )

    moderate_count = int(
        (
            drift_results[
                "psi_status"
            ]
            == "moderate"
        ).sum()
    )

    significant_count = int(
        (
            drift_results[
                "psi_status"
            ]
            == "significant"
        ).sum()
    )

    ks_drift_count = int(
        drift_results[
            "ks_drift"
        ].sum()
    )

    total_features = int(
        len(
            drift_results
        )
    )

    drift_percentage = (
        100
        * (
            moderate_count
            + significant_count
        )
        / total_features
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "DRIFT SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"Features evaluated:      "
        f"{total_features}"
    )

    print(
        f"Stable features:         "
        f"{stable_count}"
    )

    print(
        f"Moderate PSI drift:      "
        f"{moderate_count}"
    )

    print(
        f"Significant PSI drift:   "
        f"{significant_count}"
    )

    print(
        f"KS drift alerts:         "
        f"{ks_drift_count}"
    )

    print(
        f"PSI drift percentage:    "
        f"{drift_percentage:.2f}%"
    )

    print(
        "\nTOP 15 FEATURES BY PSI"
    )

    print(
        "-" * 75
    )

    print(
        drift_results[
            [
                "feature",
                "psi",
                "psi_status",
                "ks_statistic",
                "ks_p_value",
            ]
        ]
        .head(15)
        .round(6)
        .to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # SAVE CSV
    # -----------------------------------------------------

    drift_results.to_csv(
        DRIFT_RESULTS_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # SAVE SUMMARY JSON
    # -----------------------------------------------------

    summary = {
        "monitoring_type":
            "feature_distribution_drift",

        "simulation_mode":
            bool(
                args.simulate
            ),

        "reference_population":
            "training_split",

        "current_population":
            (
                "simulated_validation_batch"
                if args.simulate
                else "validation_split"
            ),

        "features_evaluated":
            total_features,

        "stable_features":
            stable_count,

        "moderate_drift_features":
            moderate_count,

        "significant_drift_features":
            significant_count,

        "ks_drift_alerts":
            ks_drift_count,

        "psi_drift_percentage":
            float(
                drift_percentage
            ),

        "psi_thresholds":
            {
                "stable":
                    "< 0.10",

                "moderate":
                    "0.10 to < 0.25",

                "significant":
                    ">= 0.25",
            },

        "test_set_used":
            False,
    }

    with open(
        DRIFT_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # PLOT TOP FEATURES
    # -----------------------------------------------------

    top_features = (
        drift_results
        .head(20)
        .sort_values(
            "psi"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features[
            "feature"
        ],
        top_features[
            "psi"
        ],
    )

    plt.axvline(
        0.10,
        linestyle="--",
        label="Moderate drift threshold",
    )

    plt.axvline(
        0.25,
        linestyle="--",
        label="Significant drift threshold",
    )

    plt.xlabel(
        "Population Stability Index (PSI)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top Feature Drift Scores"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DRIFT_FIGURE_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nDrift results saved to:"
    )

    print(
        DRIFT_RESULTS_PATH
    )

    print(
        "\nDrift summary saved to:"
    )

    print(
        DRIFT_SUMMARY_PATH
    )

    print(
        "\nDrift figure saved to:"
    )

    print(
        DRIFT_FIGURE_PATH
    )

    print(
        "\nIMPORTANT:"
        " Test set was not used for drift monitoring."
    )

    print(
        "\nDrift monitoring completed successfully."
    )


if __name__ == "__main__":
    main()