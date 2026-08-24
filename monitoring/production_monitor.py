import argparse
import json
from pathlib import Path

import pandas as pd

from monitoring.drift_detection import (
    evaluate_feature_drift,
)


DRIFT_FEATURE_PERCENTAGE_THRESHOLD = 10.0


def summarize_drift(drift_results):
    """
    Summarize feature-level drift results and decide
    whether a production drift alert should be raised.
    """

    total_features = len(drift_results)

    if total_features == 0:
        return {
            "total_features": 0,
            "stable_features": 0,
            "moderate_drift_features": 0,
            "significant_drift_features": 0,
            "ks_drift_alerts": 0,
            "drift_percentage": 0.0,
            "alert": False,
        }

    stable_features = int(
        (
            drift_results["psi_status"]
            == "stable"
        ).sum()
    )

    moderate_features = int(
        (
            drift_results["psi_status"]
            == "moderate"
        ).sum()
    )

    significant_features = int(
        (
            drift_results["psi_status"]
            == "significant"
        ).sum()
    )

    if "ks_drift" in drift_results.columns:
        ks_drift_alerts = int(
            drift_results["ks_drift"].sum()
        )
    else:
        ks_drift_alerts = 0

    drifted_features = (
        moderate_features
        + significant_features
    )

    drift_percentage = (
        100.0
        * drifted_features
        / total_features
    )

    alert = bool(
        significant_features > 0
        or drift_percentage
        >= DRIFT_FEATURE_PERCENTAGE_THRESHOLD
    )

    return {
        "total_features":
            int(total_features),

        "stable_features":
            stable_features,

        "moderate_drift_features":
            moderate_features,

        "significant_drift_features":
            significant_features,

        "ks_drift_alerts":
            ks_drift_alerts,

        "drift_percentage":
            float(drift_percentage),

        "alert":
            alert,
    }


def monitor_batch(
    reference_data,
    production_data,
):
    """
    Compare a production batch against the reference
    population and return drift results plus alert summary.
    """

    if not isinstance(
        reference_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "reference_data must be a pandas DataFrame"
        )

    if not isinstance(
        production_data,
        pd.DataFrame,
    ):
        raise TypeError(
            "production_data must be a pandas DataFrame"
        )

    if reference_data.empty:
        raise ValueError(
            "reference_data cannot be empty"
        )

    if production_data.empty:
        raise ValueError(
            "production_data cannot be empty"
        )

    common_columns = [
        column
        for column in reference_data.columns
        if column in production_data.columns
    ]

    if not common_columns:
        raise ValueError(
            "Reference and production data "
            "have no common features."
        )

    reference_subset = (
        reference_data[
            common_columns
        ].copy()
    )

    production_subset = (
        production_data[
            common_columns
        ].copy()
    )

    drift_results = evaluate_feature_drift(
        reference_subset,
        production_subset,
    )

    summary = summarize_drift(
        drift_results
    )

    return (
        drift_results,
        summary,
    )


def load_csv(path):
    """
    Load a CSV file and validate that it is not empty.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    data = pd.read_csv(
        path
    )

    if data.empty:
        raise ValueError(
            f"CSV file is empty: {path}"
        )

    return data


def save_monitoring_outputs(
    drift_results,
    summary,
    results_path,
    summary_path,
):
    """
    Save feature-level drift results and summary.
    """

    results_path = Path(
        results_path
    )

    summary_path = Path(
        summary_path
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_results.to_csv(
        results_path,
        index=False,
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Production batch drift monitoring "
            "for Customer Transaction Prediction"
        )
    )

    parser.add_argument(
        "--reference",
        required=True,
        help=(
            "Reference CSV, normally a training "
            "or approved baseline population."
        ),
    )

    parser.add_argument(
        "--current",
        required=True,
        help=(
            "Current production prediction batch CSV."
        ),
    )

    parser.add_argument(
        "--results-output",
        default=(
            "reports/metrics/"
            "production_drift_results.csv"
        ),
        help=(
            "Output path for feature-level "
            "drift results."
        ),
    )

    parser.add_argument(
        "--summary-output",
        default=(
            "reports/metrics/"
            "production_drift_summary.json"
        ),
        help=(
            "Output path for the production "
            "drift summary."
        ),
    )

    args = parser.parse_args()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PRODUCTION DRIFT MONITORING"
    )

    print(
        "=" * 70
    )

    reference_data = load_csv(
        args.reference
    )

    production_data = load_csv(
        args.current
    )

    print(
        f"\nReference batch shape: "
        f"{reference_data.shape}"
    )

    print(
        f"Production batch shape: "
        f"{production_data.shape}"
    )

    drift_results, summary = (
        monitor_batch(
            reference_data,
            production_data,
        )
    )

    print(
        "\nProduction drift summary:"
    )

    print(
        json.dumps(
            summary,
            indent=4,
        )
    )

    print(
        "\nTop drifted features:"
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
        .head(10)
        .round(6)
        .to_string(
            index=False
        )
    )

    save_monitoring_outputs(
        drift_results=
            drift_results,

        summary=
            summary,

        results_path=
            args.results_output,

        summary_path=
            args.summary_output,
    )

    print(
        "\nResults saved to:"
    )

    print(
        args.results_output
    )

    print(
        "\nSummary saved to:"
    )

    print(
        args.summary_output
    )

    if summary["alert"]:
        print(
            "\nALERT: Production drift "
            "threshold exceeded."
        )
    else:
        print(
            "\nSTATUS: No significant "
            "production drift detected."
        )


if __name__ == "__main__":
    main()