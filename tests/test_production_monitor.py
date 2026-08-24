import numpy as np
import pandas as pd

from monitoring.production_monitor import (
    monitor_batch,
    summarize_drift,
)


def test_stable_batch_does_not_trigger_alert():
    reference = pd.DataFrame(
        {
            "var_1": np.arange(100),
            "var_2": np.arange(100),
        }
    )

    current = reference.copy()

    _, summary = monitor_batch(
        reference,
        current,
    )

    assert summary["alert"] is False


def test_shifted_batch_triggers_alert():
    reference = pd.DataFrame(
        {
            "var_1": np.arange(100),
            "var_2": np.arange(100),
        }
    )

    current = pd.DataFrame(
        {
            "var_1":
                np.arange(100) + 1000,
            "var_2":
                np.arange(100) + 1000,
        }
    )

    _, summary = monitor_batch(
        reference,
        current,
    )

    assert summary["alert"] is True


def test_summary_contains_required_fields():
    drift_results = pd.DataFrame(
        {
            "psi_status": [
                "stable",
                "moderate",
                "significant",
            ]
        }
    )

    summary = summarize_drift(
        drift_results
    )

    assert (
        summary["total_features"]
        == 3
    )

    assert (
        summary[
            "moderate_drift_features"
        ]
        == 1
    )

    assert (
        summary[
            "significant_drift_features"
        ]
        == 1
    )

    assert "drift_percentage" in summary
    assert "alert" in summary


def test_empty_results_do_not_crash():
    drift_results = pd.DataFrame(
        {
            "psi_status": []
        }
    )

    summary = summarize_drift(
        drift_results
    )

    assert (
        summary["total_features"]
        == 0
    )

    assert (
        summary["drift_percentage"]
        == 0.0
    )