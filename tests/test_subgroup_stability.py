import numpy as np
import pandas as pd

from src.evaluation.subgroup_stability import (
    calculate_slice_metrics,
    select_stability_features,
)


def test_select_stability_features_returns_requested_count():
    X = pd.DataFrame(
        {
            "var_0": [0, 1, 2, 3, 4, 5],
            "var_1": [5, 4, 3, 2, 1, 0],
            "var_2": [1, 1, 1, 1, 1, 1],
            "var_3": [0, 0, 1, 1, 2, 2],
        }
    )

    y = pd.Series([0, 0, 0, 1, 1, 1])

    selected = select_stability_features(
        X,
        y,
        number_of_features=2,
    )

    assert len(selected) == 2
    assert "var_2" not in selected


def test_slice_metrics_return_expected_fields():
    y_true = pd.Series([0, 0, 1, 1])

    probabilities = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    metrics = calculate_slice_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    expected_keys = {
        "sample_count",
        "actual_positive_rate",
        "predicted_positive_rate",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert expected_keys.issubset(
        metrics.keys()
    )


def test_slice_metrics_are_bounded():
    y_true = pd.Series([0, 0, 1, 1])

    probabilities = np.array(
        [0.10, 0.20, 0.80, 0.90]
    )

    metrics = calculate_slice_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    bounded_metrics = [
        "actual_positive_rate",
        "predicted_positive_rate",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    for metric in bounded_metrics:
        assert 0.0 <= metrics[metric] <= 1.0


def test_perfect_slice_predictions_score_perfectly():
    y_true = pd.Series([0, 0, 1, 1])

    probabilities = np.array(
        [0.05, 0.10, 0.90, 0.95]
    )

    metrics = calculate_slice_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_single_class_slice_returns_nan_roc_auc():
    y_true = pd.Series([0, 0, 0, 0])

    probabilities = np.array(
        [0.10, 0.20, 0.30, 0.40]
    )

    metrics = calculate_slice_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert np.isnan(
        metrics["roc_auc"]
    )