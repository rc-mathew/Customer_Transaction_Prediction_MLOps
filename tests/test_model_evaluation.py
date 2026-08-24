import pandas as pd
import pytest

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.evaluation.calibration import (
    build_tuned_model,
    evaluate_probabilities,
)

import src.evaluation.threshold_optimization as threshold_module


def test_probability_metrics_are_valid():
    y_true = [0, 0, 1, 1]

    probabilities = [
        0.10,
        0.20,
        0.80,
        0.90,
    ]

    results = evaluate_probabilities(
        "Test Model",
        y_true,
        probabilities,
    )

    assert results["Model"] == "Test Model"

    assert 0.0 <= results["ROC-AUC"] <= 1.0
    assert 0.0 <= results["PR-AUC"] <= 1.0
    assert 0.0 <= results["Brier Score"] <= 1.0
    assert results["Log Loss"] >= 0.0


def test_better_probabilities_have_lower_brier_score():
    y_true = [0, 0, 1, 1]

    good_probabilities = [
        0.05,
        0.10,
        0.90,
        0.95,
    ]

    poor_probabilities = [
        0.45,
        0.40,
        0.60,
        0.55,
    ]

    good_results = evaluate_probabilities(
        "Good Model",
        y_true,
        good_probabilities,
    )

    poor_results = evaluate_probabilities(
        "Poor Model",
        y_true,
        poor_probabilities,
    )

    assert (
        good_results["Brier Score"]
        <
        poor_results["Brier Score"]
    )


def test_tuned_model_pipeline_is_created():
    best_params = {
        "classifier__learning_rate": 0.1,
        "classifier__max_iter": 50,
        "classifier__max_leaf_nodes": 15,
    }

    model = build_tuned_model(best_params)

    assert isinstance(model, Pipeline)

    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps

    assert (
        model.get_params()[
            "classifier__learning_rate"
        ]
        == 0.1
    )

    assert (
        model.get_params()[
            "classifier__max_iter"
        ]
        == 50
    )

    assert (
        model.get_params()[
            "classifier__max_leaf_nodes"
        ]
        == 15
    )


def test_best_calibration_method_uses_lowest_brier_score(
    tmp_path,
    monkeypatch,
):
    calibration_file = (
        tmp_path / "calibration_results.csv"
    )

    results = pd.DataFrame(
        {
            "Model": [
                "Uncalibrated",
                "Sigmoid",
                "Isotonic",
            ],
            "ROC-AUC": [
                0.88,
                0.88,
                0.87,
            ],
            "PR-AUC": [
                0.57,
                0.58,
                0.57,
            ],
            "Brier Score": [
                0.10,
                0.08,
                0.09,
            ],
            "Log Loss": [
                0.30,
                0.25,
                0.27,
            ],
        }
    )

    results.to_csv(
        calibration_file,
        index=False,
    )

    monkeypatch.setattr(
        threshold_module,
        "CALIBRATION_RESULTS_PATH",
        calibration_file,
    )

    method = (
        threshold_module
        .get_best_calibration_method()
    )

    assert method == "sigmoid"


@pytest.mark.parametrize(
    "method",
    [
        "sigmoid",
        "isotonic",
    ],
)
def test_calibrated_model_is_created(method):
    best_params = {
        "classifier__learning_rate": 0.1,
        "classifier__max_iter": 50,
        "classifier__max_leaf_nodes": 15,
    }

    model = (
        threshold_module
        .build_calibrated_model(
            best_params,
            method,
        )
    )

    assert isinstance(
        model,
        CalibratedClassifierCV,
    )

    assert model.method == method