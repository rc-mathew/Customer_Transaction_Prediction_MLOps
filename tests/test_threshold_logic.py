import numpy as np

from sklearn.metrics import f1_score


def apply_threshold(
    probabilities,
    threshold,
):
    return (
        np.asarray(probabilities)
        >= threshold
    ).astype(int)


def test_threshold_predictions():

    probabilities = np.array(
        [0.10, 0.30, 0.60, 0.90]
    )

    predictions = apply_threshold(
        probabilities,
        0.50,
    )

    expected = np.array(
        [0, 0, 1, 1]
    )

    assert np.array_equal(
        predictions,
        expected,
    )


def test_lower_threshold_increases_positive_predictions():

    probabilities = np.array(
        [0.10, 0.25, 0.40, 0.55, 0.80]
    )

    predictions_050 = apply_threshold(
        probabilities,
        0.50,
    )

    predictions_030 = apply_threshold(
        probabilities,
        0.30,
    )

    assert (
        predictions_030.sum()
        >= predictions_050.sum()
    )


def test_f1_calculation():

    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.10, 0.20, 0.70, 0.90]
    )

    predictions = apply_threshold(
        probabilities,
        0.50,
    )

    score = f1_score(
        y_true,
        predictions,
    )

    assert score == 1.0