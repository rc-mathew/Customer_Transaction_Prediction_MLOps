import numpy as np

from monitoring.drift_detection import (
    calculate_psi,
    classify_psi,
)


def test_identical_distributions_have_low_psi():

    reference = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dtype=float,
    )

    current = reference.copy()

    psi = calculate_psi(
        reference,
        current,
    )

    assert psi < 0.10


def test_shifted_distribution_has_higher_psi():

    reference = np.arange(
        1,
        101,
        dtype=float,
    )

    current = reference + 100

    psi = calculate_psi(
        reference,
        current,
    )

    assert psi > 0.25


def test_psi_classification():

    assert classify_psi(0.05) == "stable"
    assert classify_psi(0.15) == "moderate"
    assert classify_psi(0.30) == "significant"