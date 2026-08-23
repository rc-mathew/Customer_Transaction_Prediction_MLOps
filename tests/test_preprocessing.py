import numpy as np
import pandas as pd

from src.features.preprocessing import (
    build_linear_preprocessor,
    build_tree_preprocessor,
)


def make_sample_data():
    return pd.DataFrame(
        {
            "var_0": [1.0, 2.0, np.nan, 4.0],
            "var_1": [10.0, 11.0, 12.0, 13.0],
            "var_2": [5.0, 5.5, 6.0, 6.5],
        }
    )


def test_linear_preprocessor_runs():

    X = make_sample_data()

    preprocessor = (
        build_linear_preprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    assert transformed.shape == X.shape
    assert np.isfinite(transformed).all()


def test_tree_preprocessor_runs():

    X = make_sample_data()

    preprocessor = (
        build_tree_preprocessor()
    )

    transformed = (
        preprocessor.fit_transform(X)
    )

    assert transformed.shape == X.shape
    assert np.isfinite(transformed).all()