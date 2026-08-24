from pathlib import Path

import joblib
import numpy as np


MODEL_PATH = Path("models/customer_transaction_model.joblib")


def test_model_artifact_exists():
    assert MODEL_PATH.exists(), f"Model artifact not found: {MODEL_PATH}"


def test_model_loads_successfully():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_returns_probability_between_zero_and_one():
    model = joblib.load(MODEL_PATH)

    sample = np.zeros((1, 200))

    probability = model.predict_proba(sample)[0, 1]

    assert 0.0 <= probability <= 1.0


def test_model_returns_binary_prediction():
    model = joblib.load(MODEL_PATH)

    sample = np.zeros((1, 200))

    prediction = model.predict(sample)[0]

    assert prediction in [0, 1]