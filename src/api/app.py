from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/customer_transaction_model.joblib")
DECISION_THRESHOLD = 0.26


app = FastAPI(
    title="Customer Transaction Prediction API",
    description=(
        "REST API for predicting whether a customer "
        "will make a transaction."
    ),
    version="1.0.0",
)


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model file not found: {MODEL_PATH}. "
        "Run the training pipeline first."
    )


model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=200,
        max_length=200,
        description="Exactly 200 features: var_0 to var_199",
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float


@app.get("/")
def root():
    return {
        "message": "Customer Transaction Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    try:
        feature_names = [
            f"var_{i}"
            for i in range(200)
        ]

        input_df = pd.DataFrame(
            [request.features],
            columns=feature_names,
        )

        probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        prediction = int(
            probability >= DECISION_THRESHOLD
        )

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            threshold=DECISION_THRESHOLD,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc