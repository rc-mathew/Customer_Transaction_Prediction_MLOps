from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint_returns_200():
    payload = {
        "features": [0.0] * 200
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_predict_response_structure():
    payload = {
        "features": [0.0] * 200
    }

    response = client.post("/predict", json=payload)

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "threshold" in data

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert 0.0 <= data["threshold"] <= 1.0


def test_predict_rejects_missing_features():
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


def test_predict_rejects_wrong_number_of_features():
    payload = {
        "features": [0.0] * 10
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422