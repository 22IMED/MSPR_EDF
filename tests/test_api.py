from fastapi.testclient import TestClient

import main


class DummyModel:
    def predict(self, X):
        return [55000.0]


client = TestClient(main.app)


def setup_module():
    main._model_cache["random_forest"] = DummyModel()


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert "model_name" in data


def test_models():
    response = client.get("/models")

    assert response.status_code == 200

    data = response.json()

    assert "available_models" in data
    assert "count" in data


def test_predict():
    response = client.post(
        "/predict",
        json={
            "date": "2025-01-15",
            "prevision_j1": 55000,
            "lag_1": 54000,
            "lag_7": 53000,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction_mw" in data
    assert data["prediction_mw"] == 55000.0


def test_predict_bad_date():
    response = client.post(
        "/predict",
        json={
            "date": "2019-01-01",
            "prevision_j1": 55000,
            "lag_1": 54000,
            "lag_7": 53000,
        },
    )

    assert response.status_code == 422


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "edf_requests_total" in response.text
