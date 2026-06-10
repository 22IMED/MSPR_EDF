"""Tests E2E — API en production."""

import os
import time
import pytest
import requests

API_URL = os.getenv(
    "API_URL",
    "https://edf-api.orangebeach-b5cf8765.francecentral.azurecontainerapps.io",
)


@pytest.mark.e2e
def test_api_health():
    """L'API répond 200 sur /health."""
    response = requests.get(f"{API_URL}/health", timeout=30)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


@pytest.mark.e2e
def test_api_models():
    """L'API liste les modèles disponibles."""
    response = requests.get(f"{API_URL}/models", timeout=30)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    assert len(data["available_models"]) > 0


@pytest.mark.e2e
def test_api_predict():
    """L'API fait une prédiction correcte."""
    response = requests.post(
        f"{API_URL}/predict",
        json={
            "date": "2025-01-15",
            "prevision_j1": 65000,
            "lag_1": 64000,
            "lag_7": 63000,
        },
        timeout=30,
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction_mw" in data
    assert data["prediction_mw"] > 0
    assert data["prediction_mw"] < 200_000
    assert data["r2_score"] > 0.8


@pytest.mark.e2e
def test_api_predict_all_models():
    """Tous les modèles peuvent faire des prédictions."""
    models_response = requests.get(f"{API_URL}/models", timeout=30)
    models = models_response.json()["available_models"]

    for model in models:
        for attempt in range(3):
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "date": "2025-06-01",
                    "prevision_j1": 55000,
                    "model_name": model["model_name"],
                },
                timeout=30,
            )
            if response.status_code == 200:
                break
            time.sleep(5)

        assert response.status_code == 200, (
            f"Modèle '{model['model_name']}' — "
            f"HTTP {response.status_code}: {response.text}"
        )


@pytest.mark.e2e
def test_api_metrics():
    """L'endpoint /metrics expose des métriques Prometheus."""
    response = requests.get(f"{API_URL}/metrics", timeout=30)
    assert response.status_code == 200
    assert "edf_requests_total" in response.text
    assert "edf_model_r2" in response.text


@pytest.mark.e2e
def test_api_invalid_date():
    """Une date invalide retourne 422."""
    response = requests.post(
        f"{API_URL}/predict",
        json={"date": "invalid"},
        timeout=30,
    )
    assert response.status_code == 422


@pytest.mark.e2e
def test_api_response_time():
    """L'API répond en moins de 2 secondes."""
    start = time.time()
    requests.post(
        f"{API_URL}/predict",
        json={"date": "2025-01-15", "prevision_j1": 65000},
        timeout=30,
    )
    duration = time.time() - start
    assert duration < 2.0, f"Temps de réponse trop long : {duration:.2f}s"
