"""
API FastAPI — Prédiction de consommation électrique nationale française.

Endpoints :
- POST /predict  : prédiction pour une date donnée
- GET  /health   : santé de l'API + infos MLflow
- GET  /models   : liste des modèles disponibles
- GET  /metrics  : métriques Prometheus
"""

import json
import logging
import math
import os
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field, field_validator

from preprocessing.constants import FEATURE_COLS, JOURS_FERIES_FR, SAISON_MAP

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
_DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "random_forest")
_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "100"))
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ─── Application FastAPI ─────────────────────────────────────────────────────
app = FastAPI(
    title="EDF Consumption Prediction API",
    description="Pipeline ML de prédiction de la consommation électrique nationale française.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Rate limiting ────────────────────────────────────────────────────────────
_request_counts: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(ip: str) -> bool:
    """Retourne True si la limite est dépassée, False sinon."""
    now = time.time()
    window = 60.0
    counts = _request_counts[ip]
    # Purge des timestamps > 1 minute
    counts[:] = [t for t in counts if now - t < window]
    if len(counts) >= _RATE_LIMIT:
        return True
    counts.append(now)
    return False


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    if _check_rate_limit(ip):
        return Response(
            content='{"detail": "Too Many Requests"}',
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            media_type="application/json",
        )
    return await call_next(request)


# ─── Cache des modèles ────────────────────────────────────────────────────────
_model_cache: dict[str, object] = {}
_metrics_cache: dict[str, dict] = {}

# ─── Prometheus ───────────────────────────────────────────────────────────────
_prom_requests_total: dict[str, int] = defaultdict(int)
_prom_latency_sum: dict[str, float] = defaultdict(float)
_prom_errors_total: int = 0


def _load_model(model_name: str):
    """Charge un modèle depuis le cache ou le disque."""
    if model_name not in _model_cache:
        model_path = _MODELS_DIR / f"{model_name}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")
        _model_cache[model_name] = joblib.load(model_path)
        logger.info(f"Modèle chargé : {model_name}")
    return _model_cache[model_name]


def _load_metrics(model_name: str) -> dict:
    """Charge les métriques d'un modèle depuis le fichier .metrics.json."""
    if model_name not in _metrics_cache:
        # Essai du fichier latest
        paths_to_try = [
            _MODELS_DIR / f"{model_name}_latest.metrics.json",
            _MODELS_DIR / "random_forest_latest.metrics.json",
            _MODELS_DIR / f"{model_name}.metrics.json",
        ]
        for p in paths_to_try:
            if p.exists():
                with open(p) as f:
                    _metrics_cache[model_name] = json.load(f)
                break
        else:
            _metrics_cache[model_name] = {}
    return _metrics_cache[model_name]


# ─── Feature engineering (réplique de preprocessing/transform.py) ─────────────


def _build_features(
    target_date: date,
    prevision_j1: Optional[float],
    lag_1: Optional[float],
    lag_7: Optional[float],
    nucleaire: Optional[float] = None,
    eolien: Optional[float] = None,
    solaire: Optional[float] = None,
    hydraulique: Optional[float] = None,
    gaz: Optional[float] = None,
    fioul: Optional[float] = None,
    taux_co2: Optional[float] = None,
) -> np.ndarray:
    """
    Construit le vecteur de features pour une date donnée.

    Parameters
    ----------
    target_date : date
    prevision_j1 : float | None
    lag_1 : float | None
    lag_7 : float | None
    nucleaire : float | None
    eolien : float | None
    solaire : float | None
    hydraulique : float | None
    gaz : float | None
    fioul : float | None
    taux_co2 : float | None

    Returns
    -------
    np.ndarray shape (1, len(FEATURE_COLS))
    """
    dt = datetime.combine(target_date, datetime.min.time())
    month = dt.month
    dow = dt.weekday()
    day_of_year = dt.timetuple().tm_yday

    date_str = target_date.strftime("%Y-%m-%d")
    is_holiday = 1 if date_str in JOURS_FERIES_FR else 0
    is_weekend = 1 if dow >= 5 else 0
    saison = SAISON_MAP.get(month, 0)

    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    dow_sin = math.sin(2 * math.pi * dow / 7)
    dow_cos = math.cos(2 * math.pi * dow / 7)

    # Valeurs par défaut
    pj1 = prevision_j1 if prevision_j1 is not None else 55_000.0
    l1 = lag_1 if lag_1 is not None else 55_000.0
    l7 = lag_7 if lag_7 is not None else 55_000.0
    pj1_lag1 = pj1

    feature_map = {
        "prevision_j1": pj1,
        "day_of_week": float(dow),
        "month": float(month),
        "day_of_year": float(day_of_year),
        "is_weekend": float(is_weekend),
        "is_holiday": float(is_holiday),
        "saison": float(saison),
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "lag_1": l1,
        "lag_7": l7,
        "prevision_j1_lag1": pj1_lag1,
        "nucleaire": nucleaire if nucleaire is not None else 40_000.0,
        "eolien": eolien if eolien is not None else 5_000.0,
        "solaire": solaire if solaire is not None else 3_000.0,
        "hydraulique": hydraulique if hydraulique is not None else 8_000.0,
        "gaz": gaz if gaz is not None else 6_000.0,
        "fioul": fioul if fioul is not None else 500.0,
        "taux_co2": taux_co2 if taux_co2 is not None else 50.0,
    }

    return np.array([[feature_map[col] for col in FEATURE_COLS]])

# ─── Schémas Pydantic ────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    date: str = Field(..., description="Date de prédiction au format YYYY-MM-DD")
    prevision_j1: Optional[float] = Field(None, description="Prévision RTE J-1 (MW)", ge=0)
    lag_1: Optional[float] = Field(None, description="Consommation J-1 (MW)", ge=0)
    lag_7: Optional[float] = Field(None, description="Consommation J-7 (MW)", ge=0)
    nucleaire: Optional[float] = Field(None, description="Production nucléaire (MW)", ge=0)
    eolien: Optional[float] = Field(None, description="Production éolienne (MW)", ge=0)
    solaire: Optional[float] = Field(None, description="Production solaire (MW)", ge=0)
    hydraulique: Optional[float] = Field(None, description="Production hydraulique (MW)", ge=0)
    gaz: Optional[float] = Field(None, description="Production gaz (MW)", ge=0)
    fioul: Optional[float] = Field(None, description="Production fioul (MW)", ge=0)
    taux_co2: Optional[float] = Field(None, description="Taux de CO2 (g/kWh)", ge=0)
    model_name: Optional[str] = Field(None, description="Nom du modèle à utiliser")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            dt = datetime.strptime(v, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("La date doit être au format YYYY-MM-DD.")
        if dt.year < 2020:
            raise ValueError("La date doit être >= 2020-01-01.")
        return v


class PredictResponse(BaseModel):
    date: str
    prediction_mw: float
    model_name: str
    model_version: str
    r2_score: float
    mape_percent: float
    prevision_rte_j1_mw: Optional[float]
    latency_ms: float
    timestamp: str
    mlflow_run_id: str
    mlflow_registry_version: str
    mlflow_registry_stage: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    mlflow_run_id: str
    registry_version: str
    registry_stage: str
    timestamp: str


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Prédiction de la consommation électrique pour une date donnée."""
    global _prom_errors_total
    t_start = time.time()

    model_name = request.model_name or _DEFAULT_MODEL
    _prom_requests_total[model_name] += 1

    try:
        pipeline = _load_model(model_name)
    except FileNotFoundError as exc:
        _prom_errors_total += 1
        raise HTTPException(status_code=404, detail=str(exc))

    target_date = datetime.strptime(request.date, "%Y-%m-%d").date()
    features = _build_features(
        target_date,
        request.prevision_j1,
        request.lag_1,
        request.lag_7,
        request.nucleaire,
        request.eolien,
        request.solaire,
        request.hydraulique,
        request.gaz,
        request.fioul,
        request.taux_co2,
    )

    try:
        prediction = float(pipeline.predict(features)[0])
    except Exception as exc:
        _prom_errors_total += 1
        logger.error(f"Erreur prédiction : {exc}")
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction : {exc}"
        )

    latency_ms = (time.time() - t_start) * 1000
    _prom_latency_sum[model_name] += latency_ms

    metrics = _load_metrics(model_name)

    return PredictResponse(
        date=request.date,
        prediction_mw=round(prediction, 2),
        model_name=model_name,
        model_version=metrics.get("registry_version", "N/A"),
        r2_score=metrics.get("r2_score", 0.0),
        mape_percent=metrics.get("mape_percent", 0.0),
        prevision_rte_j1_mw=request.prevision_j1,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
        mlflow_run_id=metrics.get("mlflow_run_id", ""),
        mlflow_registry_version=metrics.get("registry_version", "N/A"),
        mlflow_registry_stage=metrics.get("registry_stage", "N/A"),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérification de l'état de l'API et des composants."""
    model_name = _DEFAULT_MODEL
    model_loaded = False
    try:
        _load_model(model_name)
        model_loaded = True
    except Exception:
        pass

    metrics = _load_metrics(model_name)
    api_status = "ok" if model_loaded else "degraded"

    return HealthResponse(
        status=api_status,
        model_loaded=model_loaded,
        model_name=model_name,
        mlflow_run_id=metrics.get("mlflow_run_id", ""),
        registry_version=metrics.get("registry_version", "N/A"),
        registry_stage=metrics.get("registry_stage", "N/A"),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/models")
async def list_models():
    """Liste les modèles disponibles avec leurs informations MLflow."""
    available = []
    for joblib_path in _MODELS_DIR.glob("*.joblib"):
        name = joblib_path.stem
        if name.endswith("_candidate"):
            continue
        metrics = _load_metrics(name)
        available.append(
            {
                "model_name": name,
                "path": str(joblib_path),
                "r2_score": metrics.get("r2_score"),
                "mape_percent": metrics.get("mape_percent"),
                "rmse_mw": metrics.get("rmse_mw"),
                "mlflow_run_id": metrics.get("mlflow_run_id"),
                "registry_version": metrics.get("registry_version"),
                "registry_stage": metrics.get("registry_stage"),
                "validated_at": metrics.get("validated_at"),
            }
        )
    return {"available_models": available, "count": len(available)}


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Métriques Prometheus pour monitoring."""
    lines = [
        "# HELP edf_requests_total Total des requêtes de prédiction par modèle",
        "# TYPE edf_requests_total counter",
    ]
    for model_name, count in _prom_requests_total.items():
        lines.append(f'edf_requests_total{{model="{model_name}"}} {count}')

    lines += [
        "# HELP edf_errors_total Total des erreurs de prédiction",
        "# TYPE edf_errors_total counter",
        f"edf_errors_total {_prom_errors_total}",
    ]

    # Métriques modèle depuis metrics.json
    metrics = _load_metrics(_DEFAULT_MODEL)
    r2 = metrics.get("r2_score", 0.0)
    mape = metrics.get("mape_percent", 0.0)
    rmse = metrics.get("rmse_mw", 0.0)

    lines += [
        "# HELP edf_model_r2 R² du modèle en production",
        "# TYPE edf_model_r2 gauge",
        f'edf_model_r2{{model="{_DEFAULT_MODEL}"}} {r2}',
        "# HELP edf_model_mape_percent MAPE (%) du modèle en production",
        "# TYPE edf_model_mape_percent gauge",
        f'edf_model_mape_percent{{model="{_DEFAULT_MODEL}"}} {mape}',
        "# HELP edf_model_mae_mw RMSE (MW) du modèle en production",
        "# TYPE edf_model_mae_mw gauge",
        f'edf_model_mae_mw{{model="{_DEFAULT_MODEL}"}} {rmse}',
    ]

    # Latence moyenne
    for model_name, total_lat in _prom_latency_sum.items():
        count = _prom_requests_total.get(model_name, 1)
        avg_lat = total_lat / count if count > 0 else 0
        lines.append(f'edf_latency_ms_avg{{model="{model_name}"}} {avg_lat:.2f}')

    return "\n".join(lines) + "\n"
