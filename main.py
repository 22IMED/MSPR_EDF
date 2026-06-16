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
from datetime import timedelta

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
    heure: float = 12.0,
    lag_1h: Optional[float] = None,
    lag_24h: Optional[float] = None,
    lag_48h: Optional[float] = None,
    lag_j7: Optional[float] = None,
    roll_24h_mean: Optional[float] = None,
    roll_24h_std: Optional[float] = None,
    roll_7j_mean: Optional[float] = None,
    delta_j1: Optional[float] = None,
    charbon: Optional[float] = None,
    fioul: Optional[float] = None,
    gaz: Optional[float] = None,
    eolien: Optional[float] = None,
    solaire: Optional[float] = None,
    hydraulique: Optional[float] = None,
    nucleaire: Optional[float] = None,
    ech_physiques: Optional[float] = None,
) -> np.ndarray:
    """Construit le vecteur de 27 features pour le modèle MLP."""

    month = target_date.month
    dow = target_date.weekday()
    day_of_year = target_date.timetuple().tm_yday
    is_weekend = 1 if dow >= 5 else 0
    is_heure_pointe = 1 if int(heure) in [8, 9, 10, 11, 18, 19, 20] else 0

    def saison(m):
        if m in [12, 1, 2]: return 1
        if m in [3, 4, 5]:  return 2
        if m in [6, 7, 8]:  return 3
        return 4

    feature_map = {
        "Mois_sin":        math.sin(2 * math.pi * month / 12),
        "Mois_cos":        math.cos(2 * math.pi * month / 12),
        "Jour_sin":        math.sin(2 * math.pi * dow / 7),
        "Jour_cos":        math.cos(2 * math.pi * dow / 7),
        "Heure_sin":       math.sin(2 * math.pi * heure / 24),
        "Heure_cos":       math.cos(2 * math.pi * heure / 24),
        "JourAnnee_sin":   math.sin(2 * math.pi * day_of_year / 365),
        "JourAnnee_cos":   math.cos(2 * math.pi * day_of_year / 365),
        "Est_Weekend":     float(is_weekend),
        "Est_Heure_Pointe": float(is_heure_pointe),
        "Saison":          float(saison(month)),
        "Lag_1h":          lag_1h if lag_1h is not None else 55_000.0,
        "Lag_24h":         lag_24h if lag_24h is not None else 55_000.0,
        "Lag_48h":         lag_48h if lag_48h is not None else 55_000.0,
        "Lag_J7":          lag_j7 if lag_j7 is not None else 55_000.0,
        "Roll_24h_mean":   roll_24h_mean if roll_24h_mean is not None else 55_000.0,
        "Roll_24h_std":    roll_24h_std if roll_24h_std is not None else 2_000.0,
        "Roll_7j_mean":    roll_7j_mean if roll_7j_mean is not None else 55_000.0,
        "Delta_J1":        delta_j1 if delta_j1 is not None else 0.0,
        "Charbon":         charbon if charbon is not None else 500.0,
        "Fioul":           fioul if fioul is not None else 500.0,
        "Gaz":             gaz if gaz is not None else 6_000.0,
        "Eolien":          eolien if eolien is not None else 5_000.0,
        "Solaire":         solaire if solaire is not None else 3_000.0,
        "Hydraulique":     hydraulique if hydraulique is not None else 8_000.0,
        "Nucléaire":       nucleaire if nucleaire is not None else 40_000.0,
        "Ech. physiques":  ech_physiques if ech_physiques is not None else -2_000.0,
    }

    FEATURES = [
        "Mois_sin", "Mois_cos", "Jour_sin", "Jour_cos",
        "Heure_sin", "Heure_cos", "JourAnnee_sin", "JourAnnee_cos",
        "Est_Weekend", "Est_Heure_Pointe", "Saison",
        "Lag_1h", "Lag_24h", "Lag_48h", "Lag_J7",
        "Roll_24h_mean", "Roll_24h_std", "Roll_7j_mean", "Delta_J1",
        "Charbon", "Fioul", "Gaz", "Eolien",
        "Solaire", "Hydraulique", "Nucléaire", "Ech. physiques",
    ]

    return np.array([[feature_map[col] for col in FEATURES]])


# ─── Schémas Pydantic ────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    date: str = Field(..., description="Date de prédiction au format YYYY-MM-DD")
    heure: Optional[float] = Field(12.0, description="Heure de la journée (0-23)", ge=0, le=23)
    lag_1h: Optional[float] = Field(None, description="Consommation il y a 1h (MW)", ge=0)
    lag_24h: Optional[float] = Field(None, description="Consommation il y a 24h (MW)", ge=0)
    lag_48h: Optional[float] = Field(None, description="Consommation il y a 48h (MW)", ge=0)
    lag_j7: Optional[float] = Field(None, description="Consommation il y a 7 jours (MW)", ge=0)
    roll_24h_mean: Optional[float] = Field(None, description="Moyenne rolling 24h (MW)", ge=0)
    roll_24h_std: Optional[float] = Field(None, description="Écart-type rolling 24h (MW)", ge=0)
    roll_7j_mean: Optional[float] = Field(None, description="Moyenne rolling 7 jours (MW)", ge=0)
    delta_j1: Optional[float] = Field(None, description="Delta consommation J-1 (MW)")
    charbon: Optional[float] = Field(None, description="Production charbon (MW)")
    fioul: Optional[float] = Field(None, description="Production fioul (MW)", ge=0)
    gaz: Optional[float] = Field(None, description="Production gaz (MW)", ge=0)
    eolien: Optional[float] = Field(None, description="Production éolienne (MW)", ge=0)
    solaire: Optional[float] = Field(None, description="Production solaire (MW)", ge=0)
    hydraulique: Optional[float] = Field(None, description="Production hydraulique (MW)", ge=0)
    nucleaire: Optional[float] = Field(None, description="Production nucléaire (MW)", ge=0)
    ech_physiques: Optional[float] = Field(None, description="Échanges physiques (MW)")
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


class ForecastRequest(BaseModel):
    start_date: str = Field(..., description="Date début YYYY-MM-DD")
    end_date: str = Field(..., description="Date fin YYYY-MM-DD")
    model_name: Optional[str] = Field(None, description="Nom du modèle")

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Format YYYY-MM-DD requis.")
        return v


class ForecastPoint(BaseModel):
    date: str
    prediction_mw: float


class ForecastResponse(BaseModel):
    start_date: str
    end_date: str
    model_name: str
    predictions: list[ForecastPoint]
    count: int
    r2_score: float
    mape_percent: float


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """Prédictions sur une plage de dates."""
    start = datetime.strptime(request.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(request.end_date, "%Y-%m-%d").date()

    if end < start:
        raise HTTPException(status_code=400, detail="end_date doit être >= start_date.")
    if (end - start).days > 365:
        raise HTTPException(
            status_code=400, detail="La plage ne peut pas dépasser 365 jours."
        )

    model_name = request.model_name or _DEFAULT_MODEL
    try:
        pipeline = _load_model(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    predictions = []
    current = start
    while current <= end:
        features = _build_features(current)
        pred = float(pipeline.predict(features)[0])
        predictions.append(
            ForecastPoint(
                date=current.strftime("%Y-%m-%d"),
                prediction_mw=round(pred, 2),
            )
        )
        current += timedelta(days=1)

    metrics = _load_metrics(model_name)
    return ForecastResponse(
        start_date=request.start_date,
        end_date=request.end_date,
        model_name=model_name,
        predictions=predictions,
        count=len(predictions),
        r2_score=metrics.get("r2_score", 0.0),
        mape_percent=metrics.get("mape_percent", 0.0),
    )
