"""
Orchestration du pipeline ML en 10 étapes.

Chaque fonction peut être appelée indépendamment :
    python -c "from pipeline.steps import run_train; run_train()"
"""

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
_DATA_SOURCE = os.getenv("DATA_SOURCE", "xls")
_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
_MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
_MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
_MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "edf-consumption")
_MAPE_PROD_THRESHOLD = float(os.getenv("MAPE_PRODUCTION_THRESHOLD", "5.0"))
_R2_THRESHOLD = float(os.getenv("R2_THRESHOLD", "0.85"))

# ─── Imports internes ─────────────────────────────────────────────────────────
from models import (
    benchmark_rte,
    evaluate_all,
    save_model_if_better,
    select_best_model,
    train_all,
)
from pipeline.artifacts import (
    artifact_path,
    load_meta,
    load_parquet,
    load_splits,
    save_meta,
    save_parquet,
    save_splits,
)
from pipeline.snowflake_io import SnowflakeUnavailableError, load_from_snowflake, write_predictions_to_snowflake
from preprocessing.constants import FEATURE_COLS, TARGET_COL
from preprocessing.extract import extract
from preprocessing.load import load
from preprocessing.transform import aggregate_daily, clean, engineer_features


# ─── Helpers MLflow run ID ────────────────────────────────────────────────────

def _save_mlflow_run_id(mlflow_run_id: str, run_id: str | None) -> None:
    path = artifact_path("mlflow_run_id.txt", run_id)
    path.write_text(mlflow_run_id)
    logger.info(f"MLflow run_id sauvegardé : {mlflow_run_id}")


def _load_mlflow_run_id(run_id: str | None) -> str | None:
    path = artifact_path("mlflow_run_id.txt", run_id)
    if path.exists():
        return path.read_text().strip()
    return None


# ─── Step 1 : Extraction ─────────────────────────────────────────────────────

def run_extract(run_id: str | None = None) -> str:
    """
    Étape 1 — Extraction des données brutes.

    Si DATA_SOURCE=snowflake → charge via Snowflake.
    Sinon → charge les fichiers .xls depuis DATA_DIR.
    Sauvegarde le résultat en raw.parquet.

    Parameters
    ----------
    run_id : str | None
        Identifiant du run pipeline.

    Returns
    -------
    str
        run_id utilisé (généré si None).
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
        logger.info(f"Nouveau run_id généré : {run_id}")

    logger.info("=== STEP 1 : EXTRACTION ===")

    if _DATA_SOURCE.lower() == "snowflake":
        logger.info("Source : Snowflake")
        try:
            df = load_from_snowflake()
            # DataFrame Snowflake déjà agrégé → on le sauvegarde directement
            save_parquet(df, "raw.parquet", run_id)
            save_parquet(df, "daily.parquet", run_id)
            save_meta({"source": "snowflake", "run_id": run_id, "n_rows": len(df)}, run_id)
            logger.info(f"Extraction Snowflake terminée : {len(df):,} lignes.")
            return run_id
        except SnowflakeUnavailableError as exc:
            logger.error(f"Snowflake indisponible : {exc}")
            raise

    # Source XLS
    logger.info(f"Source : fichiers .xls dans {_DATA_DIR}")
    xls_files = list(_DATA_DIR.glob("*.xls")) + list(_DATA_DIR.glob("*.xlsx"))
    if not xls_files:
        raise FileNotFoundError(f"Aucun fichier .xls trouvé dans {_DATA_DIR.resolve()}")

    df_raw = extract(xls_files)
    # Nettoyer les colonnes mixtes
    for col in df_raw.columns:
        if df_raw[col].dtype == object:
            df_raw[col] = df_raw[col].astype(str).replace({'nan': None, 'ND': None})
    save_parquet(df_raw, "raw.parquet", run_id)
    save_meta({"source": "xls", "run_id": run_id, "n_rows": len(df_raw), "files": [f.name for f in xls_files]}, run_id)
    logger.info(f"Extraction XLS terminée : {len(df_raw):,} lignes.")
    return run_id


# ─── Step 2 : Nettoyage ──────────────────────────────────────────────────────

def run_clean(run_id: str | None = None) -> None:
    """
    Étape 2 — Nettoyage des données brutes.

    Charge raw.parquet, applique clean(), sauvegarde cleaned.parquet.
    """
    logger.info("=== STEP 2 : NETTOYAGE ===")
    meta = load_meta(run_id)

    if meta.get("source") in ("snowflake",):
        logger.info("Source Snowflake : nettoyage allégé (données déjà propres).")
        df = load_parquet("raw.parquet", run_id)
        save_parquet(df, "cleaned.parquet", run_id)
        return

    df_raw = load_parquet("raw.parquet", run_id)
    df_clean = clean(df_raw)
    save_parquet(df_clean, "cleaned.parquet", run_id)
    logger.info(f"Nettoyage terminé : {len(df_clean):,} lignes.")


# ─── Step 3 : Agrégation ─────────────────────────────────────────────────────

def run_aggregate(run_id: str | None = None) -> None:
    """
    Étape 3 — Agrégation journalière.

    Charge cleaned.parquet, applique aggregate_daily(), sauvegarde daily.parquet.
    Pour source Snowflake, daily.parquet est déjà créé à l'étape 1.
    """
    logger.info("=== STEP 3 : AGRÉGATION JOURNALIÈRE ===")
    meta = load_meta(run_id)

    if meta.get("source") == "snowflake":
        logger.info("Source Snowflake : agrégation déjà effectuée.")
        return

    df_clean = load_parquet("cleaned.parquet", run_id)
    df_daily = aggregate_daily(df_clean)
    save_parquet(df_daily, "daily.parquet", run_id)
    logger.info(f"Agrégation terminée : {len(df_daily):,} jours.")


# ─── Step 4 : Feature engineering ────────────────────────────────────────────

def run_features(run_id: str | None = None) -> None:
    """
    Étape 4 — Feature engineering.

    Charge daily.parquet, applique engineer_features(), sauvegarde features.parquet.
    """
    logger.info("=== STEP 4 : FEATURE ENGINEERING ===")
    df_daily = load_parquet("daily.parquet", run_id)

    # S'assurer que l'index est un DatetimeIndex
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        df_daily.index = pd.to_datetime(df_daily.index)

    df_features = engineer_features(df_daily)
    save_parquet(df_features, "features.parquet", run_id)
    logger.info(f"Feature engineering terminé : {len(df_features):,} lignes, {len(FEATURE_COLS)} features.")


# ─── Step 5 : Split train/val/test ───────────────────────────────────────────

def run_load_splits(run_id: str | None = None) -> None:
    """
    Étape 5 — Découpage chronologique train / val / test.

    Charge features.parquet, applique load(), sauvegarde les splits.
    """
    logger.info("=== STEP 5 : SPLIT TRAIN/VAL/TEST ===")
    df_features = load_parquet("features.parquet", run_id)

    if not isinstance(df_features.index, pd.DatetimeIndex):
        df_features.index = pd.to_datetime(df_features.index)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load(df_features)
    save_splits(X_train, X_val, X_test, y_train, y_val, y_test, feature_names, run_id)
    logger.info(
        f"Splits sauvegardés : train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )


# ─── Step 6 : Entraînement ───────────────────────────────────────────────────

def run_train(run_id: str | None = None) -> None:
    """
    Étape 6 — Entraînement des modèles avec tracking MLflow.

    - Charge les splits
    - Entraîne les 4 modèles
    - Log dans MLflow (params, métriques train, modèles)
    - Sauvegarde le mlflow_run_id
    """
    logger.info("=== STEP 6 : ENTRAÎNEMENT ===")

    splits = load_splits(run_id)
    X_train = splits["x_train"]
    y_train = splits["y_train"]
    feature_names = splits["feature_names"]

    import mlflow
    import mlflow.sklearn
    from sklearn.metrics import r2_score

    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    t0 = time.time()
    trained = train_all(X_train, y_train)
    train_duration = time.time() - t0

    with mlflow.start_run(run_name=f"train-{run_id or 'default'}") as mlrun:
        mlflow_run_id = mlrun.info.run_id
        _save_mlflow_run_id(mlflow_run_id, run_id)

        # Params globaux
        mlflow.log_params({
            "n_features": len(feature_names),
            "n_train": len(X_train),
            "feature_names": ",".join(feature_names),
            "data_source": _DATA_SOURCE,
        })
        mlflow.log_metric("train_duration_s", round(train_duration, 2))

        # Log de chaque modèle
        for name, info in trained.items():
            pipeline = info["pipeline"]
            y_pred_train = pipeline.predict(X_train)
            r2 = float(r2_score(y_train, y_pred_train))

            # Params modèle
            model_params = {f"{name}__{k}": v for k, v in info["params"].items()
                            if isinstance(v, (int, float, str, bool))}
            mlflow.log_params(model_params)
            mlflow.log_metrics({
                f"{name}_r2_train": r2,
                f"{name}_train_time_s": info["train_time_s"],
            })

            # Log du modèle sklearn
            try:
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path=f"models/{name}",
                    registered_model_name=None,
                )
                logger.info(f"Modèle {name} loggé dans MLflow.")
            except Exception as exc:
                logger.warning(f"MLflow log_model {name} échoué : {exc}")

        logger.info(f"MLflow run_id : {mlflow_run_id}")

    # Sauvegarde locale des modèles entraînés (pour étape 7)
    trained_meta = {
        name: {"train_time_s": info["train_time_s"]}
        for name, info in trained.items()
    }
    import joblib
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, info in trained.items():
        joblib.dump(info["pipeline"], _MODELS_DIR / f"{name}_candidate.joblib")

    logger.info(f"Entraînement terminé : {len(trained)} modèles, {train_duration:.1f}s.")


# ─── Step 7 : Validation ─────────────────────────────────────────────────────

def run_validate(run_id: str | None = None) -> str:
    """
    Étape 7 — Évaluation et validation du meilleur modèle.

    - Évalue tous les modèles sur val et test
    - Reprend le run MLflow existant pour logger les métriques
    - Lance pytest automatiquement
    - Lève RuntimeError si R² < seuil ou si pytest échoue

    Returns
    -------
    str
        Nom du meilleur modèle.
    """
    logger.info("=== STEP 7 : VALIDATION ===")
    import joblib
    import mlflow

    splits = load_splits(run_id)
    X_val = splits["x_val"]
    y_val = splits["y_val"]
    X_test = splits["x_test"]
    y_test = splits["y_test"]

    # Chargement des modèles candidats
    trained = {}
    for joblib_path in _MODELS_DIR.glob("*_candidate.joblib"):
        name = joblib_path.stem.replace("_candidate", "")
        try:
            trained[name] = {"pipeline": joblib.load(joblib_path)}
        except Exception as exc:
            logger.warning(f"Impossible de charger {joblib_path} : {exc}")

    if not trained:
        raise RuntimeError("Aucun modèle candidat trouvé dans models/. Relancer run_train().")

    results_df = evaluate_all(trained, X_val, y_val, X_test, y_test)
    best_model_name = select_best_model(results_df)

    if best_model_name is None:
        raise RuntimeError(
            f"Aucun modèle ne dépasse le seuil R²={_R2_THRESHOLD}. "
            f"Meilleur : {results_df.iloc[0].to_dict() if not results_df.empty else 'N/A'}"
        )

    best_row = results_df[results_df["model"] == best_model_name].iloc[0]

    # Sauvegarde des résultats d'évaluation
    save_parquet(results_df, "evaluation_results.parquet", run_id)
    save_meta({
        **load_meta(run_id),
        "best_model": best_model_name,
        "r2_test": float(best_row["r2_test"]),
        "rmse_test": float(best_row["rmse_test"]),
        "mape_test": float(best_row["mape_test"]),
        "r2_val": float(best_row["r2_val"]),
        "rmse_val": float(best_row["rmse_val"]),
        "mape_val": float(best_row["mape_val"]),
    }, run_id)

    # Log dans MLflow (run existant)
    mlflow_run_id = _load_mlflow_run_id(run_id)
    if mlflow_run_id:
        try:
            mlflow.set_tracking_uri(_MLFLOW_URI)
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_metrics({
                    "best_r2_val": float(best_row["r2_val"]),
                    "best_rmse_val": float(best_row["rmse_val"]),
                    "best_mape_val": float(best_row["mape_val"]),
                    "best_r2_test": float(best_row["r2_test"]),
                    "best_rmse_test": float(best_row["rmse_test"]),
                    "best_mape_test": float(best_row["mape_test"]),
                })
                mlflow.log_param("best_model", best_model_name)
        except Exception as exc:
            logger.warning(f"MLflow log validation échoué : {exc}")

    # Lancement des tests pytest
    logger.info("Lancement des tests pytest...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode not in (0, 5):  # 5 = aucun test collecté
            logger.error(f"Tests pytest échoués :\n{result.stdout}\n{result.stderr}")
            raise RuntimeError(f"Tests pytest échoués (code {result.returncode}).")
        logger.info("Tests pytest : OK.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout pytest (120s).")
    except FileNotFoundError:
        logger.warning("pytest non trouvé — tests ignorés.")

    logger.info(f"Validation OK : meilleur modèle = '{best_model_name}' (R²={best_row['r2_test']:.4f})")
    return best_model_name


# ─── Step 8 : Enregistrement MLflow ──────────────────────────────────────────

def run_register_model(run_id: str | None = None) -> None:
    """
    Étape 8 — Enregistrement du meilleur modèle dans MLflow Model Registry.

    - Enregistre sous "edf-random-forest"
    - Transition Staging systématiquement
    - Transition Production si mape_test <= MAPE_PRODUCTION_THRESHOLD
    - Archive les anciennes versions Production
    - Écrit models/<model_name>_latest.metrics.json
    """
    logger.info("=== STEP 8 : ENREGISTREMENT MODÈLE ===")
    import joblib
    import mlflow
    from mlflow import MlflowClient

    meta = load_meta(run_id)
    best_model_name = meta.get("best_model")
    if not best_model_name:
        raise RuntimeError("Aucun best_model dans meta. Relancer run_validate().")

    mlflow_run_id = _load_mlflow_run_id(run_id)
    if not mlflow_run_id:
        raise RuntimeError("MLflow run_id introuvable. Relancer run_train().")

    mlflow.set_tracking_uri(_MLFLOW_URI)
    client = MlflowClient()
    registry_name = "edf-random-forest"

    # Chargement et sauvegarde du meilleur modèle
    candidate_path = _MODELS_DIR / f"{best_model_name}_candidate.joblib"
    pipeline = joblib.load(candidate_path)
    metrics_dict = {
        "r2_score": meta["r2_test"],
        "mape_percent": meta["mape_test"],
        "rmse_mw": meta["rmse_test"],
        "mlflow_run_id": mlflow_run_id,
        "mlflow_tracking_uri": _MLFLOW_URI,
        "model_name": best_model_name,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Sauvegarde locale
    final_path = save_model_if_better(
        pipeline, best_model_name, metrics_dict, str(_MODELS_DIR)
    )

    # Enregistrement dans MLflow Registry
    try:
        model_uri = f"runs:/{mlflow_run_id}/models/{best_model_name}"
        mv = mlflow.register_model(model_uri, registry_name)
        version = mv.version
        logger.info(f"Modèle enregistré dans MLflow Registry : {registry_name} v{version}")

        # Transition Staging
        client.transition_model_version_stage(
            name=registry_name, version=version, stage="Staging", archive_existing_versions=False
        )
        logger.info(f"Modèle transitionné vers Staging (v{version}).")
        registry_stage = "Staging"

        # Transition Production si MAPE OK
        if meta["mape_test"] <= _MAPE_PROD_THRESHOLD:
            # Archive anciennes versions Production
            try:
                for mv_old in client.get_latest_versions(registry_name, stages=["Production"]):
                    client.transition_model_version_stage(
                        name=registry_name, version=mv_old.version,
                        stage="Archived", archive_existing_versions=False
                    )
                    logger.info(f"Ancienne version Production archivée : v{mv_old.version}")
            except Exception as exc:
                logger.warning(f"Archive anciennes versions échoué : {exc}")

            client.transition_model_version_stage(
                name=registry_name, version=version, stage="Production", archive_existing_versions=False
            )
            registry_stage = "Production"
            logger.info(f"Modèle transitionné vers Production (MAPE={meta['mape_test']:.2f}% ≤ {_MAPE_PROD_THRESHOLD}%).")
        else:
            logger.info(
                f"Modèle maintenu en Staging (MAPE={meta['mape_test']:.2f}% > {_MAPE_PROD_THRESHOLD}%)."
            )

    except Exception as exc:
        logger.error(f"MLflow Registry indisponible : {exc}")
        version = "N/A"
        registry_stage = "local-only"

    # Écriture du fichier metrics.json
    metrics_json = {
        "r2_score": meta["r2_test"],
        "mape_percent": meta["mape_test"],
        "rmse_mw": meta["rmse_test"],
        "mlflow_run_id": mlflow_run_id,
        "mlflow_tracking_uri": _MLFLOW_URI,
        "registry_version": str(version),
        "registry_stage": registry_stage,
        "registered_model": registry_name,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": best_model_name,
    }
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = _MODELS_DIR / f"{best_model_name}_latest.metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Métriques écrites : {metrics_file}")

    # Alias random_forest_latest.metrics.json pour l'API
    alias = _MODELS_DIR / "random_forest_latest.metrics.json"
    with open(alias, "w") as f:
        json.dump(metrics_json, f, indent=2)


# ─── Step 9 : Prédictions ────────────────────────────────────────────────────

def run_predictions(run_id: str | None = None) -> pd.DataFrame:
    """
    Étape 9 — Génération des prédictions sur l'ensemble de test.

    Charge le meilleur modèle, génère les prédictions,
    sauvegarde predictions.parquet.

    Returns
    -------
    pd.DataFrame
        DataFrame des prédictions.
    """
    logger.info("=== STEP 9 : PRÉDICTIONS ===")
    import joblib

    meta = load_meta(run_id)
    best_model_name = meta.get("best_model", "random_forest")

    # Chargement modèle
    model_path = _MODELS_DIR / f"{best_model_name}.joblib"
    if not model_path.exists():
        model_path = _MODELS_DIR / f"{best_model_name}_candidate.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    pipeline = joblib.load(model_path)
    splits = load_splits(run_id)
    X_test = splits["x_test"]
    y_test = splits["y_test"]

    y_pred = pipeline.predict(X_test)

    df_features = load_parquet("features.parquet", run_id)
    if not isinstance(df_features.index, pd.DatetimeIndex):
        df_features.index = pd.to_datetime(df_features.index)

    n_test = len(X_test)
    test_dates = df_features.index[-n_test:]

    df_pred = pd.DataFrame({
        "prediction_date": test_dates,
        "prediction_mw": y_pred,
        "actual_mw": y_test,
        "model_name": best_model_name,
        "pipeline_run_id": run_id or "default",
    })

    # Ajout prévision RTE J-1 si disponible
    if "prevision_j1" in df_features.columns:
        df_pred["prevision_rte_j1_mw"] = df_features["prevision_j1"].values[-n_test:]

    save_parquet(df_pred, "predictions.parquet", run_id)
    logger.info(f"Prédictions générées : {len(df_pred)} dates.")
    return df_pred


# ─── Step 10 : Écriture Snowflake ────────────────────────────────────────────

def run_write_snowflake(run_id: str | None = None) -> int:
    """
    Étape 10 — Écriture des prédictions dans Snowflake.

    Charge predictions.parquet et écrit dans EDF_PREDICTIONS.
    Si Snowflake est indisponible, log un warning sans bloquer.

    Returns
    -------
    int
        Nombre de lignes écrites (0 si Snowflake indisponible).
    """
    logger.info("=== STEP 10 : ÉCRITURE SNOWFLAKE ===")

    if _DATA_SOURCE.lower() != "snowflake" and not os.getenv("FORCE_SNOWFLAKE_WRITE"):
        logger.info("DATA_SOURCE != snowflake et FORCE_SNOWFLAKE_WRITE non défini. Étape ignorée.")
        return 0

    try:
        df_pred = load_parquet("predictions.parquet", run_id)
    except FileNotFoundError:
        logger.warning("predictions.parquet introuvable. Relancer run_predictions().")
        return 0

    try:
        n = write_predictions_to_snowflake(df_pred)
        logger.info(f"Snowflake : {n} prédictions écrites.")
        return n
    except SnowflakeUnavailableError as exc:
        logger.warning(f"Snowflake indisponible, prédictions non écrites : {exc}")
        return 0
    except Exception as exc:
        logger.error(f"Erreur inattendue Snowflake : {exc}")
        return 0