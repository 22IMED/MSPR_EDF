"""Entraînement, évaluation et sélection des modèles ML."""

import json
import logging
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logging.warning("XGBoost non disponible — le modèle XGBRegressor sera ignoré.")

from preprocessing.constants import R2_THRESHOLD

logger = logging.getLogger(__name__)

# ─── Seuil de qualité ─────────────────────────────────────────────────────────
# (importé depuis constants mais redéfini ici pour accès direct)
R2_THRESHOLD = float(os.getenv("R2_THRESHOLD", str(R2_THRESHOLD)))


# ─── Fonctions de métriques ───────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le Mean Absolute Percentage Error (MAPE)."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule r2, rmse, mape."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
    }


# ─── Entraînement ─────────────────────────────────────────────────────────────

def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, dict]:
    """
    Entraîne les 4 modèles : RandomForest, GradientBoosting, Ridge, XGBoost.

    Chaque modèle est encapsulé dans un sklearn Pipeline avec StandardScaler
    (utile pour Ridge, transparent pour les arbres).

    Parameters
    ----------
    X_train : np.ndarray
        Matrice features d'entraînement.
    y_train : np.ndarray
        Vecteur cible d'entraînement.

    Returns
    -------
    dict
        {"model_name": {"pipeline": Pipeline, "train_time_s": float, "params": dict}}
    """
    model_configs = {
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        "ridge": Ridge(alpha=10.0),
    }

    if _XGBOOST_AVAILABLE:
        model_configs["xgboost"] = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    trained: dict[str, dict] = {}

    for name, estimator in model_configs.items():
        logger.info(f"Entraînement de {name}...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        t0 = time.time()
        try:
            pipeline.fit(X_train, y_train)
            duration = time.time() - t0
            params = estimator.get_params()
            trained[name] = {
                "pipeline": pipeline,
                "train_time_s": round(duration, 2),
                "params": params,
            }
            logger.info(f"  → {name} entraîné en {duration:.1f}s")
        except Exception as exc:
            logger.error(f"Erreur lors de l'entraînement de {name} : {exc}")

    return trained


# ─── Évaluation ───────────────────────────────────────────────────────────────

def evaluate_all(
    trained: dict[str, dict],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Évalue tous les modèles entraînés sur val et test.

    Parameters
    ----------
    trained : dict
        Sortie de train_all().
    X_val, y_val : np.ndarray
        Ensemble de validation.
    X_test, y_test : np.ndarray
        Ensemble de test.

    Returns
    -------
    pd.DataFrame
        Colonnes : model, r2_val, rmse_val, mape_val, r2_test, rmse_test, mape_test.
        Triée par r2_test décroissant.
    """
    rows = []
    for name, info in trained.items():
        pipeline = info["pipeline"]
        try:
            y_pred_val = pipeline.predict(X_val)
            y_pred_test = pipeline.predict(X_test)
            val_m = _compute_metrics(y_val, y_pred_val)
            test_m = _compute_metrics(y_test, y_pred_test)
            rows.append({
                "model": name,
                "r2_val": val_m["r2"],
                "rmse_val": val_m["rmse"],
                "mape_val": val_m["mape"],
                "r2_test": test_m["r2"],
                "rmse_test": test_m["rmse"],
                "mape_test": test_m["mape"],
            })
            logger.info(
                f"{name:25s} | val R²={val_m['r2']:.4f} MAPE={val_m['mape']:.2f}% "
                f"| test R²={test_m['r2']:.4f} MAPE={test_m['mape']:.2f}%"
            )
        except Exception as exc:
            logger.error(f"Erreur évaluation de {name} : {exc}")

    df_results = pd.DataFrame(rows)
    if not df_results.empty:
        df_results = df_results.sort_values("r2_test", ascending=False).reset_index(drop=True)
    return df_results


# ─── Sélection du meilleur modèle ─────────────────────────────────────────────

def select_best_model(
    results_df: pd.DataFrame,
    benchmark: dict | None = None,
) -> str | None:
    """
    Retourne le nom du meilleur modèle si r2_test > R2_THRESHOLD.

    Si un benchmark (prévision RTE) est fourni, le modèle doit aussi
    surpasser le benchmark en MAPE.

    Parameters
    ----------
    results_df : pd.DataFrame
        Sortie d'evaluate_all().
    benchmark : dict, optional
        Dict avec clés "mape" et "r2" (issu de benchmark_rte()).

    Returns
    -------
    str | None
        Nom du meilleur modèle, ou None si aucun ne passe le seuil.
    """
    if results_df.empty:
        logger.warning("Aucun résultat d'évaluation disponible.")
        return None

    best = results_df.iloc[0]
    if best["r2_test"] < R2_THRESHOLD:
        logger.warning(
            f"Meilleur modèle '{best['model']}' : R²={best['r2_test']:.4f} < seuil {R2_THRESHOLD}. "
            "Aucun modèle sélectionné."
        )
        return None

    if benchmark and "mape" in benchmark:
        if best["mape_test"] > benchmark["mape"]:
            logger.warning(
                f"MAPE du modèle ({best['mape_test']:.2f}%) > MAPE benchmark RTE ({benchmark['mape']:.2f}%). "
                "Le modèle ne surpasse pas la prévision RTE J-1."
            )
            # On sélectionne quand même si R² est bon

    logger.info(f"Meilleur modèle sélectionné : '{best['model']}' (R²={best['r2_test']:.4f})")
    return str(best["model"])


# ─── Benchmark RTE ────────────────────────────────────────────────────────────

def benchmark_rte(
    y_true: np.ndarray,
    y_prevision: np.ndarray,
) -> dict[str, float]:
    """
    Calcule les métriques de la prévision RTE J-1 comme baseline.

    Parameters
    ----------
    y_true : np.ndarray
        Consommation réelle (MW).
    y_prevision : np.ndarray
        Prévision J-1 RTE (MW).

    Returns
    -------
    dict
        {"r2": float, "rmse": float, "mape": float}
    """
    # Masque des valeurs non-NaN dans les deux séries
    mask = ~(np.isnan(y_true) | np.isnan(y_prevision))
    if mask.sum() == 0:
        logger.warning("Benchmark RTE : aucune paire valide (tout NaN).")
        return {"r2": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    metrics = _compute_metrics(y_true[mask], y_prevision[mask])
    logger.info(
        f"Benchmark RTE J-1 : R²={metrics['r2']:.4f}, "
        f"RMSE={metrics['rmse']:.1f} MW, MAPE={metrics['mape']:.2f}%"
    )
    return metrics


# ─── Sauvegarde conditionnelle ─────────────────────────────────────────────────

def save_model_if_better(
    pipeline: Pipeline,
    model_name: str,
    metrics: dict[str, float],
    models_dir: str = "models/",
) -> Path | None:
    """
    Sauvegarde le pipeline en joblib seulement s'il est meilleur que l'existant.

    Compare le R² test avec le modèle déjà sur disque (si présent).
    Écrit aussi un fichier <model_name>.metrics.json.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline sklearn entraîné.
    model_name : str
        Nom du modèle (ex: "random_forest").
    metrics : dict
        Dict avec au moins "r2_test", "rmse_test", "mape_test".
    models_dir : str
        Répertoire de sauvegarde.

    Returns
    -------
    Path | None
        Chemin du fichier .joblib sauvegardé, ou None si ignoré.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    joblib_path = models_path / f"{model_name}.joblib"
    metrics_path = models_path / f"{model_name}.metrics.json"

    new_r2 = metrics.get("r2_test", 0.0)

    # Chargement des métriques existantes
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                existing = json.load(f)
            existing_r2 = existing.get("r2_score", existing.get("r2_test", 0.0))
            if new_r2 <= existing_r2:
                logger.info(
                    f"Modèle existant meilleur (R²={existing_r2:.4f} vs {new_r2:.4f}). "
                    "Sauvegarde ignorée."
                )
                return None
            logger.info(f"Amélioration détectée (R²: {existing_r2:.4f} → {new_r2:.4f}).")
        except Exception as exc:
            logger.warning(f"Impossible de lire les métriques existantes : {exc}")

    # Sauvegarde du modèle
    joblib.dump(pipeline, joblib_path)
    logger.info(f"Modèle sauvegardé : {joblib_path}")

    # Sauvegarde des métriques
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Métriques sauvegardées : {metrics_path}")

    return joblib_path