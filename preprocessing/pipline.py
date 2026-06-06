"""Pipeline complet ETL : extract → transform → load (orchestration locale)."""

import logging
from pathlib import Path

import pandas as pd

from preprocessing.extract import extract
from preprocessing.load import load
from preprocessing.transform import aggregate_daily, clean, engineer_features

logger = logging.getLogger(__name__)


def run_full_pipeline(
    xls_paths: list[Path],
) -> tuple:
    """
    Exécute l'intégralité du pipeline ETL depuis les fichiers .xls.

    Parameters
    ----------
    xls_paths : list[Path]
        Chemins vers les fichiers RTE eCO2mix.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, df_features)
        df_features : DataFrame journalier enrichi (pour inspection).
    """
    logger.info("=== PIPELINE ETL DÉMARRÉ ===")

    # 1. Extract
    df_raw = extract(xls_paths)

    # 2. Clean
    df_clean = clean(df_raw)

    # 3. Agrégation journalière
    df_daily = aggregate_daily(df_clean)

    # 4. Feature engineering
    df_features = engineer_features(df_daily)

    # 5. Split
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load(df_features)

    logger.info("=== PIPELINE ETL TERMINÉ ===")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names, df_features