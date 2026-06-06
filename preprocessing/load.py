"""Split chronologique train / val / test du dataset enrichi."""

import logging

import numpy as np
import pandas as pd

from preprocessing.constants import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

# Proportions du split (pas de shuffle — chronologique)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (implicite)


def load(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    """
    Découpe le DataFrame enrichi en ensembles train / validation / test.

    Le split est purement chronologique (pas de shuffle) pour respecter
    la nature temporelle des données de consommation électrique.

    Proportions : 70 % train / 15 % val / 15 % test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame enrichi issu d'engineer_features(), indexé par DatetimeIndex.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        Toutes les matrices X sont des np.ndarray de forme (n, len(FEATURE_COLS)).
        Les vecteurs y sont des np.ndarray 1D.
        feature_names est la liste ordonnée des features (= FEATURE_COLS).

    Raises
    ------
    ValueError
        Si le DataFrame est trop petit pour un split significatif.
    """
    # Vérification colonnes
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour le split : {missing}")

    n = len(df)
    if n < 30:
        raise ValueError(f"Dataset trop petit ({n} lignes) pour un split train/val/test.")

    # Tri chronologique
    df = df.sort_index()

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    logger.info(
        f"Split chronologique : train={n_train} ({TRAIN_RATIO*100:.0f}%), "
        f"val={n_val} ({VAL_RATIO*100:.0f}%), test={n_test} ({(1-TRAIN_RATIO-VAL_RATIO)*100:.0f}%)"
    )

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train = X[:n_train]
    X_val = X[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]

    y_train = y[:n_train]
    y_val = y[n_train : n_train + n_val]
    y_test = y[n_train + n_val :]

    # Log des plages de dates
    train_dates = df.index[:n_train]
    val_dates = df.index[n_train : n_train + n_val]
    test_dates = df.index[n_train + n_val :]
    logger.info(f"  Train : {train_dates.min().date()} → {train_dates.max().date()}")
    logger.info(f"  Val   : {val_dates.min().date()} → {val_dates.max().date()}")
    logger.info(f"  Test  : {test_dates.min().date()} → {test_dates.max().date()}")

    return X_train, X_val, X_test, y_train, y_val, y_test, list(FEATURE_COLS)