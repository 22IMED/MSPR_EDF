"""Nettoyage, agrégation journalière et feature engineering des données RTE."""

import logging
import math

import pandas as pd

from preprocessing.constants import FEATURE_COLS, JOURS_FERIES_FR, SAISON_MAP

logger = logging.getLogger(__name__)

_RTE_COL_DATE = "Date"
_RTE_COL_CONSO = "Consommation"
_RTE_COL_PREVISION = "Prévision J-1"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame brut issu de l'extraction RTE.

    Opérations :
    - Conversion des colonnes numériques
    - Suppression des doublons (Date + Heures)
    - Suppression des lignes avec NaN sur Consommation
    - Suppression des valeurs aberrantes (z-score > 4)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut issu d'extract().

    Returns
    -------
    pd.DataFrame
        DataFrame nettoyé.
    """
    logger.info(f"Nettoyage : {len(df):,} lignes en entrée.")
    df = df.copy()

    # Conversion types
    for col in [_RTE_COL_CONSO, _RTE_COL_PREVISION]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Suppression doublons
    dup_cols = [c for c in ["Date", "Heures"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dup_cols if dup_cols else None)
    logger.debug(f"Doublons supprimés : {before - len(df):,}")

    # Suppression NaN sur consommation
    before = len(df)
    df = df.dropna(subset=[_RTE_COL_CONSO])
    logger.debug(f"NaN consommation supprimés : {before - len(df):,}")

    # Suppression valeurs aberrantes (consommation <= 0 ou > 200 000 MW)
    before = len(df)
    df = df[(df[_RTE_COL_CONSO] > 0) & (df[_RTE_COL_CONSO] < 200_000)]
    logger.debug(f"Valeurs aberrantes supprimées : {before - len(df):,}")

    # Z-score sur consommation (seuil 4)
    mean_c = df[_RTE_COL_CONSO].mean()
    std_c = df[_RTE_COL_CONSO].std()
    if std_c > 0:
        before = len(df)
        z_scores = (df[_RTE_COL_CONSO] - mean_c) / std_c
        df = df[z_scores.abs() <= 4]
        logger.debug(f"Outliers z-score supprimés : {before - len(df):,}")

    logger.info(f"Nettoyage terminé : {len(df):,} lignes restantes.")
    return df.reset_index(drop=True)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les données demi-horaires/horaires à la journée.

    La consommation et la prévision J-1 sont moyennées par jour.
    L'index du DataFrame résultant est un DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame nettoyé avec colonnes Date, Consommation, Prévision J-1.

    Returns
    -------
    pd.DataFrame
        DataFrame journalier avec colonnes consommation, prevision_j1,
        indexé par un DatetimeIndex trié.
    """
    logger.info("Agrégation journalière...")
    df = df.copy()

    # Parsing de la date
    df["_date"] = pd.to_datetime(df[_RTE_COL_DATE], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["_date"])

    agg = (
        df.groupby("_date")
        .agg(
            consommation=(_RTE_COL_CONSO, "mean"),
            prevision_j1=(_RTE_COL_PREVISION, "mean"),
        )
        .sort_index()
    )
    agg.index.name = "date"
    agg.index = pd.to_datetime(agg.index)

    # Interpolation linéaire pour prevision_j1 manquante
    agg["prevision_j1"] = agg["prevision_j1"].interpolate(
        method="linear", limit_direction="both"
    )

    logger.info(
        f"Agrégation terminée : {len(agg):,} jours, de {agg.index.min().date()} à {agg.index.max().date()}."
    )
    return agg


def _saison(month: int) -> int:
    """Retourne le code saison (0=Hiver, 1=Printemps, 2=Été, 3=Automne)."""
    return SAISON_MAP.get(month, 0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute toutes les features nécessaires à l'entraînement.

    Features ajoutées :
    - Temporelles : day_of_week, month, day_of_year, is_weekend
    - Calendaires : is_holiday, saison
    - Cycliques : month_sin, month_cos, dow_sin, dow_cos
    - Lags : lag_1 (J-1), lag_7 (J-7), prevision_j1_lag1

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame journalier issu d'aggregate_daily(), indexé par DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame enrichi des features FEATURE_COLS + TARGET_COL.
    """
    logger.info("Feature engineering...")
    df = df.copy()

    idx = df.index

    # Features temporelles de base
    df["day_of_week"] = idx.dayofweek  # 0=Lundi, 6=Dimanche
    df["month"] = idx.month
    df["day_of_year"] = idx.dayofyear
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Jours fériés
    date_strs = idx.strftime("%Y-%m-%d")
    df["is_holiday"] = [1 if d in JOURS_FERIES_FR else 0 for d in date_strs]

    # Saison
    df["saison"] = [_saison(m) for m in idx.month]

    # Features cycliques (encodage sinusoïdal)
    df["month_sin"] = df["month"].apply(lambda m: math.sin(2 * math.pi * m / 12))
    df["month_cos"] = df["month"].apply(lambda m: math.cos(2 * math.pi * m / 12))
    df["dow_sin"] = df["day_of_week"].apply(lambda d: math.sin(2 * math.pi * d / 7))
    df["dow_cos"] = df["day_of_week"].apply(lambda d: math.cos(2 * math.pi * d / 7))

    # Lags sur la consommation
    df["lag_1"] = df["consommation"].shift(1)
    df["lag_7"] = df["consommation"].shift(7)

    # Lag sur prévision J-1
    df["prevision_j1_lag1"] = df["prevision_j1"].shift(1)

    # Suppression des lignes avec NaN sur les lags (7 premières lignes)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    logger.debug(f"Lignes supprimées après lags : {before - len(df):,}")

    # Vérification que toutes les features sont présentes
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes après engineering : {missing}")

    logger.info(
        f"Feature engineering terminé : {len(df):,} lignes, {len(FEATURE_COLS)} features."
    )
    return df
