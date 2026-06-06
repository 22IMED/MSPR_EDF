"""Extraction des données brutes depuis les fichiers RTE eCO2mix (.xls)."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Colonnes attendues dans les fichiers RTE eCO2mix
_RTE_COL_DATE = "Date"
_RTE_COL_HEURE = "Heures"
_RTE_COL_CONSO = "Consommation"
_RTE_COL_PREVISION = "Prévision J-1"

# Alias supplémentaires pour robustesse
_COL_ALIASES: dict[str, list[str]] = {
    _RTE_COL_DATE: ["Date", "date", "DATE"],
    _RTE_COL_HEURE: ["Heures", "Heure", "heures", "heure", "HEURES"],
    _RTE_COL_CONSO: ["Consommation", "consommation", "CONSOMMATION", "Consomm."],
    _RTE_COL_PREVISION: [
        "Prévision J-1",
        "Prevision J-1",
        "prévision j-1",
        "Prévision J1",
    ],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes vers les noms canoniques RTE."""
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and alias != canonical:
                rename_map[alias] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _read_single_file(path: Path) -> pd.DataFrame:
    """
    Lit un fichier .xls RTE eCO2mix et retourne un DataFrame brut.

    Les fichiers RTE utilisent un séparateur point-virgule et contiennent
    des lignes d'en-tête supplémentaires à ignorer.

    Parameters
    ----------
    path : Path
        Chemin vers le fichier .xls

    Returns
    -------
    pd.DataFrame
        DataFrame brut avec les colonnes RTE canoniques.

    Raises
    ------
    ValueError
        Si les colonnes attendues ne sont pas trouvées dans le fichier.
    """
    logger.info(f"Lecture du fichier : {path}")

    # Tentative 1 : format XLS standard avec xlrd
    df: pd.DataFrame | None = None
    errors: list[str] = []

    # Liste des stratégies de lecture
    strategies = [
        dict(engine="xlrd", skiprows=0),
        dict(engine="xlrd", skiprows=1),
        dict(engine="xlrd", skiprows=2),
        dict(engine="openpyxl", skiprows=0),
        dict(engine="openpyxl", skiprows=1),
    ]

    for strategy in strategies:
        try:
            candidate = pd.read_excel(path, **strategy)
            candidate = _normalise_columns(candidate)
            if (
                _RTE_COL_DATE in candidate.columns
                and _RTE_COL_CONSO in candidate.columns
            ):
                df = candidate
                logger.debug(f"Lecture réussie avec {strategy}")
                break
        except Exception as exc:
            errors.append(f"{strategy}: {exc}")
            continue

    # Fallback : lecture CSV sep=";" ou tabulation
    if df is None:
        for sep in [";", "\t", ","]:
            try:
                candidate = pd.read_csv(
                    path,
                    sep=sep,
                    encoding="latin-1",
                    skiprows=0,
                    on_bad_lines="skip",
                )
                candidate = _normalise_columns(candidate)
                if (
                    _RTE_COL_DATE in candidate.columns
                    and _RTE_COL_CONSO in candidate.columns
                ):
                    df = candidate
                    logger.debug(f"Lecture réussie en mode CSV (sep={repr(sep)})")
                    break
            except Exception as exc:
                errors.append(f"CSV fallback sep={repr(sep)}: {exc}")

    if df is None:
        raise ValueError(
            f"Impossible de lire {path}. Colonnes attendues : "
            f"{[_RTE_COL_DATE, _RTE_COL_CONSO]}. Erreurs : {errors}"
        )

    # Vérification des colonnes minimales
    required = [_RTE_COL_DATE, _RTE_COL_CONSO]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path} : {missing}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    # Ajout colonne Prévision J-1 si absente
    if _RTE_COL_PREVISION not in df.columns:
        logger.warning(
            f"Colonne '{_RTE_COL_PREVISION}' absente dans {path}, remplissage NaN."
        )
        df[_RTE_COL_PREVISION] = float("nan")

    return df


def extract(paths: list[Path]) -> pd.DataFrame:
    """
    Lit et fusionne une liste de fichiers .xls RTE eCO2mix.

    Parameters
    ----------
    paths : list[Path]
        Liste des chemins vers les fichiers .xls à charger.

    Returns
    -------
    pd.DataFrame
        DataFrame brut fusionné avec les colonnes :
        Date, Heures, Consommation, Prévision J-1

    Raises
    ------
    FileNotFoundError
        Si aucun fichier n'est fourni ou si un fichier n'existe pas.
    ValueError
        Si les fichiers ne contiennent pas les colonnes attendues.
    """
    if not paths:
        raise FileNotFoundError("Aucun fichier .xls fourni pour l'extraction.")

    frames: list[pd.DataFrame] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")
        try:
            df = _read_single_file(path)
            df["_source_file"] = path.name
            frames.append(df)
            logger.info(f"  → {len(df):,} lignes chargées depuis {path.name}")
        except Exception as exc:
            logger.error(f"Erreur lors de la lecture de {path} : {exc}")
            raise

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Extraction terminée : {len(merged):,} lignes au total depuis {len(paths)} fichier(s)."
    )
    return merged
