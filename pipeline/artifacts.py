"""Gestion des artifacts du pipeline : sauvegarde locale et Azure Blob optionnelle."""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
_USE_AZURE = os.getenv("USE_AZURE_STORAGE", "false").lower() == "true"
_ARTIFACTS_BASE = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
_AZURE_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER_ARTIFACTS", "mlflow-artifacts")


# ─── Azure Blob (lazy import) ─────────────────────────────────────────────────

def _get_blob_client(blob_name: str):
    """Retourne un BlobClient Azure configuré."""
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        account = os.getenv("AZURE_STORAGE_ACCOUNT")
        key = os.getenv("AZURE_STORAGE_KEY")
        if conn_str:
            svc = BlobServiceClient.from_connection_string(conn_str)
        elif account and key:
            svc = BlobServiceClient(
                account_url=f"https://{account}.blob.core.windows.net",
                credential=key,
            )
        else:
            raise EnvironmentError("Azure Storage non configuré (CONNECTION_STRING ou ACCOUNT+KEY requis).")
        return svc.get_blob_client(container=_AZURE_CONTAINER, blob=blob_name)
    except ImportError:
        raise ImportError("azure-storage-blob non installé. pip install azure-storage-blob")


def _upload_to_azure(local_path: Path, blob_name: str) -> None:
    """Uploade un fichier local vers Azure Blob Storage."""
    try:
        client = _get_blob_client(blob_name)
        with open(local_path, "rb") as f:
            client.upload_blob(f, overwrite=True)
        logger.info(f"Azure upload : {local_path} → {_AZURE_CONTAINER}/{blob_name}")
    except Exception as exc:
        logger.error(f"Erreur upload Azure ({blob_name}) : {exc}")
        raise


def _download_from_azure(blob_name: str, local_path: Path) -> None:
    """Télécharge un blob Azure vers un fichier local."""
    try:
        client = _get_blob_client(blob_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            data = client.download_blob()
            data.readinto(f)
        logger.info(f"Azure download : {_AZURE_CONTAINER}/{blob_name} → {local_path}")
    except Exception as exc:
        logger.error(f"Erreur download Azure ({blob_name}) : {exc}")
        raise


# ─── Gestion des chemins ──────────────────────────────────────────────────────

def artifact_path(filename: str, run_id: str | None = None) -> Path:
    """
    Retourne le chemin local pour un artifact.

    Si USE_AZURE_STORAGE=true, télécharge automatiquement depuis Azure Blob
    si le fichier n'existe pas localement.

    Parameters
    ----------
    filename : str
        Nom du fichier artifact (ex: "raw.parquet").
    run_id : str | None
        Identifiant du run (sous-dossier). Si None, utilise "default".

    Returns
    -------
    Path
        Chemin local vers l'artifact.
    """
    run_dir = _ARTIFACTS_BASE / (run_id or "default")
    run_dir.mkdir(parents=True, exist_ok=True)
    local = run_dir / filename
    return local


def _blob_name(filename: str, run_id: str | None) -> str:
    return f"{run_id or 'default'}/{filename}"


# ─── Parquet ─────────────────────────────────────────────────────────────────

def save_parquet(df: pd.DataFrame, filename: str, run_id: str | None = None) -> None:
    """
    Sauvegarde un DataFrame en Parquet (localement + Azure si activé).

    Parameters
    ----------
    df : pd.DataFrame
    filename : str
    run_id : str | None
    """
    local = artifact_path(filename, run_id)
    df.to_parquet(local, index=True)
    logger.info(f"Parquet sauvegardé : {local} ({len(df):,} lignes)")
    if _USE_AZURE:
        _upload_to_azure(local, _blob_name(filename, run_id))


def load_parquet(filename: str, run_id: str | None = None) -> pd.DataFrame:
    """
    Charge un DataFrame depuis un fichier Parquet.

    Parameters
    ----------
    filename : str
    run_id : str | None

    Returns
    -------
    pd.DataFrame
    """
    local = artifact_path(filename, run_id)
    if not local.exists() and _USE_AZURE:
        _download_from_azure(_blob_name(filename, run_id), local)
    if not local.exists():
        raise FileNotFoundError(f"Artifact introuvable : {local}")
    df = pd.read_parquet(local)
    logger.info(f"Parquet chargé : {local} ({len(df):,} lignes)")
    return df


# ─── Splits ──────────────────────────────────────────────────────────────────

def save_splits(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    run_id: str | None = None,
) -> None:
    """
    Sauvegarde les splits train/val/test en Parquet.

    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    feature_names : list[str]
    run_id : str | None
    """
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        df_X = pd.DataFrame(X, columns=feature_names)
        df_y = pd.Series(y, name="consommation")
        df = pd.concat([df_X, df_y], axis=1)
        save_parquet(df, f"split_{name}.parquet", run_id)

    # Sauvegarde des feature_names
    meta_path = artifact_path("feature_names.json", run_id)
    with open(meta_path, "w") as f:
        json.dump(feature_names, f)
    logger.info(f"feature_names sauvegardé : {meta_path}")


def load_splits(run_id: str | None = None) -> dict:
    """
    Charge les splits depuis les fichiers Parquet.

    Parameters
    ----------
    run_id : str | None

    Returns
    -------
    dict
        Clés : x_train, x_val, x_test, y_train, y_val, y_test, feature_names
    """
    # Chargement feature_names
    fn_path = artifact_path("feature_names.json", run_id)
    if not fn_path.exists() and _USE_AZURE:
        _download_from_azure(_blob_name("feature_names.json", run_id), fn_path)
    with open(fn_path) as f:
        feature_names = json.load(f)

    result = {"feature_names": feature_names}
    for name in ["train", "val", "test"]:
        df = load_parquet(f"split_{name}.parquet", run_id)
        result[f"x_{name}"] = df[feature_names].values
        result[f"y_{name}"] = df["consommation"].values

    return result


# ─── Meta ────────────────────────────────────────────────────────────────────

def save_meta(data: dict, run_id: str | None = None) -> None:
    """
    Sauvegarde un dictionnaire de métadonnées en JSON.

    Parameters
    ----------
    data : dict
    run_id : str | None
    """
    local = artifact_path("meta.json", run_id)
    with open(local, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Meta sauvegardé : {local}")
    if _USE_AZURE:
        _upload_to_azure(local, _blob_name("meta.json", run_id))


def load_meta(run_id: str | None = None) -> dict:
    """
    Charge les métadonnées depuis meta.json.

    Parameters
    ----------
    run_id : str | None

    Returns
    -------
    dict
    """
    local = artifact_path("meta.json", run_id)
    if not local.exists() and _USE_AZURE:
        _download_from_azure(_blob_name("meta.json", run_id), local)
    with open(local) as f:
        return json.load(f)