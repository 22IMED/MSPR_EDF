"""Lecture et écriture de données dans Snowflake."""

import logging
import os
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Connexion ────────────────────────────────────────────────────────────────
_SF_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
_SF_USER = os.getenv("SNOWFLAKE_USER", "")
_SF_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "")
_SF_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "EDF_DB")
_SF_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
_SF_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")

_TABLE_CONSUMPTION = "EDF_CONSUMPTION"
_TABLE_PREDICTIONS = "EDF_PREDICTIONS"


class SnowflakeUnavailableError(Exception):
    """Levée lorsque Snowflake est inaccessible ou non configuré."""


def _get_connection():
    """
    Retourne une connexion Snowflake.

    Raises
    ------
    SnowflakeUnavailableError
        Si les variables d'environnement sont manquantes ou la connexion échoue.
    """
    if not all([_SF_ACCOUNT, _SF_USER, _SF_PASSWORD]):
        raise SnowflakeUnavailableError(
            "Variables Snowflake manquantes : SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD."
        )
    try:
        import snowflake.connector
        conn = snowflake.connector.connect(
            account=_SF_ACCOUNT,
            user=_SF_USER,
            password=_SF_PASSWORD,
            database=_SF_DATABASE,
            schema=_SF_SCHEMA,
            warehouse=_SF_WAREHOUSE,
            role="ACCOUNTADMIN",
            authenticator="snowflake",
            login_timeout=30,
            network_timeout=60,
            insecure_mode=True,
            ocsp_fail_open=True,
            client_session_keep_alive=True,
        )
        logger.info(f"Connexion Snowflake établie : {_SF_ACCOUNT}/{_SF_DATABASE}.{_SF_SCHEMA}")
        return conn
    except ImportError:
        raise SnowflakeUnavailableError(
            "snowflake-connector-python non installé. pip install snowflake-connector-python"
        )
    except Exception as exc:
        raise SnowflakeUnavailableError(f"Connexion Snowflake échouée : {exc}") from exc


def load_from_snowflake() -> pd.DataFrame:
    conn = _get_connection()
    query = f"""
        SELECT
            DATE            AS date,
            CONSOMMATION    AS consommation,
            PREVISION_J1    AS prevision_j1
        FROM {_SF_DATABASE}.{_SF_SCHEMA}.{_TABLE_CONSUMPTION}
        WHERE DATE IS NOT NULL
          AND CONSOMMATION IS NOT NULL
        ORDER BY date ASC
        LIMIT 50000
    """
    try:
        logger.info(f"Chargement depuis Snowflake : {_TABLE_CONSUMPTION}")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cols = [desc[0].lower() for desc in cursor.description]
        df = pd.DataFrame(rows, columns=cols)
        cursor.close()
    except Exception as exc:
        raise SnowflakeUnavailableError(f"Requête Snowflake échouée : {exc}") from exc
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if df.empty:
        raise SnowflakeUnavailableError(f"La table {_TABLE_CONSUMPTION} est vide.")

    df["date"] = pd.to_datetime(df["date"])
    df["consommation"] = pd.to_numeric(df["consommation"], errors="coerce")
    df["prevision_j1"] = pd.to_numeric(df["prevision_j1"], errors="coerce")
    df = df.set_index("date").sort_index()

    logger.info(f"Snowflake : {len(df):,} lignes chargées.")
    return df


def write_predictions_to_snowflake(df: pd.DataFrame) -> int:
    """
    Écrit les prédictions dans la table EDF_PREDICTIONS via MERGE (upsert).

    Colonnes attendues dans df :
    - prediction_date (date)
    - prediction_mw (float)
    - actual_mw (float, optionnel)
    - model_name (str)
    - pipeline_run_id (str)
    - prevision_rte_j1_mw (float, optionnel)

    La colonne predicted_at est ajoutée automatiquement (UTC now).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame des prédictions à écrire.

    Returns
    -------
    int
        Nombre de lignes écrites (insertées ou mises à jour).

    Raises
    ------
    SnowflakeUnavailableError
        Si la connexion ou l'écriture échoue.
    """
    if df.empty:
        logger.warning("DataFrame vide : aucune prédiction à écrire.")
        return 0

    conn = _get_connection()

    # Ajout predicted_at
    now_utc = datetime.now(timezone.utc).isoformat()
    df = df.copy()
    df["predicted_at"] = now_utc

    # Colonnes optionnelles
    if "actual_mw" not in df.columns:
        df["actual_mw"] = None
    if "prevision_rte_j1_mw" not in df.columns:
        df["prevision_rte_j1_mw"] = None

    # Création de la table si nécessaire
    _create_predictions_table_if_not_exists(conn)

    # MERGE upsert
    rows_written = 0
    cursor = conn.cursor()
    try:
        # Staging temporaire
        cursor.execute(f"""
            CREATE TEMPORARY TABLE IF NOT EXISTS tmp_predictions (
                prediction_date     DATE,
                prediction_mw       FLOAT,
                actual_mw           FLOAT,
                model_name          VARCHAR(100),
                pipeline_run_id     VARCHAR(200),
                predicted_at        TIMESTAMP_TZ,
                prevision_rte_j1_mw FLOAT
            )
        """)
        cursor.execute("TRUNCATE TABLE tmp_predictions")

        # Insert dans staging
        for _, row in df.iterrows():
            cursor.execute(
                """
                INSERT INTO tmp_predictions VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    row.get("prediction_date"),
                    row.get("prediction_mw"),
                    row.get("actual_mw"),
                    row.get("model_name"),
                    row.get("pipeline_run_id"),
                    row.get("predicted_at"),
                    row.get("prevision_rte_j1_mw"),
                ),
            )

        # MERGE
        cursor.execute(f"""
            MERGE INTO {_TABLE_PREDICTIONS} AS tgt
            USING tmp_predictions AS src
              ON tgt.prediction_date = src.prediction_date
             AND tgt.model_name = src.model_name
            WHEN MATCHED THEN UPDATE SET
                tgt.prediction_mw       = src.prediction_mw,
                tgt.actual_mw           = src.actual_mw,
                tgt.pipeline_run_id     = src.pipeline_run_id,
                tgt.predicted_at        = src.predicted_at,
                tgt.prevision_rte_j1_mw = src.prevision_rte_j1_mw
            WHEN NOT MATCHED THEN INSERT (
                prediction_date, prediction_mw, actual_mw, model_name,
                pipeline_run_id, predicted_at, prevision_rte_j1_mw
            ) VALUES (
                src.prediction_date, src.prediction_mw, src.actual_mw,
                src.model_name, src.pipeline_run_id, src.predicted_at,
                src.prevision_rte_j1_mw
            )
        """)
        rows_written = len(df)
        conn.commit()
        logger.info(f"Snowflake : {rows_written} prédictions écrites dans {_TABLE_PREDICTIONS}.")
    except Exception as exc:
        raise SnowflakeUnavailableError(f"Écriture Snowflake échouée : {exc}") from exc
    finally:
        cursor.close()
        try:
            conn.close()
        except Exception:
            pass

    return rows_written


def _create_predictions_table_if_not_exists(conn) -> None:
    """Crée la table EDF_PREDICTIONS si elle n'existe pas."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {_TABLE_PREDICTIONS} (
                prediction_date     DATE        NOT NULL,
                prediction_mw       FLOAT,
                actual_mw           FLOAT,
                model_name          VARCHAR(100) NOT NULL,
                pipeline_run_id     VARCHAR(200),
                predicted_at        TIMESTAMP_TZ,
                prevision_rte_j1_mw FLOAT,
                PRIMARY KEY (prediction_date, model_name)
            )
        """)
        conn.commit()
    finally:
        cursor.close()