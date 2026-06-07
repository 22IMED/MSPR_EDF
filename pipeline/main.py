import os
import sys
import uuid
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    run_id = os.getenv("RUN_ID", str(uuid.uuid4())[:8])
    logger.info(f"=== PIPELINE START — run_id={run_id} ===")

    from pipeline.steps import (
        run_extract,
        run_clean,
        run_aggregate,
        run_features,
        run_load_splits,
        run_train,
        run_validate,
        run_register_model,
        run_predictions,
        run_write_snowflake,
    )

    run_extract(run_id)
    run_clean(run_id)
    run_aggregate(run_id)
    run_features(run_id)
    run_load_splits(run_id)
    run_train(run_id)

    try:
        run_validate(run_id)
    except RuntimeError as e:
        logger.error(f"Validation échouée : {e}")
        sys.exit(1)

    run_register_model(run_id)
    run_predictions(run_id)
    run_write_snowflake(run_id)

    # Upload modèles vers Azure Blob
    _upload_models()

    logger.info(f"=== PIPELINE END — run_id={run_id} ===")


def _upload_models():
    account = os.getenv("AZURE_STORAGE_ACCOUNT")
    key = os.getenv("AZURE_STORAGE_KEY")
    if not account or not key:
        logger.warning("AZURE_STORAGE_ACCOUNT/KEY non définis — upload ignoré.")
        return

    try:
        from azure.storage.blob import BlobServiceClient

        client = BlobServiceClient(
            account_url=f"https://{account}.blob.core.windows.net",
            credential=key,
        )
        container = client.get_container_client("models")

        models_dir = Path(os.getenv("MODELS_DIR", "models"))
        uploaded = 0
        for f in list(models_dir.glob("*.joblib")) + list(
            models_dir.glob("*.metrics.json")
        ):
            with open(f, "rb") as data:
                container.upload_blob(f.name, data, overwrite=True)
            logger.info(f"Uploadé : {f.name}")
            uploaded += 1

        logger.info(f"Upload terminé : {uploaded} fichiers.")
    except Exception as e:
        logger.error(f"Upload Azure Blob échoué : {e}")


if __name__ == "__main__":
    main()
