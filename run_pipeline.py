"""Point d'entrée pour lancer le pipeline ML complet."""
from dotenv import load_dotenv


from pipeline.steps import (
    run_extract, run_clean, run_aggregate, run_features,
    run_load_splits, run_train, run_validate,
    run_register_model, run_predictions, run_write_snowflake
)

load_dotenv()

if __name__ == "__main__":
    print("Lancement du pipeline ML complet...")
    run_id = run_extract()
    run_clean(run_id)
    run_aggregate(run_id)
    run_features(run_id)
    run_load_splits(run_id)
    run_train(run_id)
    run_validate(run_id)
    run_register_model(run_id)
    run_predictions(run_id)
    run_write_snowflake(run_id)
    print(f"Pipeline complet ! run_id={run_id}")
