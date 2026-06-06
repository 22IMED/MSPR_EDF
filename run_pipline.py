from dotenv import load_dotenv

from pathlib import Path
from preprocessing.pipline import run_full_pipeline
from models import train_all, evaluate_all, select_best_model
import mlflow
import os
import joblib

load_dotenv()

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(exist_ok=True)

# 1. ETL
xls_files = list(Path("data").glob("*.xls")) + list(Path("data").glob("*.csv"))
print(f"Fichiers trouves : {[f.name for f in xls_files]}")

X_train, X_val, X_test, y_train, y_val, y_test, feature_names, df_features = (
    run_full_pipeline(xls_files)
)
print(f"Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# 2. Entrainement
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "edf-consumption"))

trained = train_all(X_train, y_train)
print(f"Modeles entraines : {list(trained.keys())}")

# 3. Evaluation
results = evaluate_all(trained, X_val, y_val, X_test, y_test)
best = select_best_model(results)
print(f"Meilleur modele : {best}")
print(results)

# 4. Sauvegarde
for name, info in trained.items():
    joblib.dump(info["pipeline"], MODELS_DIR / f"{name}.joblib")
    print(f"Modele sauvegarde : {name}.joblib")

print("Pipeline complet !")
