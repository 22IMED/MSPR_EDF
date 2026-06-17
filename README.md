# EDF - Prédiction de la consommation électrique nationale

Ce projet a été réalisé dans le cadre de la MSPR Bloc 3 (Préparer la maintenabilité et le déploiement de la solution IA, RNCP 36582, Chef.fe de projet Expert en Intelligence Artificielle, Niveau 7, EPSI).

L'objectif est de prédire la consommation électrique nationale française à partir des données RTE éco2mix. Le projet couvre l'entraînement et la sélection automatique de modèles, l'exposition via une API REST, un pipeline de ré-entraînement programmé, le suivi de la qualité du modèle (MLflow, Prometheus, Grafana) et le déploiement sur Azure Container Apps.

## Architecture

Le flux principal du projet, exécuté chaque semaine, est le suivant :

1. Les données (RTE éco2mix) sont récupérées depuis Snowflake.
2. Le pipeline ML (extraction, nettoyage, feature engineering, entraînement) s'exécute.
3. MLflow trace les expériences et enregistre le modèle dans le Model Registry.
4. Le modèle validé est exporté vers Azure Blob Storage.
5. L'API FastAPI (edf-api) charge ce modèle et sert les prédictions.
6. Le dashboard Streamlit (edf-streamlit) affiche les résultats aux utilisateurs.
7. Prometheus et Grafana assurent le monitoring et les alertes.

Le pipeline de ré-entraînement (edf-pipeline-job) est déclenché automatiquement chaque lundi à 3h UTC via GitHub Actions (workflow ml_pipeline.yml).

Le détail de l'architecture (environnements dev/test/prod, processus de maintenabilité, RACI, sécurité DIC/RGPD) est documenté dans :
- docs/architecture.md
- docs/conception_solution_edf.md
- les livrables 1 à 3 dans le dossier docs/

## Modèles de Machine Learning

Quatre modèles sont entraînés et comparés à chaque cycle de ré-entraînement :

- Decision Tree (sklearn.tree.DecisionTreeRegressor)
- Random Forest, modèle retenu en production (sklearn.ensemble.RandomForestRegressor)
- KNN (sklearn.neighbors.KNeighborsRegressor)
- MLP (sklearn.neural_network.MLPRegressor)

Le meilleur modèle est sélectionné automatiquement par la fonction select_best_model, sous condition d'un R² supérieur ou égal à 0,85. Il est ensuite comparé au modèle actuellement en production par save_model_if_better, qui ne le remplace que s'il est meilleur.

### Features utilisées

La liste complète des features est définie dans preprocessing/constants.py (FEATURE_COLS) :

- prevision_j1 : prévision RTE de consommation à J-1 (MW)
- lag_1 : consommation réelle à J-1 (MW)
- lag_7 : consommation réelle à J-7 (MW)
- prevision_j1_lag1 : prévision J-1 décalée d'un jour
- day_of_week, month, day_of_year : variables calendaires
- is_weekend, is_holiday : indicateurs week-end et jour férié
- saison : saison de l'année, déduite du mois
- month_sin, month_cos, dow_sin, dow_cos : encodage cyclique du mois et du jour de la semaine

Les variables de production énergétique (nucléaire, éolien, solaire, hydraulique, gaz, fioul, taux de CO2) ne sont volontairement pas utilisées dans les features du modèle, car elles sont trop fortement corrélées à la consommation totale (risque de data leakage). Ces champs restent acceptés par l'API et le dashboard pour une utilisation future, mais ne sont pas pris en compte par le modèle actuel.

### Métriques et seuils

| Métrique | Seuil | Valeur en production (Random Forest) |
|---|---|---|
| R2 | >= 0,85 | 0,943 |
| MAPE | <= 5 % | 4 % |
| RMSE | suivi d'évolution | 941 MW |

## Structure du projet

```
.
├── main.py                  API FastAPI
├── run_server.py            lance le serveur uvicorn
├── models.py                entraînement et sélection des 4 modèles
├── streamlit_app.py          dashboard utilisateur
│
├── pipeline/                  pipeline de ré-entraînement
│   ├── main.py                orchestration du pipeline
│   ├── steps.py                étapes : extract, clean, train, validate, register...
│   ├── snowflake_io.py         lecture et écriture Snowflake
│   └── artifacts.py
│
├── preprocessing/              extraction, nettoyage, feature engineering
│   ├── extract.py
│   ├── transform.py
│   ├── load.py
│   ├── pipline.py
│   └── constants.py            FEATURE_COLS, seuils qualité, jours fériés
│
├── tests/
│   ├── test_api.py             tests des endpoints (avec DummyModel)
│   ├── test_preprocessing.py   tests du pipeline de données
│   ├── test_pipline.py         test de structure du projet
│   ├── integration/             tests d'intégration Snowflake
│   └── e2e/                    tests end to end de l'API
│
├── docker/
│   ├── api/                    Dockerfile et entrypoint de l'API
│   ├── pipeline/                Dockerfile du pipeline ML
│   └── streamlit/               Dockerfile du dashboard
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/                  dashboards et provisioning Grafana
│
├── infra/
│   └── setup_azure.sh           script de provisioning Azure
│
├── docs/                        documentation et livrables MSPR
│
└── .github/workflows/
    ├── ci.yml                   lint, tests, build des images Docker
    ├── cd.yml                   déploiement Azure Container Apps
    └── ml_pipeline.yml          ré-entraînement hebdomadaire programmé
```

## Installation et démarrage local

Prérequis : Python 3.11 et un environnement virtuel.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copier le fichier .env.example vers .env et ajuster les variables si besoin :

```
cp .env.example .env
```

Pour développer en local sans dépendre de Snowflake, laisser DATA_SOURCE=local. Les données sont alors lues depuis le dossier ./data.

Lancer l'API :

```
python run_server.py
```

L'API est accessible sur http://localhost:8000. La documentation interactive (Swagger) est sur http://localhost:8000/docs.

Lancer le dashboard :

```
streamlit run streamlit_app.py
```

## API

Endpoints principaux :

| Méthode | Endpoint | Description |
|---|---|---|
| GET | /health | statut de santé de l'API et du modèle chargé |
| GET | /models | liste des modèles disponibles |
| POST | /predict | prédiction pour une date donnée |
| POST | /forecast | prévision sur une période |
| GET | /metrics | métriques au format Prometheus |

API de production : https://edf-api.orangebeach-b5cf8765.francecentral.azurecontainerapps.io

## Pipeline ML et MLflow

Le pipeline (pipeline/main.py) exécute, dans l'ordre : extraction, nettoyage, agrégation, feature engineering, split train/val/test (70/15/15, chronologique), entraînement des 4 modèles, validation (R2 >= 0,85), enregistrement dans le Model Registry MLflow (fonction run_register_model, nom de registre edf-random-forest), puis export du modèle validé vers Azure Blob Storage.

Le suivi des expériences et du registre est disponible via l'interface MLflow déployée (edf-mlflow sur Azure Container Apps).

## Tests

```
pytest -v
```

Les tests sont lancés automatiquement en CI (ci.yml) à chaque push ou pull request sur main, avec DATA_SOURCE=xls (pas de dépendance à Snowflake).

- tests/test_api.py : endpoints, validation Pydantic, métriques
- tests/test_preprocessing.py : extraction, nettoyage, feature engineering, split
- tests/test_pipline.py : structure du projet
- tests/integration/ : intégration Snowflake
- tests/e2e/ : tests end to end de l'API

## CI/CD

- ci.yml : lint (Ruff), format (Ruff), tests (Pytest), puis build et push des images Docker vers Azure Container Registry si tout passe et que la branche est main.
- cd.yml : déploiement sur Azure Container Apps (pipeline, MLflow, API, healthcheck, Streamlit).
- ml_pipeline.yml : ré-entraînement hebdomadaire programmé (lundi 3h UTC).

## Monitoring

Métriques exposées via /metrics au format Prometheus :

- edf_requests_total : nombre de prédictions par modèle
- edf_errors_total : nombre d'erreurs
- edf_model_r2, edf_model_mape_percent : qualité du modèle actif
- edf_latency_ms_avg : latence moyenne

Les dashboards Grafana sont provisionnés dans monitoring/grafana/.

## Documentation du projet

- Livrable 1 : dossier déploiement et maintenabilité (architecture, processus de maintenabilité, tests)
- Livrable 2 : documentation technique, runbook d'exploitation, note d'expertise technique
- Livrable 3 : plan d'accompagnement au changement (ADKAR, kit de bonne utilisation, fiche A3)

## Équipe

Projet réalisé par Arthur Méry, Martin Dié et Imed Eddine Zeroual, EPSI, Pro Alterna, MSPR Bloc 3 (RNCP 36582), 2026.
