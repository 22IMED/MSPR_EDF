#!/usr/bin/env bash
# =============================================================================
# setup_azure.sh — Provisionnement complet de l'infrastructure Azure pour mspr3
#
# Prérequis : az CLI installé et connecté (az login)
# Usage     : bash infra/setup_azure.sh
# =============================================================================

set -euo pipefail

# ─── Couleurs ──────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓ $1${NC}"; }
info() { echo -e "${CYAN}→ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
err()  { echo -e "${RED}✗ $1${NC}" >&2; exit 1; }

# ─── Variables ─────────────────────────────────────────────────────────────
LOCATION="${LOCATION:-francecentral}"
RG_NAME="rg-edf-mspr3"
ACR_NAME="edfmspr3acr"
STORAGE_NAME="edfmspr3storage"
PG_NAME="edf-mlflow-pg"
PG_DB="mlflowdb"
PG_ADMIN="${PG_ADMIN:-edfadmin}"
PG_PASSWORD="${PG_PASSWORD:-$(openssl rand -base64 24)}"
ACA_ENV="env-edf-mspr3"
MLFLOW_APP="edf-mlflow"
API_APP="edf-api"
PIPELINE_JOB="edf-pipeline-job"
SP_NAME="sp-edf-mspr3-github"

MLFLOW_IMAGE="ghcr.io/mlflow/mlflow:v2.13.2"
API_IMAGE="${ACR_NAME}.azurecr.io/edf-api:latest"
PIPELINE_IMAGE="${ACR_NAME}.azurecr.io/edf-pipeline:latest"

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║   SETUP AZURE — EDF MSPR3 Infrastructure    ║"
echo "  ║   Location : ${LOCATION}                    ║"
echo "  ╚══════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── 1. Resource Group ──────────────────────────────────────────────────────
info "1/9 — Création du Resource Group : ${RG_NAME}"
az group create \
    --name "${RG_NAME}" \
    --location "${LOCATION}" \
    --output none
ok "Resource Group créé : ${RG_NAME} (${LOCATION})"

# ─── 2. Azure Container Registry ────────────────────────────────────────────
info "2/9 — Création Azure Container Registry : ${ACR_NAME}"
az acr create \
    --resource-group "${RG_NAME}" \
    --name "${ACR_NAME}" \
    --sku Basic \
    --admin-enabled true \
    --location "${LOCATION}" \
    --output none
ok "ACR créé : ${ACR_NAME}.azurecr.io"

ACR_USERNAME=$(az acr credential show --name "${ACR_NAME}" --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name "${ACR_NAME}" --query "passwords[0].value" -o tsv)
ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"

# ─── 3. Storage Account ─────────────────────────────────────────────────────
info "3/9 — Création du Storage Account : ${STORAGE_NAME}"
az storage account create \
    --resource-group "${RG_NAME}" \
    --name "${STORAGE_NAME}" \
    --location "${LOCATION}" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output none

STORAGE_KEY=$(az storage account keys list \
    --resource-group "${RG_NAME}" \
    --account-name "${STORAGE_NAME}" \
    --query "[0].value" -o tsv)

STORAGE_CONN=$(az storage account show-connection-string \
    --resource-group "${RG_NAME}" \
    --name "${STORAGE_NAME}" \
    --query "connectionString" -o tsv)

# Création des containers blob
for container in mlflow-artifacts eco2mix-data models; do
    az storage container create \
        --name "${container}" \
        --account-name "${STORAGE_NAME}" \
        --account-key "${STORAGE_KEY}" \
        --output none
    ok "  Container blob créé : ${container}"
done
ok "Storage Account créé : ${STORAGE_NAME}"

# ─── 4. PostgreSQL Flexible Server ──────────────────────────────────────────
info "4/9 — Création PostgreSQL Flexible Server : ${PG_NAME}"
az postgres flexible-server create \
    --resource-group "${RG_NAME}" \
    --name "${PG_NAME}" \
    --location "${LOCATION}" \
    --admin-user "${PG_ADMIN}" \
    --admin-password "${PG_PASSWORD}" \
    --sku-name Standard_B1ms \
    --tier Burstable \
    --version 15 \
    --storage-size 32 \
    --yes \
    --output none

az postgres flexible-server db create \
    --resource-group "${RG_NAME}" \
    --server-name "${PG_NAME}" \
    --database-name "${PG_DB}" \
    --output none

# Règle firewall pour Azure services
az postgres flexible-server firewall-rule create \
    --resource-group "${RG_NAME}" \
    --name "${PG_NAME}" \
    --rule-name AllowAzureServices \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0 \
    --output none

PG_FQDN="${PG_NAME}.postgres.database.azure.com"
MLFLOW_DB_URI="postgresql+psycopg2://${PG_ADMIN}:${PG_PASSWORD}@${PG_FQDN}/${PG_DB}"
MLFLOW_ARTIFACT_ROOT="wasbs://mlflow-artifacts@${STORAGE_NAME}.blob.core.windows.net/"
ok "PostgreSQL créé : ${PG_FQDN}"

# ─── 5. Container Apps Environment ──────────────────────────────────────────
info "5/9 — Création Container Apps Environment : ${ACA_ENV}"
az containerapp env create \
    --resource-group "${RG_NAME}" \
    --name "${ACA_ENV}" \
    --location "${LOCATION}" \
    --output none
ok "Container Apps Environment créé : ${ACA_ENV}"

# ─── 6. Container App edf-mlflow (interne) ──────────────────────────────────
info "6/9 — Création Container App : ${MLFLOW_APP} (interne)"
az containerapp create \
    --resource-group "${RG_NAME}" \
    --name "${MLFLOW_APP}" \
    --environment "${ACA_ENV}" \
    --image "${MLFLOW_IMAGE}" \
    --command "mlflow" \
    --args "server" "--host" "0.0.0.0" "--port" "5000" \
      "--backend-store-uri" "${MLFLOW_DB_URI}" \
      "--default-artifact-root" "${MLFLOW_ARTIFACT_ROOT}" \
    --cpu 0.5 \
    --memory 1Gi \
    --min-replicas 1 \
    --max-replicas 1 \
    --ingress internal \
    --target-port 5000 \
    --env-vars \
      "BACKEND_STORE_URI=${MLFLOW_DB_URI}" \
      "DEFAULT_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT}" \
    --output none
ok "Container App créée : ${MLFLOW_APP} (interne)"

# ─── 7. Container App edf-api (externe) ─────────────────────────────────────
info "7/9 — Création Container App : ${API_APP} (externe)"
az containerapp create \
    --resource-group "${RG_NAME}" \
    --name "${API_APP}" \
    --environment "${ACA_ENV}" \
    --image "${API_IMAGE}" \
    --cpu 0.5 \
    --memory 1Gi \
    --min-replicas 1 \
    --max-replicas 3 \
    --ingress external \
    --target-port 8000 \
    --registry-server "${ACR_LOGIN_SERVER}" \
    --registry-username "${ACR_USERNAME}" \
    --registry-password "${ACR_PASSWORD}" \
    --env-vars \
      "MODELS_DIR=/app/models" \
      "DEFAULT_MODEL=random_forest" \
      "MLFLOW_TRACKING_URI=http://${MLFLOW_APP}" \
      "MLFLOW_EXPERIMENT=edf-consumption" \
      "DATA_SOURCE=snowflake" \
      "USE_AZURE_STORAGE=true" \
      "AZURE_STORAGE_ACCOUNT=${STORAGE_NAME}" \
    --output none

API_FQDN=$(az containerapp show \
    --name "${API_APP}" \
    --resource-group "${RG_NAME}" \
    --query "properties.configuration.ingress.fqdn" -o tsv)
ok "Container App créée : ${API_APP} → https://${API_FQDN}"

# ─── 8. Container App Job edf-pipeline-job ───────────────────────────────────
info "8/9 — Création Container App Job : ${PIPELINE_JOB}"
az containerapp job create \
    --resource-group "${RG_NAME}" \
    --name "${PIPELINE_JOB}" \
    --environment "${ACA_ENV}" \
    --image "${PIPELINE_IMAGE}" \
    --trigger-type Manual \
    --replica-timeout 3600 \
    --cpu 1.0 \
    --memory 2Gi \
    --registry-server "${ACR_LOGIN_SERVER}" \
    --registry-username "${ACR_USERNAME}" \
    --registry-password "${ACR_PASSWORD}" \
    --env-vars \
      "DATA_SOURCE=snowflake" \
      "MLFLOW_TRACKING_URI=http://${MLFLOW_APP}" \
      "MLFLOW_EXPERIMENT=edf-consumption" \
      "USE_AZURE_STORAGE=true" \
      "AZURE_STORAGE_ACCOUNT=${STORAGE_NAME}" \
      "AZURE_STORAGE_KEY=${STORAGE_KEY}" \
    --output none
ok "Container App Job créé : ${PIPELINE_JOB} (déclenchement manuel)"

# ─── 9. Service Principal GitHub Actions ─────────────────────────────────────
info "9/9 — Création Service Principal : ${SP_NAME}"
SUBSCRIPTION_ID=$(az account show --query "id" -o tsv)
RG_SCOPE="/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RG_NAME}"

SP_JSON=$(az ad sp create-for-rbac \
    --name "${SP_NAME}" \
    --role Contributor \
    --scopes "${RG_SCOPE}" \
    --sdk-auth 2>/dev/null)
ok "Service Principal créé : ${SP_NAME}"

# ─── Résumé des secrets GitHub à configurer ──────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SECRETS GITHUB À CONFIGURER (Settings → Secrets → Actions)  ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}AZURE_CREDENTIALS${NC}"
echo "${SP_JSON}"
echo ""
echo -e "${YELLOW}ACR_LOGIN_SERVER${NC}      = ${ACR_LOGIN_SERVER}"
echo -e "${YELLOW}ACR_USERNAME${NC}          = ${ACR_USERNAME}"
echo -e "${YELLOW}ACR_PASSWORD${NC}          = ${ACR_PASSWORD}"
echo ""
echo -e "${YELLOW}AZURE_STORAGE_ACCOUNT${NC} = ${STORAGE_NAME}"
echo -e "${YELLOW}AZURE_STORAGE_KEY${NC}     = ${STORAGE_KEY}"
echo ""
echo -e "${YELLOW}MLFLOW_DB_URI${NC}         = ${MLFLOW_DB_URI}"
echo -e "${YELLOW}MLFLOW_ARTIFACT_ROOT${NC}  = ${MLFLOW_ARTIFACT_ROOT}"
echo -e "${YELLOW}MLFLOW_TRACKING_URI${NC}   = http://${MLFLOW_APP}"
echo ""
echo -e "${YELLOW}SNOWFLAKE_ACCOUNT${NC}     = <votre_account>"
echo -e "${YELLOW}SNOWFLAKE_USER${NC}        = <votre_user>"
echo -e "${YELLOW}SNOWFLAKE_PASSWORD${NC}    = <votre_password>"
echo -e "${YELLOW}SNOWFLAKE_DATABASE${NC}    = EDF_DB"
echo -e "${YELLOW}SNOWFLAKE_SCHEMA${NC}      = PUBLIC"
echo -e "${YELLOW}SNOWFLAKE_WAREHOUSE${NC}   = COMPUTE_WH"
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}INFRASTRUCTURE DÉPLOYÉE AVEC SUCCÈS ✓${NC}"
echo -e "API URL : ${YELLOW}https://${API_FQDN}${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"