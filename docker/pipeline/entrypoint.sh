#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — Orchestration du pipeline ML EDF eCO2mix
# Exécute les 10 étapes dans l'ordre avec logs et gestion d'erreurs.
# =============================================================================

set -euo pipefail

# ─── Couleurs et helpers ─────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_step() { echo -e "\n${CYAN}════════════════════════════════════════${NC}"; echo -e "${CYAN}  $1${NC}"; echo -e "${CYAN}════════════════════════════════════════${NC}"; }
log_ok()   { echo -e "${GREEN}✓ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
log_err()  { echo -e "${RED}✗ $1${NC}" >&2; }

PIPELINE_START=$(date +%s)
RUN_ID=${RUN_ID:-$(python -c "import uuid; print(str(uuid.uuid4())[:8])")}
export RUN_ID

echo -e "${CYAN}"
echo "  ██████ ██████  ███████"
echo "  ██     ██   ██ ██     "
echo "  █████  ██   ██ █████  "
echo "  ██     ██   ██ ██     "
echo "  ██████ ██████  ██     "
echo ""
echo "  Pipeline ML — Consommation Électrique FR"
echo "  Run ID : ${RUN_ID}"
echo "  Date   : $(date '+%Y-%m-%d %H:%M:%S UTC')"
echo -e "${NC}"

# ─── Téléchargement données Azure Blob (si DATA_SOURCE=xls et Azure configuré) ─
if [[ "${DATA_SOURCE:-xls}" == "xls" && -n "${AZURE_STORAGE_ACCOUNT:-}" ]]; then
    log_step "PRÉ-ÉTAPE : Téléchargement données Azure Blob"
    CONTAINER="${AZURE_BLOB_CONTAINER_DATA:-eco2mix-data}"
    DATA_DIR="${DATA_DIR:-/app/data}"
    mkdir -p "${DATA_DIR}"

    if command -v az &>/dev/null; then
        log_warn "Téléchargement fichiers .xls depuis Azure Blob (${CONTAINER})..."
        az storage blob download-batch \
            --account-name "${AZURE_STORAGE_ACCOUNT}" \
            --account-key "${AZURE_STORAGE_KEY:-}" \
            --source "${CONTAINER}" \
            --destination "${DATA_DIR}" \
            --pattern "*.xls" \
            2>/dev/null || log_warn "Aucun fichier .xls trouvé dans Azure Blob."
        log_ok "Téléchargement terminé."
    else
        log_warn "Azure CLI non disponible, téléchargement ignoré."
    fi
elif [[ "${DATA_SOURCE:-xls}" == "snowflake" ]]; then
    log_warn "DATA_SOURCE=snowflake — pas de téléchargement Azure Blob nécessaire."
fi

# ─── Fonction d'exécution d'un step ──────────────────────────────────────────
run_step() {
    local step_num="$1"
    local step_name="$2"
    local python_cmd="$3"

    log_step "STEP ${step_num}/10 : ${step_name}"
    STEP_START=$(date +%s)

    if python -c "${python_cmd}" 2>&1; then
        STEP_END=$(date +%s)
        log_ok "${step_name} terminé en $((STEP_END - STEP_START))s."
    else
        log_err "${step_name} ÉCHOUÉ."
        exit 1
    fi
}

# ─── Exécution des 10 steps ──────────────────────────────────────────────────

run_step 1 "EXTRACTION" \
    "from pipeline.steps import run_extract; run_extract('${RUN_ID}')"

run_step 2 "NETTOYAGE" \
    "from pipeline.steps import run_clean; run_clean('${RUN_ID}')"

run_step 3 "AGRÉGATION JOURNALIÈRE" \
    "from pipeline.steps import run_aggregate; run_aggregate('${RUN_ID}')"

run_step 4 "FEATURE ENGINEERING" \
    "from pipeline.steps import run_features; run_features('${RUN_ID}')"

run_step 5 "SPLIT TRAIN/VAL/TEST" \
    "from pipeline.steps import run_load_splits; run_load_splits('${RUN_ID}')"

run_step 6 "ENTRAÎNEMENT MODÈLES" \
    "from pipeline.steps import run_train; run_train('${RUN_ID}')"

run_step 7 "VALIDATION" \
    "from pipeline.steps import run_validate; run_validate('${RUN_ID}')"

run_step 8 "ENREGISTREMENT MLFLOW" \
    "from pipeline.steps import run_register_model; run_register_model('${RUN_ID}')"

run_step 9 "GÉNÉRATION PRÉDICTIONS" \
    "from pipeline.steps import run_predictions; run_predictions('${RUN_ID}')"

run_step 10 "ÉCRITURE SNOWFLAKE" \
    "from pipeline.steps import run_write_snowflake; run_write_snowflake('${RUN_ID}')"

# ─── Upload modèles vers Azure Blob ─────────────────────────────────────────
if [[ -n "${AZURE_STORAGE_ACCOUNT:-}" ]] && command -v az &>/dev/null; then
    log_step "POST-ÉTAPE : Upload modèles vers Azure Blob"
    MODELS_CONTAINER="${AZURE_BLOB_CONTAINER_MODELS:-models}"
    MODELS_DIR="${MODELS_DIR:-/app/models}"

    for f in "${MODELS_DIR}"/*.joblib "${MODELS_DIR}"/*.metrics.json; do
        [[ -f "$f" ]] || continue
        BLOB_NAME="$(basename "$f")"
        az storage blob upload \
            --account-name "${AZURE_STORAGE_ACCOUNT}" \
            --account-key "${AZURE_STORAGE_KEY:-}" \
            --container-name "${MODELS_CONTAINER}" \
            --name "${BLOB_NAME}" \
            --file "$f" \
            --overwrite 2>/dev/null && log_ok "Uploadé : ${BLOB_NAME}" \
            || log_warn "Échec upload : ${BLOB_NAME}"
    done
fi

# ─── Résumé final ─────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
TOTAL_DURATION=$((PIPELINE_END - PIPELINE_START))

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  PIPELINE TERMINÉ AVEC SUCCÈS ✓${NC}"
echo -e "${GREEN}  Run ID   : ${RUN_ID}${NC}"
echo -e "${GREEN}  Durée    : ${TOTAL_DURATION}s${NC}"
echo -e "${GREEN}  Fin      : $(date '+%Y-%m-%d %H:%M:%S UTC')${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"