FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl bash \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code pipeline
COPY pipeline/ ./pipeline/
COPY preprocessing/ ./preprocessing/
COPY models.py ./

# Dossiers de travail
RUN mkdir -p /app/data /app/models /app/artifacts

# Script d'entrée
COPY docker/pipeline/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Utilisateur non-root
RUN useradd -m -u 1000 pipelineuser \
    && chown -R pipelineuser:pipelineuser /app /entrypoint.sh
USER pipelineuser

ENTRYPOINT ["/entrypoint.sh"]