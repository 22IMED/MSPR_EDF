FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code applicatif
COPY main.py run_server.py models.py ./
COPY preprocessing/ ./preprocessing/
COPY pipeline/ ./pipeline/

# Dossier modèles (monté via volume en prod)
RUN mkdir -p /app/models

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "run_server.py"]