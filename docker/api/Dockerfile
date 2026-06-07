FROM python:3.11-slim
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir azure-storage-blob

COPY main.py run_server.py models.py ./
COPY preprocessing/ ./preprocessing/
COPY pipeline/ ./pipeline/

RUN mkdir -p /app/models

# Script de démarrage qui télécharge les modèles depuis Azure Blob
COPY docker/api/start.sh /app/start.sh
RUN chmod +x /app/start.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["/app/start.sh"]