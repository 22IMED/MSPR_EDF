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
RUN cat > /app/start.sh << 'EOF'
#!/bin/bash
echo "Téléchargement des modèles depuis Azure Blob..."
python -c "
import os
from azure.storage.blob import BlobServiceClient

account = os.getenv('AZURE_STORAGE_ACCOUNT', '')
key = os.getenv('AZURE_STORAGE_KEY', '')

if account and key:
    client = BlobServiceClient(account_url=f'https://{account}.blob.core.windows.net', credential=key)
    container = client.get_container_client('models')
    for blob in container.list_blobs():
        if blob.name.endswith('.joblib') or blob.name.endswith('.json'):
            with open(f'/app/models/{blob.name}', 'wb') as f:
                f.write(container.get_blob_client(blob.name).download_blob().readall())
            print(f'Téléchargé : {blob.name}')
    print('Modèles téléchargés !')
else:
    print('Azure Storage non configuré, modèles locaux utilisés.')
"
exec python run_server.py
EOF
RUN chmod +x /app/start.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["/app/start.sh"]