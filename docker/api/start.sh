#!/bin/bash
echo "Téléchargement des modèles depuis Azure Blob..."
python -c "
import os, sys
from azure.storage.blob import BlobServiceClient

account = os.getenv('AZURE_STORAGE_ACCOUNT', '')
key = os.getenv('AZURE_STORAGE_KEY', '')

if account and key:
    try:
        client = BlobServiceClient(account_url=f'https://{account}.blob.core.windows.net', credential=key)
        container = client.get_container_client('models')
        count = 0
        for blob in container.list_blobs():
            if blob.name.endswith('.joblib') or blob.name.endswith('.json'):
                with open(f'/app/models/{blob.name}', 'wb') as f:
                    f.write(container.get_blob_client(blob.name).download_blob().readall())
                print(f'Téléchargé : {blob.name}')
                count += 1
        print(f'{count} fichiers téléchargés !')
    except Exception as e:
        print(f'Erreur Azure Blob : {e}', file=sys.stderr)
else:
    print('Azure Storage non configuré, modèles locaux utilisés.')
"
exec python run_server.py
