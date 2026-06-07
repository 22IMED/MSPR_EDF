FROM python:3.11-slim
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY preprocessing/ ./preprocessing/
COPY pipeline/ ./pipeline/
COPY tests/ ./tests/

RUN mkdir -p /app/models /app/artifacts /app/data

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "pipeline.main"]