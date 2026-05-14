FROM python:3.11-slim

WORKDIR /app

# Install uv via pip (avoids pulling from ghcr.io)
RUN pip install uv --quiet

# Copy dependency files first (layer cache — only reinstalls on pyproject change)
COPY pyproject.toml uv.lock README.md ./

# Install runtime deps only — skip local package build
RUN uv sync --no-dev --frozen --no-install-project

# Copy application code used by both training jobs and the API
COPY main.py ./
COPY src/ ./src/
COPY configs/ ./configs/

# Make the project importable without installing the package
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH=/app/src

# Artifact paths — mounted at runtime via docker-compose volumes
ENV MODELS_DIR=/app/models
ENV FEATURES_DIR=/app/data/features
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

RUN mkdir -p /app/data /app/models /app/reports /app/mlartifacts /app/logs

EXPOSE 8000

# Use venv's uvicorn directly — avoids uv re-triggering project build at startup
CMD ["uvicorn", "serving.predict:app", "--host", "0.0.0.0", "--port", "8000"]
