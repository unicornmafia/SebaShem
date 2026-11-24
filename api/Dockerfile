# Build from the repository root (the API depends on ../shared):
#   docker build -f api/Dockerfile -t sebashem-api .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/models

WORKDIR /app
COPY shared/ /src/shared/
COPY api/pyproject.toml /src/api/pyproject.toml
COPY api/sebashem_api/ /src/api/sebashem_api/
RUN pip install /src/shared /src/api && rm -rf /src

RUN useradd --system --uid 10001 --no-create-home app
USER app

# Models are mounted at /models (see docker-compose.yml); bake them in instead with
#   COPY models/ /models/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "sebashem_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
