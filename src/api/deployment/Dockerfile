FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IHMI_PROJECT_ROOT=/app \
    IHMI_GOLD_DIR=/app/src/api/deployment/api_data/gold

WORKDIR /app

COPY src/api/deployment/requirements-api.txt /tmp/requirements-api.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements-api.txt

COPY src /app/src

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]