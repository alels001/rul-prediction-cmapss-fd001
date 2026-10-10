# syntax=docker/dockerfile:1

# ── Stage 1: build ───────────────────────────────────────────────────────────
# Install the dependencies and our package into a virtual environment.
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Dependencies first: this layer is cached and only rebuilt when the
# requirements file changes, not on every code change.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Then our own code
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# A fresh, small image that receives only the finished environment.
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RUL_MODEL_DIR=/app/models

# Run as an ordinary user, never as root
RUN useradd --create-home --uid 10001 app

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY configs/ configs/
COPY models/ models/
RUN test -f models/xgb_model.json || \
    (echo "ERROR: models/ is empty. Train a model first: make docker-train" && exit 1)

USER app
EXPOSE 8000

# Docker calls /health every 30 s; three failures in a row mark the container unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "rul.api.main:app", "--host", "0.0.0.0", "--port", "8000"]