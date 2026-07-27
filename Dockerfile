FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/root/.local/bin:/app/.venv/bin:$PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY backend ./backend

EXPOSE 8000

CMD ["sh", "-lc", "uv run uvicorn backend.app.main:app --host 0.0.0.0 --port ${TOOX_PORT:-8011}"]
