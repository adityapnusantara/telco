FROM python:3.11-slim

# Install system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates openssl curl \
 && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.5
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root --no-dev --extras dev

# Copy application code
COPY . .

# environment default
ENV APP_MODULE=app.main:app \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "-c", "uvicorn ${APP_MODULE} --host ${HOST} --port ${PORT}"]
