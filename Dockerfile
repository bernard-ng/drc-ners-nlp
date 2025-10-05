# syntax=docker/dockerfile:1

# Minimal Linux base (glibc) – Python will be installed by uv
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_INSTALL_DIR=/usr/local/bin \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /app

# System deps for building/using common scientific stack
# Keep minimal; rely on wheels where possible
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      build-essential pkg-config \
      libssl-dev libffi-dev \
      libopenblas0 libstdc++6 \
      libfreetype6 libpng16-16 libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

# Install uv (static binary)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy project metadata first for layer caching
COPY pyproject.toml README.md ./

# Install a managed Python via uv and create the project venv
RUN uv python install 3.11 \
    && uv venv /app/.venv --python 3.11

# Resolve and install runtime deps into project venv
# Use lockfile if present for reproducibility
RUN if [ -f uv.lock ]; then uv sync --no-dev --no-install-project --frozen; else uv sync --no-dev --no-install-project; fi

# Copy source code and optional templates
COPY src ./src

# Re-sync to ensure the local package is installed
RUN uv sync --no-dev \
    && rm -rf /root/.cache

# Default command shows help; override in compose or docker run
CMD ["ners", "--help"]
