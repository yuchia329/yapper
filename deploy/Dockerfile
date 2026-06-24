# One image for the API and all Celery workers (command differs per service).
# Debian's ffmpeg is built WITH libass, so subtitle burn-in works on Linux — the
# repo's vendored static ffmpeg is a macOS binary and must NOT be used here (we point
# FFMPEG_BIN/FFPROBE_BIN at the system one via env in docker-compose).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FFMPEG_BIN=/usr/bin/ffmpeg \
    FFPROBE_BIN=/usr/bin/ffprobe

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-cjk \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching, then the package with the web extra.
COPY pyproject.toml README.md ./
COPY yapper ./yapper
COPY yapper_web ./yapper_web
COPY config ./config
RUN pip install ".[web]"

# gRPC stubs are committed generated code, NOT a pip-installed package — they ride PYTHONPATH
# (same as on the GPU box). Copied after pip install so editing stubs doesn't bust the deps
# layer. Without this the web/worker import `from yapper_rpc import ...` fails at startup.
COPY yapper_rpc ./yapper_rpc
ENV PYTHONPATH=/app

EXPOSE 8000

# Default command is the API; workers override `command:` in docker-compose.
CMD ["gunicorn", "yapper_web.api:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
