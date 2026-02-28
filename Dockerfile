FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime libs commonly required by OCR/vision deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Copy full project so hatchling has README.md and app/ when building docling-server.
COPY . /app
RUN uv sync --frozen --extra cuda

# Verify torch and AutoProcessor (fail build with clear error if not)
RUN .venv/bin/python -c "import torch; print('torch:', torch.__version__)"
RUN .venv/bin/python scripts/check_autoprocessor.py

EXPOSE 12138
CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "12138", "--timeout-keep-alive", "480"]
