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

COPY . /app

# Install dependencies from lockfile for reproducible builds.
RUN uv sync --frozen

EXPOSE 12138

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "12138"]
