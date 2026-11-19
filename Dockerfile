# ============================================================
# Bengaluru Rider Skin Health & Sun Exposure Advisor
# Docker Image — optimized for IBM Cloud Code Engine
# ============================================================

FROM python:3.11-slim

# --- OS dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libssl-dev libffi-dev curl \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Application source ---
COPY backend/     ./backend/
COPY frontend/    ./frontend/
COPY database/    ./database/
COPY .env.example .env.example

# --- Environment defaults (override at runtime) ---
ENV APP_ENV=production \
    APP_PORT=8000 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# --- Health check for Code Engine ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# --- Entry point ---
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
