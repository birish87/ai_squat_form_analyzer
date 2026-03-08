# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed by opencv-python-headless and mediapipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a prefix so we can copy cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy only the runtime system libs we need
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy backend source
COPY backend/ ./backend/

# Copy frontend static files — served by FastAPI via StaticFiles
COPY frontend/ ./frontend/

# Railway injects $PORT at runtime. Default to 8000 for local docker run.
ENV PORT=8000

EXPOSE $PORT

# Run from the backend directory so sibling imports resolve correctly
WORKDIR /app/backend

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}