# Dockerfile for Multi-Camera Tracking API & Live Video Streamer

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=8000

WORKDIR /app

# Install system dependencies for OpenCV, FFmpeg & video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and dependencies specification
COPY pyproject.toml /app/

# Install Python packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    opencv-python-headless \
    numpy \
    prometheus_client \
    pydantic \
    pyyaml \
    sortedcontainers \
    sqlalchemy

# Copy project source code
COPY src /app/src

# Create temporary upload/storage directories
RUN mkdir -p /tmp/mctracker_uploads

# Expose port (Railway / Render override via $PORT env var)
EXPOSE ${PORT:-8000}

# Start FastAPI — binds to $PORT so Railway/Render can route traffic correctly
CMD ["sh", "-c", "exec uvicorn mctracker.api.server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
