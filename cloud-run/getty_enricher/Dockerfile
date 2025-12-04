# Use official Python base image (upgrade to 3.11 for consistency with other services)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies:
# - curl for API calls
# - build-essential/pkg-config for compiling Python deps
# - libssl/libffi for requests/cryptography
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Cloud Run port (must be 8080)
EXPOSE 8080

# Environment variables (Cloud Run will override with secrets/config)
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Start the enrichment service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
