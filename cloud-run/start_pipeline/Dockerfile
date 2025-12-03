# Use a lightweight Python base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (for things like requests, SSL, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port (Cloud Run expects 8080)
EXPOSE 8080

# Environment variables (Cloud Run will override with secrets/config)
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=INFO

# Start the service (FastAPI via uvicorn, adjust if Flask or other framework)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
