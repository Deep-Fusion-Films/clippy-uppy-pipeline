# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (FFmpeg + OpenCV support + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Cloud Run port (must be 8080)
EXPOSE 8080

# Start the FastAPI service
# Use Cloud Run's injected PORT env var for flexibility
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

