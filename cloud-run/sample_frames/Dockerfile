FROM python:3.11-slim

WORKDIR /app

# OpenCV runtime deps + base system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080
ENV PORT=8080 PYTHONUNBUFFERED=1 LOG_LEVEL=INFO

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
