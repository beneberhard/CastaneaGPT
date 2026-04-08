# ---------------------------------------------------
# Base image
# ---------------------------------------------------
FROM python:3.11-slim-bookworm

# ---------------------------------------------------
# Working directory
# ---------------------------------------------------
WORKDIR /app

# ---------------------------------------------------
# System dependencies (GDAL + build tools)
# ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    gdal-bin \
    libgdal-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# Python environment settings
# ---------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------
COPY requirements.docker.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---------------------------------------------------
# Copy application code
# ---------------------------------------------------
COPY . /app

# ---------------------------------------------------
# Expose FastAPI port
# ---------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------
# Start API
# ---------------------------------------------------
CMD ["uvicorn", "app.api.api:app", "--host", "0.0.0.0", "--port", "8000"]