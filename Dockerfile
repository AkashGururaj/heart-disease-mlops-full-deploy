# ----------------------------
# Base Image
# ----------------------------
FROM python:3.11-bullseye

# ----------------------------
# Set working directory
# ----------------------------
WORKDIR /app

# ----------------------------
# Install system dependencies for ML packages
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Copy and install Python dependencies
# ----------------------------
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Copy application code
# ----------------------------
COPY src/ src/

# Copy templates folder if exists
COPY src/templates/ src/templates/

# ----------------------------
# Create outputs and logs directories
# ----------------------------
RUN mkdir -p /app/outputs /app/logs

# ----------------------------
# Set environment variable
# ----------------------------
ENV PYTHONPATH=/app

# ----------------------------
# Expose port
# ----------------------------
EXPOSE 8000

# ----------------------------
# Start FastAPI app
# ----------------------------
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
