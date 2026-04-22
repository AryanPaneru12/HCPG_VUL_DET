FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY deployment/requirements.cloud.txt requirements.cloud.txt
RUN pip install --no-cache-dir -r requirements.cloud.txt

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY models/best_hgt_model.pt models/best_hgt_model.pt
COPY models/model_metrics.json models/model_metrics.json

# Expose port
EXPOSE 8080

# Run the application
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
