# Use Python 3.12 slim as base image
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libpq-dev \
    gcc \
    swig \
    libpulse-dev \
    portaudio19-dev \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy first as it's a dependency for other packages
RUN pip install --no-cache-dir numpy==1.24.0

# Install other AI/ML packages
RUN pip install --no-cache-dir \
    pandas==2.0.0 \
    scikit-learn==1.3.0 \
    sentence-transformers==2.2.2 \
    qdrant-client==1.6.0

# Install Django and other dependencies
RUN pip install --no-cache-dir -r requirements.prod.txt

# Final stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    libpulse0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy backend code
COPY backend/ /app/

# Create necessary directories
RUN mkdir -p /app/media /app/static

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run migrations and start server
CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]