# Papers QA - Production Docker Image
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install optional API dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard]

# Production stage
FROM python:3.11-slim as production

LABEL maintainer="Papers QA Team <info@papersqa.com>"
LABEL description="Medical Paper Question Answering System"
LABEL version="1.0.0"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY pyproject.toml ./
COPY src/ ./src/

# Create data directories
RUN mkdir -p /app/data/raw /app/data/generated /app/data/cache /app/logs

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash papersqa && \
    chown -R papersqa:papersqa /app

USER papersqa

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || python -c "from papers_qa import get_settings; get_settings()" || exit 1

# Default command - show help
ENTRYPOINT ["python", "-m", "papers_qa.cli"]
CMD ["--help"]

# Development stage
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    ipython \
    jupyter

USER papersqa

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]
CMD []
