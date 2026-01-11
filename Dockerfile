FROM python:3.11-slim

LABEL maintainer="Papers QA Team <info@papersqa.com>"
LABEL description="Medical Paper Question Answering System - Production Ready"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY pyproject.toml setup.py* ./
COPY src/ ./src/
COPY data/ ./data/

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -u 1000 papersqa && \
    chown -R papersqa:papersqa /app
USER papersqa

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from papers_qa import get_settings; get_settings()" || exit 1

# Entrypoint
ENTRYPOINT ["python", "-m", "papers_qa.cli"]
CMD ["--help"]
