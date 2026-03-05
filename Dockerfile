FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Ensure SSL certificates are readable by the non-root agent user.
# The base image's /etc/ssl/certs directory has mode 700 (root-only), so we
# copy the CA bundle directly to a world-readable path and point the SSL env
# vars there. This prevents [Errno 2] / [Errno 13] errors on HTTPS/LLM calls.
RUN mkdir -p /usr/local/share/ca-certificates \
    && cp /etc/ssl/certs/ca-certificates.crt /usr/local/share/ca-certificates/ca-certificates.crt \
    && chmod 644 /usr/local/share/ca-certificates/ca-certificates.crt

# Copy source code
COPY src/ ./src/

# Create workspace directory
RUN mkdir -p /tmp/workspace

# Non-root user for security
RUN useradd -m -u 1000 agent && chown -R agent:agent /app /tmp/workspace
USER agent

# Configure git for the agent user (required for commits at runtime and in CI)
RUN git config --global user.email "agent@langgraph.local" \
    && git config --global user.name "LangGraph Agent"

# Point SSL to our readable cert bundle so non-root HTTPS calls work
ENV SSL_CERT_FILE=/usr/local/share/ca-certificates/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/ca-certificates.crt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
