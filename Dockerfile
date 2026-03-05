FROM python:3.12-slim

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

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/

# Create workspace directory
RUN mkdir -p /tmp/workspace

# Configure git
RUN git config --global user.email "agent@langgraph.local" \
    && git config --global user.name "LangGraph Agent"

# Non-root user for security
RUN useradd -m -u 1000 agent && chown -R agent:agent /app /tmp/workspace
USER agent

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
