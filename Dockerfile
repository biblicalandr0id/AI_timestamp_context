# AI Chatbot - Docker Container
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for persistent data
RUN mkdir -p /app/data /app/checkpoints /app/exports

# Expose port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=chatbot_server.py
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health')"

# Run the chatbot server
CMD ["python", "launch_chatbot.py", "server", "--host", "0.0.0.0", "--port", "5000"]
