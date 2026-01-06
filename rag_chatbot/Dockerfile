# Multi-stage build for optimized image
# Build stage
FROM python:3.12-slim as builder

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy only requirements first for better caching
COPY pyproject.toml .

# Create a requirements.txt from pyproject.toml using uv
RUN uv pip compile pyproject.toml --output-file requirements.txt

# Install dependencies in a virtual environment
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.12-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies needed at runtime
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Set working directory
WORKDIR /app

# Copy the application code (from current directory - rag_chatbot)
COPY . .

# Make sure scripts are executable
RUN chmod +x -R /app

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port (Fly.io will set PORT environment variable)
EXPOSE 8000

# Use the virtual environment's Python and run the application
CMD ["/app/.venv/bin/uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]