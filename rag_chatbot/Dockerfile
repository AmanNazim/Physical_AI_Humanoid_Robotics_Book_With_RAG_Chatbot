FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python -m venv .venv

# Make sure we use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies in the virtual environment using uv
RUN uv pip install --no-cache -r requirements.txt

# Copy the application code
COPY . .

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Run the application using uvicorn to properly serve the FastAPI app
# Don't use --reload in production (Hugging Face Spaces)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]