FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Run the application using uvicorn to properly serve the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--reload", "false"]