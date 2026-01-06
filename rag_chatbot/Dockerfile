# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Change to rag_chatbot directory to install dependencies
WORKDIR /app/rag_chatbot

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Change back to main app directory
WORKDIR /app

# Expose the port
EXPOSE $PORT

# Command to run the application from rag_chatbot directory
CMD ["sh", "-c", "cd rag_chatbot && uvicorn backend.main:app --host=0.0.0.0 --port=$PORT"]