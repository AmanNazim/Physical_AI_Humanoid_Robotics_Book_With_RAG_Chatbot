---
title: RAG Chatbot Backend
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# RAG Chatbot Backend

This is a FastAPI-based RAG (Retrieval-Augmented Generation) backend for the Physical AI Humanoid Robotics Book project.

## Hugging Face Spaces Deployment

This backend can be deployed on Hugging Face Spaces using Docker.

## Environment Variables

Create secrets in your Space settings with the following names:

```
QDRANT_HOST=your-qdrant-host-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=book_embeddings
NEON_DATABASE_URL=your-neon-database-url
GEMINI_API_KEY=your-gemini-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
LOG_LEVEL=INFO
DEBUG=False
APP_NAME=RAG Chatbot
```

## Running Locally

```bash
# Install dependencies
uv pip install pyproject.toml

# Run the application
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Docker Deployment

Build and run with Docker:

```bash
# Build the image
docker build -t rag-chatbot .

# Run the container
docker run -p 8000:8000 -e QDRANT_HOST=... -e QDRANT_API_KEY=... rag-chatbot
```