# Hugging Face Deployment Guide for FastAPI RAG Backend

This guide provides step-by-step instructions for deploying your FastAPI RAG backend to Hugging Face Spaces.

## About Hugging Face Spaces

Hugging Face Spaces provides free GPU-powered containers for deploying machine learning applications. You get:

- Free tier with 300 GPU hours per month
- Easy GitHub integration
- Multiple SDK options (Docker, Gradio, Streamlit, etc.)
- Built-in sharing and collaboration features

## Prerequisites

1. **Create a Hugging Face Account**: Go to [huggingface.co](https://huggingface.co) and sign up
2. **Create a Space**: From your profile, create a new Space with Docker SDK
3. **Connect your GitHub repository** or push your code directly

## Deployment Methods

### Method 1: GitHub Integration (Recommended)

1. Push this repository to GitHub
2. Connect your GitHub repo to a Hugging Face Space
3. Hugging Face will automatically build and deploy using the Dockerfile

### Method 2: Direct Push to Hugging Face

1. Create a repository on Hugging Face
2. Clone it locally
3. Copy your files
4. Push to Hugging Face

## Required Files for Hugging Face Spaces

Your repository should contain:

- `Dockerfile` - Multi-stage Docker build (already provided)
- `space.yaml` - Hugging Face Spaces configuration (already provided)
- `pyproject.toml` - Project dependencies
- `backend/main.py` - FastAPI application
- `.env.example` - Example environment variables file
- `README.md` - Updated for Hugging Face deployment

## Environment Variables Setup

### Create Secrets in Hugging Face Spaces:

In your Space settings, go to the "Secrets" tab and add:

#### Required Secrets:

- `QDRANT_HOST` - Your Qdrant Cloud host URL
- `QDRANT_API_KEY` - Your Qdrant Cloud API key
- `QDRANT_COLLECTION_NAME` - Qdrant collection name (e.g., "book_embeddings")
- `NEON_DATABASE_URL` - Your Neon PostgreSQL connection string
- `GEMINI_API_KEY` - Google Gemini API key for embeddings
- `OPENROUTER_API_KEY` - OpenRouter API key for LLM

#### Optional Secrets:

- `LOG_LEVEL` - Logging level (default: "INFO")
- `DEBUG` - Debug mode (default: "False")
- `APP_NAME` - Application name (default: "RAG Chatbot")

## Dockerfile Configuration for Hugging Face

The provided Dockerfile is already optimized for Hugging Face Spaces:

- Multi-stage build for smaller image size
- Uses uv for fast dependency installation
- Non-root user for security
- Properly handles PORT environment variable
- Exposes port 8000

## space.yaml Configuration

The space.yaml file configures the Space runtime:

- Uses Docker runtime
- Allocates CPU resources (can be upgraded to GPU if needed)
- Sets memory allocation

## Deployment Steps

### 1. Prepare Your Repository

Make sure your repository contains all necessary files:

```bash
Dockerfile
space.yaml
pyproject.toml
backend/
shared/
agents_sdk/
embedding_pipeline/
databases/
```

### 2. Create a Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - SDK: Docker
   - Hardware: CPU (or GPU if needed)
   - Visibility: Public or Private

### 3. Connect GitHub Repository

1. In your Space settings, go to "Files and repositories"
2. Connect your GitHub repository
3. Enable "Pull from repo" to automatically sync changes

### 4. Add Environment Variables

1. Go to "Settings" → "Secrets"
2. Add all required environment variables as secrets

### 5. Monitor Deployment

- Check the "Logs" tab to monitor build progress
- Once built, your application will be available at: `https://your-username-space-name.hf.space`

## API Endpoints

Once deployed, your API endpoints will be available at:

- Health: `https://your-username-space-name.hf.space/api/v1/health`
- Config: `https://your-username-space-name.hf.space/api/v1/config`
- Chat: `https://your-username-space-name.hf.space/api/v1/chat`
- Retrieve: `https://your-username-space-name.hf.space/api/v1/retrieve`
- Embed: `https://your-username-space-name.hf.space/api/v1/embed`

## Hugging Face Spaces Specific Notes

- Hugging Face Spaces provides the `$PORT` environment variable automatically
- The application should bind to `0.0.0.0` and use the provided port
- Spaces support both CPU and GPU hardware
- Free tier provides 300 GPU hours per month
- Applications sleep after 48 hours of inactivity (on free tier)
- Can be woken up by visiting the URL

## Troubleshooting

### Application Not Starting

- Check that the PORT environment variable is being used
- Verify that the application binds to 0.0.0.0, not localhost
- Review logs in the Spaces interface

### Build Failures

- Ensure all dependencies in pyproject.toml are available
- Check that the Dockerfile builds successfully locally
- Verify that the image size doesn't exceed limits

### Environment Variables Not Working

- Make sure secrets are added in the Spaces settings, not as regular environment variables
- Verify secret names match exactly what the application expects

## Scaling and Performance

- Upgrade hardware in Space settings for better performance
- Monitor resources usage in the Spaces dashboard
- Consider caching strategies for better response times

Your FastAPI RAG backend is now ready for Hugging Face Spaces deployment!
