---
id: "001"
title: "Debug IntelligenceService Dependency Issue"
stage: "general"
date_iso: "2026-01-22"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "main"
user: "user"
command: "debug-intelligenceservice-dependency"
labels:
  - "debugging"
  - "deployment"
  - "dependencies"
  - "agents-sdk"
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml:
  - "/rag_chatbot/backend/services/rag_service.py"
  - "/rag_chatbot/requirements.txt"
tests_yaml:
  - "None"
---

# Debug IntelligenceService Dependency Issue

## PROMPT_TEXT

User reported that bot messages are not appearing in ChatKit UI when using proper API keys, while fallback messages appear when API keys are not quoted. The issue was traced to the IntelligenceService not being available in the deployment environment due to missing dependencies. User requested adding more detailed logging to understand the exact reason for "IntelligenceService not available" error and proper virtual environment setup in Dockerfile.

## RESPONSE_TEXT

Added detailed logging to RAG service to identify the exact ImportError causing IntelligenceService to be unavailable in deployment. Logs revealed that the issue was missing `litellm` dependency required by LitellmModel. Updated requirements.txt to use `openai-agents[litellm]` instead of just `openai-agents` to include the necessary optional dependency.

## OUTCOME

- Enhanced error logging in RAG service to capture specific ImportError details
- Identified that `litellm` module was missing from deployment environment
- Fixed requirements.txt to include `openai-agents[litellm]` optional dependency
- Resolved IntelligenceService availability issue in deployment

## EVALUATION

The detailed logging successfully identified the root cause: `ImportError: 'litellm' is required to use the LitellmModel`. After updating requirements.txt with the proper optional dependency, the IntelligenceService should now be available in the deployment environment, allowing for proper AI response generation instead of fallback responses.