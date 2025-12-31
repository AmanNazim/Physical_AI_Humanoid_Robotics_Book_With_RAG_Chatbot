---
id: 22
title: intelligence-layer-implementation-plan
stage: plan
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: agents_sdk
branch: main
user: Claude
command: /sp.plan
labels: [intelligence, agents, plan, architecture]
links:
  constitution: Chatbot_specs/agents_sdk_specs/constitution.md
  spec: Chatbot_specs/agents_sdk_specs/specification.md
  ticket: null
  adr: null
  pr: null
files:
 - Chatbot_specs/agents_sdk_specs/plan.md
tests:
 - null
---

## Prompt

You are **Planning Architect Expert**, an expert in translating detailed specifications into fully actionable development plans.

Your task is to generate the **plan.md** file for the **Intelligence Layer / OpenAI Agents SDK Subsystem** of the RAG Chatbot project.

This plan must strictly follow SDD methodology and describe **step-by-step implementation strategy**, including subsystem integration, prompt/context engineering, persona setup, streaming, optimization for fast performance and OpenRouter API key management.

---Input Reference: (Mandatory To Read):
- Chatbot_specs/constitution.md (important)
- Chatbot_specs/specification.md (important)
- Chatbot_specs/plan.md (important)
- Chatbot_specs/tasks.md (important)
- Chatbot_specs/database_specs/constitution.md
- Chatbot_specs/database_specs/specification.md
- Chatbot_specs/database_specs/plan.md
- Chatbot_specs/database_specs/tasks.md
- Chatbot_specs/embeddings_chunking_specs/constitution.md
- Chatbot_specs/embeddings_chunking_specs/specification.md
- Chatbot_specs/embeddings_chunking_specs/plan.md
- Chatbot_specs/embeddings_chunking_specs/tasks.md
- Chatbot_specs/fast_api_specs/constitution.md
- Chatbot_specs/fast_api_specs/specification.md
- Chatbot_specs/fast_api_specs/plan.md
- Chatbot_specs/fast_api_specs/tasks.md
- Chatbot_specs/agents_sdk_specs/constitution.md (important)
- Chatbot_specs/agents_sdk_specs/specification.md (important)


# ============================
# INTELLIGENCE LAYER (OpenAI Agents SDK) — PLAN
# ============================

## **1. Project Setup**
- Setup Python/uv environment for Agents SDK integration.
- Create subsystem directory structure:
  - agents_sdk/
    - services/
    - prompts/
    - context/
    - streaming/
    - utils/
- Setup environment variable management for **OpenRouter API key** (support secure storage & rotation).
- Setup logging and monitoring tools for agent interactions and performance metrics.

---

## **2. Persona & Prompt Engineering**
- Define base agent persona:
  - Tone: professional, educational, instructional
  - Knowledge domain: Physical AI, Robotics, ROS2, Simulation, VLA
  - Output style: structured, stepwise, context-citing
- Create prompt templates:
  - Instruction template
  - Context template
  - Query template
- Implement prompt chaining for complex reasoning tasks.
- Implement dynamic persona overrides per user request.

---

## **3. Context Engineering**
- Design subsystem to receive pre-fetched context from Qdrant and Neon.
- Implement:
  - Chunk prioritization (relevance scoring)
  - Token truncation to respect LLM limits
  - Overlap control to maintain context continuity
  - Optional session memory injection for multi-turn conversations
- Ensure context is formatted and sanitized before passing to LLM.

---

## **4. OpenRouter API Key Management**
- Securely store free-tier API key.
- Implement retry logic for quota-limited or failed requests.
- Support key rotation for uninterrupted service.
- Track usage metrics (tokens, calls, errors) for monitoring.

---

## **5. Agents SDK Integration**
- Implement orchestration to:
  - Receive user query + context
  - Apply persona & prompt engineering
  - Forward structured request to OpenRouter LLM via Agents SDK
  - Receive streamed token responses
  - Assemble final response including source references and structured metadata
- Implement guardrails:
  - Check token limits
  - Validate output formatting
  - Reject hallucinated or irrelevant content
- Implement fallback strategies if LLM fails or returns invalid output.

---

## **6. Streaming & Response Handling**
- Stream token-level responses to FastAPI for ChatKit UI.
- Assemble incremental responses into final structured object.
- Include latency tracking and throughput monitoring.
- Optionally cache repeated queries to optimize performance.

---

## **7. Subsystem Integration**
- Connect to FastAPI backend:
  - Input: structured user queries and pre-fetched context
  - Output: streamed response & metadata
- Ensure proper error handling and retries between FastAPI ↔ Intelligence Layer.
- Maintain session memory mapping by `session_id` if provided.
- Maintain statelessness per request unless session memory is explicitly used.

---

## **8. Testing & Validation Plan**
- Unit test each component:
  - Persona injection
  - Prompt templates
  - Context chunking and formatting
  - OpenRouter API key usage & retry
- Integration tests:
  - FastAPI ↔ Agents SDK ↔ Streaming
  - Full RAG workflow with sample documents and queries
- Performance testing:
  - Streaming latency <1.5s
  - Final response assembly <2.5s
  - Multi-turn memory load <2% latency impact

---

## **9. Deployment & Monitoring Plan**
- Configure production environment with secure API key storage.
- Enable logging of all queries and responses for audit.
- Monitor latency, token usage, and error rates.
- Ensure streaming endpoints are production-ready.
- Optional: integrate metrics dashboard for real-time monitoring.

---

## **10. Acceptance Criteria**
- Persona applied correctly per query.
- Context engineering maintains relevance and token budget.
- Guardrails enforced: no hallucinations, correct formatting, source citations.
- Streaming works token-by-token to frontend (ChatKit).
- OpenRouter API key management ensures uninterrupted free-tier access.
- All tests pass (unit, integration, performance).
- Latency and throughput targets met.

---

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/agents_sdk_specs/plan.md

Generate the plan.md.

## Response snapshot

# Implementation Plan: Intelligence Layer (OpenAI Agents SDK) Subsystem for Global RAG Chatbot System

## 1. Project Setup

- Setup Python/uv environment for Agents SDK integration with proper dependencies
- Create subsystem directory structure:
  - agents_sdk/
    - services/
    - prompts/
    - context/
    - streaming/
    - utils/
- Setup environment variable management for **OpenRouter API key** with secure storage and rotation capabilities
- Setup logging and monitoring tools for agent interactions and performance metrics
- Configure secure credential storage and access patterns
- Initialize configuration management for different environments (dev, staging, prod)

## 2. Persona & Prompt Engineering

- Define base agent persona with:
  - Tone: professional, educational, instructional
  - Knowledge domain: Physical AI, Robotics, ROS2, Simulation, VLA
  - Output style: structured, stepwise, context-citing
- Create prompt templates with modular structure:
  - Instruction template with persona and task guidelines
  - Context template for retrieved chunks and session memory
  - Query template for user questions and formatting requirements
- Implement prompt chaining for complex reasoning tasks
- Implement dynamic persona overrides per user request
- Develop prompt validation and testing framework
- Create prompt optimization strategies for token efficiency

## 3. Context Engineering

- Design subsystem to receive pre-fetched context from Qdrant and Neon via FastAPI
- Implement context preprocessing pipeline:
  - Chunk prioritization with relevance scoring algorithms
  - Token truncation to respect LLM limits with intelligent summarization
  - Overlap control to maintain context continuity and avoid information loss
  - Optional session memory injection for multi-turn conversations
- Ensure context formatting and sanitization before passing to LLM
- Implement context quality validation and relevance checking
- Develop context assembly and disassembly mechanisms for efficient processing

## 4. OpenRouter API Key Management

- Implement secure storage of free-tier API key using environment variables and vault integration
- Implement retry logic with exponential backoff for quota-limited or failed requests
- Support key rotation mechanisms for uninterrupted service
- Track usage metrics (tokens, calls, errors) for monitoring and optimization
- Implement rate limiting and quota management to stay within free tier limits
- Create API key health checks and automatic failover mechanisms
- Implement logging for API usage and cost tracking

## 5. Agents SDK Integration

- Implement orchestration layer to:
  - Receive user query and context from FastAPI
  - Apply persona and prompt engineering
  - Forward structured request to OpenRouter LLM via Agents SDK
  - Receive and process streamed token responses
  - Assemble final response including source references and structured metadata
- Implement comprehensive guardrails:
  - Token limit checking and management
  - Output formatting validation
  - Hallucination detection and prevention
  - Content relevance and quality validation
- Implement fallback strategies for LLM failures or invalid outputs
- Create agent state management for conversation continuity
- Implement tool usage and planning capabilities when needed

## 6. Streaming & Response Handling

- Implement token-level streaming to FastAPI for ChatKit UI with real-time delivery
- Develop incremental response assembly into final structured objects
- Include comprehensive latency tracking and throughput monitoring
- Implement optional caching for repeated queries to optimize performance
- Create response formatting and validation pipeline
- Implement error handling and recovery for streaming failures
- Develop connection management for WebSocket and HTTP streaming

## 7. Subsystem Integration

- Connect to FastAPI backend with proper interfaces:
  - Input: structured user queries and pre-fetched context
  - Output: streamed response and metadata
- Implement robust error handling and retry mechanisms between FastAPI and Intelligence Layer
- Maintain session memory mapping by `session_id` with proper state management
- Ensure statelessness per request unless session memory is explicitly used
- Implement proper authentication and authorization between subsystems
- Create health check and readiness endpoints for orchestration

## 8. Testing & Validation Plan

- Unit test each component:
  - Persona injection and management
  - Prompt template generation and validation
  - Context chunking, formatting, and preprocessing
  - OpenRouter API key usage, rotation, and retry mechanisms
  - Guardrail enforcement and validation
  - Streaming functionality and error handling
- Integration tests:
  - FastAPI ↔ Agents SDK ↔ Streaming pipeline
  - Full RAG workflow with sample documents and queries
  - End-to-end conversation flow with session management
- Performance testing:
  - Streaming latency targets: <1.5s
  - Final response assembly targets: <2.5s
  - Multi-turn memory load impact: <2% latency increase
  - Throughput and concurrent user handling
- Security testing for API key management and data protection
- Load testing for API quota and rate limiting scenarios

## 9. Deployment & Monitoring Plan

- Configure production environment with secure API key storage and access controls
- Enable comprehensive logging of all queries and responses for audit and debugging
- Implement monitoring for latency, token usage, and error rates
- Ensure streaming endpoints are production-ready with proper scaling
- Integrate metrics dashboard for real-time monitoring and alerting
- Set up automated deployment pipelines with proper environment separation
- Implement backup and recovery procedures for session data
- Create operational runbooks for common issues and maintenance

## 10. Acceptance Criteria

- Persona applied correctly per query with consistent tone and domain expertise
- Context engineering maintains relevance, token budget, and proper formatting
- Guardrails enforced: no hallucinations, correct formatting, accurate source citations
- Streaming works token-by-token to frontend (ChatKit) with minimal latency
- OpenRouter API key management ensures uninterrupted free-tier access with proper usage tracking
- All tests pass (unit, integration, performance) with appropriate coverage
- Latency and throughput targets met consistently across different scenarios
- Proper error handling and graceful degradation when API limits are reached
- Security requirements fulfilled with no API key exposure or data breaches
- Session management works correctly for multi-turn conversations
- Integration with FastAPI and other subsystems functions as specified

## Outcome

- ✅ Impact: Created comprehensive implementation plan for Intelligence Layer (OpenAI Agents SDK) Subsystem with all 10 required sections
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/agents_sdk_specs/plan.md
- 🔁 Next prompts: Ready for tasks generation phase (/sp.tasks)
- 🧠 Reflection: Successfully defined Intelligence Layer subsystem implementation strategy with detailed steps for all components

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null