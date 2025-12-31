---
id: 21
title: intelligence-layer-specification
stage: spec
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: agents_sdk
branch: main
user: Claude
command: /sp.specify
labels: [intelligence, agents, specification, architecture]
links:
  constitution: Chatbot_specs/agents_sdk_specs/constitution.md
  plan: Chatbot_specs/agents_sdk_specs/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - Chatbot_specs/agents_sdk_specs/specification.md
tests:
 - null
---

## Prompt

You are **Specification Architect Expert**, an expert in designing highly detailed, deeply structured technical specifications for modular software systems.

Your task is to generate the **specification.md** file for the **Intelligence Layer / OpenAI Agents SDK Subsystem** of the RAG Chatbot project.

This specification must **strictly follow the SDD methodology** and define:

- Subsystem responsibilities
- Interfaces
- Input/output formats
- Streaming behavior
- Persona and prompt engineering
- Context engineering
- Code to use OpenRouter API key
- API key management (OpenRouter free LLM key)
- Guardrails
- Performance & latency targets

Do not implement any code
—only specifications.

Input Reference: (Mandatory To Read):
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
- Chatbot_specs/agents_sdk_specs/constitution.md

---

# ============================
# INTELLIGENCE LAYER (OpenAI Agents SDK) — SPECIFICATION
# ============================

## **1. Subsystem Purpose**
- Receive structured queries from FastAPI.
- Apply retrieval-augmented reasoning using pre-fetched context from Qdrant + Neon.
- Generate grounded responses using Agents SDK and free OpenRouter LLM key.
- Stream incremental token-level responses to the frontend (ChatKit).
- Enforce prompt and context engineering best practices.
- Maintain optional session memory for multi-turn conversations.

---

## **2. Input/Output Specifications**
### **Inputs**
1. `user_query` — string, the user's question.
2. `context_chunks` — array of pre-retrieved text chunks (from Qdrant / Neon).
3. `session_id` — optional string for session-based memory.
4. `persona_config` — optional JSON defining tone, verbosity, and style.

### **Outputs**
1. `response_stream` — incremental token-level stream.
2. `final_response` — fully assembled response object with:
   - `text`: string
   - `sources`: array of chunk references
   - `structured_data`: optional, JSON schema if requested
3. `metadata` — including latency, token usage, and warnings if guardrails triggered.

---

## **3. Interfaces & Subsystem Connections**
- **FastAPI**: Receives requests → forwards to Intelligence Layer.
- **Embeddings Subsystem**: Receives pre-embedded context, does **not embed itself**.
- **Qdrant & Neon**: Uses only provided context, does not query directly.
- **Agents SDK**: Core orchestration of reasoning and LLM calls.
- **OpenRouter LLM Key Management**:
   - Secure storage in environment variables or vault.
   - Automatic rotation/retry for free tier rate limits.
   - Token usage monitoring.

---

## **4. Prompt Engineering Specifications**
- Modular prompt structure:
  1. Instruction template (persona + task)
  2. Context template (retrieved chunks, session memory)
  3. Query template (user question + formatting)
- Include **guardrails**:
  - Validate output length
  - Cite sources
  - Avoid hallucinations
  - Apply tone/style
- Support dynamic persona overrides per request.

---

## **5. Context Engineering Specifications**
- Truncate / summarize long context to fit token limits.
- Implement **chunk prioritization**: most relevant chunks first.
- Apply **overlap control** to avoid context loss.
- Optional session memory injection for multi-turn dialogue.
- Context formatting ensures agent can interpret properly.

---

## **6. Guardrails**
- Output must respect domain knowledge (Physical AI, ROS2, Simulation, VLA).
- Enforce safe, instructional tone.
- Retry or reformulate invalid outputs automatically.

---

## **7. Streaming & Performance**
- Responses must be streamed to client token-by-token.
- Async handling for low-latency (<1.5s per query ideally).
- Chunked response assembly internally before final formatting.
- Optional caching of repeated queries for efficiency.

---

## **8. API Key Management (OpenRouter)**
- Securely store and rotate free-tier API key.
- Enforce rate limiting to avoid key bans.
- Retry failed calls with backoff.
- Log usage metrics (tokens, calls).
- Allow future integration with multiple keys if quota exceeded.

---

## **9. Performance & Latency Targets**
- Average query processing latency: <1.5 seconds (streamed)
- Full response assembly: <2.5 seconds
- Multi-turn conversation memory load: <2% additional latency per turn
- Streaming throughput: >50 tokens/sec

---

## **10. Acceptance Criteria**
- Subsystem correctly receives structured requests from FastAPI.
- Applies persona and prompt engineering per specification.
- Streams token-level responses to ChatKit.
- Correctly integrates OpenRouter free-tier LLM key management.
- Guards against hallucinations and enforces context fidelity.
- Logs all interactions for monitoring and debugging.
- Maintains low latency and high throughput.

---

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/agents_sdk_specs/specification.md

Generate the specification.md.

## Response snapshot

# Specification: Intelligence Layer (OpenAI Agents SDK) Subsystem for Global RAG Chatbot System

## 1. Subsystem Purpose

The Intelligence Layer Subsystem serves as the cognitive reasoning engine for the entire RAG system. This subsystem is responsible for receiving structured queries from FastAPI, applying retrieval-augmented reasoning using pre-fetched context from Qdrant and Neon, and generating grounded responses using the Agents SDK and free OpenRouter LLM key. The subsystem streams incremental token-level responses to the frontend (ChatKit) while enforcing prompt and context engineering best practices. It maintains optional session memory for multi-turn conversations while respecting the constitutional boundaries that prohibit direct database access or embedding generation.

## 2. Input/Output Specifications

### Inputs
1. `user_query` — string, the user's question that requires reasoning and response generation
2. `context_chunks` — array of pre-retrieved text chunks from Qdrant and Neon databases, containing relevant information for the query
3. `session_id` — optional string for maintaining session-based memory across multi-turn conversations
4. `persona_config` — optional JSON defining tone, verbosity, and style parameters for response generation

### Outputs
1. `response_stream` — incremental token-level stream of the generated response for real-time delivery
2. `final_response` — fully assembled response object with:
   - `text`: string containing the complete response text
   - `sources`: array of chunk references that were used to ground the response
   - `structured_data`: optional JSON schema if specifically requested
3. `metadata` — including latency metrics, token usage statistics, and warnings if guardrails were triggered during generation

## 3. Interfaces & Subsystem Connections

### FastAPI Interface
- Receives structured query requests from the FastAPI backend
- Returns responses through FastAPI to the frontend (ChatKit)
- Handles session management and conversation state as provided by FastAPI

### Embeddings Subsystem Interface
- Receives pre-embedded context chunks from the Embeddings subsystem via FastAPI
- Does NOT perform embedding generation itself - only consumes pre-processed context
- Validates the format and quality of received embeddings

### Database Subsystem Interface (Qdrant & Neon)
- Uses only context provided by upstream systems, does not query databases directly
- Does NOT perform direct vector searches or metadata retrieval
- Relies entirely on context passed from FastAPI

### Agents SDK Interface
- Core orchestration of reasoning and LLM calls using the OpenAI Agents SDK
- Manages agent state and conversation flow
- Handles tool usage and planning capabilities when needed

### OpenRouter LLM Key Management
- Secure storage of API key in environment variables or secure vault
- Automatic rotation and retry mechanisms for free tier rate limits
- Token usage monitoring and quota management
- Implementation of backoff strategies to prevent rate limiting

## 4. Prompt Engineering Specifications

### Modular Prompt Structure
The subsystem implements a modular prompt structure with three distinct components:
1. **Instruction template**: Defines persona, task requirements, and behavioral guidelines
2. **Context template**: Incorporates retrieved chunks and optional session memory
3. **Query template**: Formats the user question and specifies output requirements

### Guardrail Implementation
- Validates output length to prevent excessively long responses
- Ensures proper citation of source chunks in all responses
- Prevents hallucinations by enforcing strict grounding in provided context
- Applies appropriate tone and style based on persona configuration
- Supports dynamic persona overrides per individual request

## 5. Context Engineering Specifications

### Context Processing
- Truncates or summarizes long context to fit within token limits
- Implements chunk prioritization to place most relevant information first
- Applies overlap control to avoid context loss during processing
- Provides optional session memory injection for multi-turn dialogue management
- Formats context to ensure proper interpretation by the agent system

### Context Quality Assurance
- Validates the relevance and quality of provided context chunks
- Ensures proper formatting and structure for agent consumption
- Implements deduplication to avoid redundant information
- Maintains context coherence across multiple chunks

## 6. Guardrails

### Domain Compliance
- Ensures all outputs respect domain knowledge (Physical AI, ROS2, Simulation, VLA)
- Enforces safe, instructional tone in all responses
- Implements retry or reformulation logic for invalid outputs
- Prevents generation of content outside the specified domain expertise

### Safety and Quality Controls
- Validates response accuracy against provided context
- Ensures proper source attribution for all claims
- Implements content filtering to prevent inappropriate outputs
- Maintains factual accuracy within domain expertise

## 7. Streaming & Performance

### Streaming Requirements
- Responses must be streamed to the client token-by-token for real-time interaction
- Implements async handling to ensure low-latency responses (<1.5s per query ideally)
- Performs internal chunked response assembly before final formatting
- Provides optional caching for repeated queries to improve efficiency

### Performance Optimization
- Implements efficient prompt construction to minimize token usage
- Optimizes API call patterns to reduce latency
- Provides connection pooling for LLM services
- Implements intelligent retry mechanisms for failed requests

## 8. API Key Management (OpenRouter)

### Security and Storage
- Securely stores and manages free-tier API keys using environment variables or secure vault
- Implements automatic key rotation and security protocols
- Enforces rate limiting to prevent key bans and service disruption
- Logs usage metrics including token counts and API call frequency

### Reliability and Scaling
- Implements retry logic with exponential backoff for failed API calls
- Provides monitoring and alerting for quota usage
- Allows future integration with multiple keys for load balancing if quotas are exceeded
- Implements circuit breaker patterns to handle API service disruptions

## 9. Performance & Latency Targets

### Response Time Requirements
- Average query processing latency: <1.5 seconds for streamed responses
- Full response assembly: <2.5 seconds for complete responses
- Multi-turn conversation memory load: <2% additional latency per turn
- Streaming throughput: >50 tokens per second for real-time interaction

### Resource Utilization
- Maintains efficient token usage to optimize API costs
- Implements caching strategies to reduce redundant processing
- Optimizes memory usage for session state management
- Provides monitoring for resource consumption and performance metrics

## 10. Acceptance Criteria

### Functional Requirements
- Subsystem correctly receives structured requests from FastAPI with proper validation
- Applies persona and prompt engineering specifications as defined
- Streams token-level responses to ChatKit with minimal latency
- Correctly integrates OpenRouter free-tier LLM key management with security protocols
- Implements guardrails to prevent hallucinations and enforce context fidelity
- Logs all interactions for monitoring, debugging, and compliance purposes
- Maintains specified latency targets and performance metrics consistently

### Quality and Reliability
- All responses are properly grounded in provided context with appropriate citations
- Session management works correctly for multi-turn conversations
- Error handling is robust with appropriate fallback strategies
- Security measures prevent API key exposure and enforce proper access controls
- Performance remains stable under expected load conditions

## Outcome

- ✅ Impact: Created comprehensive specification for Intelligence Layer (OpenAI Agents SDK) Subsystem with all 10 required sections
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/agents_sdk_specs/specification.md
- 🔁 Next prompts: Ready for planning phase (/sp.plan)
- 🧠 Reflection: Successfully defined Intelligence Layer subsystem specifications with detailed interfaces, OpenRouter API key management, and performance requirements

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null