---
id: 1
title: "Intelligence Layer (OpenAI Agents SDK) Implementation"
stage: "general"
date_iso: "2026-01-08"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "main"
user: "user"
command: "sp.implement"
labels: ["implementation", "agents-sdk", "rag", "openai"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "rag_chatbot/agents_sdk/__init__.py"
  - "rag_chatbot/agents_sdk/services/intelligence_service.py"
  - "rag_chatbot/agents_sdk/prompts/__init__.py"
  - "rag_chatbot/agents_sdk/prompts/prompt_templates.py"
  - "rag_chatbot/agents_sdk/config.py"
  - "rag_chatbot/requirements.txt"
  - "rag_chatbot/test_intelligence_service.py"
tests:
  - "rag_chatbot/test_intelligence_service.py"
---

# Intelligence Layer (OpenAI Agents SDK) Implementation

## Prompt Text

```
You are "SENIOR EXPERT PYTHON DEVELOPER" with expertise in "OPENAI AGENTS SDK" and "PROMPT ENGINEERING" operating in **SDD Implementation Mode**.

Your task is to fully IMPLEMENT the **Agents SDK Subsystem** for the RAG Chatbot.

INPUT: (MANDATORY) (You Must follow these files)
- Chatbot_specs/agents_sdk_specs/constitution.md
- Chatbot_specs/agents_sdk_specs/specificaion.md
- Chatbot_specs/agents_sdk_specs/plan.md
- Chatbot_specs/agents_sdk_specs/tasks.md

you can also check already implemented subsystem files also in Chatbot_specs folder.

This subsystem MUST be:
- Fully integrated with the already implemented: (MUSK CHECK ALREADY IMPLEMENTED SUBSYSTEM FOR SEAMLESS INTEGRATION)
  - Database subsystem (Qdrant + Neon)
  - Embeddings subsystem
  - FastAPI backend
- Designed for seamless future integration with the ChatKit UI subsystem
- Built using:
  - OpenAI Agents SDK (skill)
  - Prompt Engineering (skill)
  - Context Engineering
  - Tool-based reasoning
  - Streaming-first response generation
- Optimized for fast latency, deterministic behavior, and safe outputs

You MUST follow the existing global architecture, interfaces, and contracts.
Do NOT create new parallel architectures.

────────────────────────────────────────────
## 🎯 CORE RESPONSIBILITY OF THIS SUBSYSTEM

The Agents SDK Subsystem is the **intelligence layer** of the system.

It is responsible for:
- Understanding user intent
- Deciding when to retrieve knowledge
- Calling retrieval logic (Qdrant via Database subsystem)
- Composing context-aware prompts
- Generating grounded responses using an LLM
- Streaming responses token-by-token
- Enforcing guardrails and persona constraints

This subsystem does NOT:
- Handle HTTP routing directly
- Store embeddings
- Manage UI rendering

────────────────────────────────────────────
## 🧠 AGENT DESIGN REQUIREMENTS

### 1. Primary Agent Persona
Implement a **single primary agent** with:

- Role:
  *"Expert Technical Instructor for the Physical AI & Humanoid Robotics Book"*

- Persona constraints:
  - Authoritative but friendly
  - Technically precise
  - Never hallucinates
  - Never answers outside the book content unless explicitly allowed
  - Clearly states uncertainty when context is insufficient

- Tone:
  - Concise
  - Structured
  - Educational
  - Uses bullet points and steps when helpful

### 2. Prompt Engineering (MANDATORY)
Implement a **layered prompt system**:

1. **System Prompt**
   - Defines role, scope, refusal rules
   - Enforces "answer only from retrieved context" behavior
   - Disables speculation

2. **Context Prompt**
   - Injects retrieved chunks
   - Includes metadata (chapter, module, section)
   - Clearly labels content as "SOURCE MATERIAL"

3. **User Prompt**
   - Raw user question
   - Optional selected text (if provided by UI)

4. **Output Instruction Prompt**
   - Enforces formatting rules
   - Disallows markdown abuse
   - Enforces citation style if applicable

All prompts must be composable and inspectable.

────────────────────────────────────────────
## 🔧 TOOL / SKILL IMPLEMENTATION (MANDATORY)

Use **OpenAI Agents SDK Skills** to implement tools.

You MUST implement the following skills:

### 1. Retrieval Skill
- Name: `retrieve_book_context`
- Input: user query + optional selected text of book
- Behavior:
  - Calls the Database subsystem's Qdrant retrieval interface
  - Uses embeddings already stored
  - Applies relevance threshold
  - Returns top-K chunks with metadata
- Output:
  - Clean, ordered context block

### 2. Conversation Memory Skill
- Name: `load_conversation_context`
- Behavior:
  - Fetches recent conversation history from Neon
  - Limits token usage
  - Formats history safely

### 3. Safety / Guardrail Skill
- Name: `validate_response_scope`
- Behavior:
  - Ensures response only references retrieved context
  - Blocks hallucinated external facts
  - Enforces refusal patterns if user asks outside scope

────────────────────────────────────────────
## 🔄 AGENT EXECUTION FLOW (MUST IMPLEMENT)

The agent MUST follow this deterministic flow:

1. Receive request from FastAPI service layer
2. Parse:
   - user query
   - optional selected text
   - page/chapter metadata
3. Decide:
   - retrieval required or not
4. If retrieval required:
   - call `retrieve_book_context`
5. Load short conversation history
6. Compose final prompt using prompt layers
7. Call LLM via **OpenRouter API key** (managed securely)
8. Stream response tokens incrementally
9. Validate output with guardrail skill
10. Return streamed response to FastAPI

────────────────────────────────────────────
## 🔑 LLM CONFIGURATION RULES

- LLM provider: OpenRouter (free tier)
- API key must be:
  - Loaded from environment
  - Never exposed to frontend
- Support:
  - Streaming responses
  - Temperature control
- Fail gracefully on rate limits or model failure

────────────────────────────────────────────
## ⚡ PERFORMANCE REQUIREMENTS

You MUST:
- Minimize prompt size
- Trim context aggressively
- Use async execution
- Stream tokens immediately (no buffering)
- Cache retrieval results per request if applicable

────────────────────────────────────────────
## 🔐 SAFETY & RELIABILITY

Implement:
- Timeouts on retrieval
- Timeouts on LLM calls
- Safe fallback message if agent fails
- Structured error propagation to FastAPI

────────────────────────────────────────────
## 🔌 FASTAPI INTEGRATION (MANDATORY)

The agent MUST expose a callable interface that:
- FastAPI can invoke directly
- Accepts a structured request object
- Returns a streaming generator / async iterator

No HTTP logic inside the agent itself.

────────────────────────────────────────────
## 🔮 FUTURE CHATKIT UI COMPATIBILITY

Ensure:
- Streaming format compatible with ChatKit
- Ability to pass `selected_text` explicitly
- Ability to attach metadata like:
  - current page
  - chapter
  - module
- Clean separation between agent logic and UI formatting

────────────────────────────────────────────
## 🚫 STRICTLY DO NOT

- Do NOT reimplement retrieval logic
- Do NOT bypass Database subsystem
- Do NOT embed UI assumptions
- Do NOT hardcode prompts inline without structure
- Do NOT generate mock logic

────────────────────────────────────────────
## 🎯 FINAL DIRECTIVE

Implement the **complete Agents SDK Subsystem**, fully integrated with:
- Database
- Embeddings
- FastAPI

Using:
- OpenAI Agents SDK
- Skills/tools
- Prompt engineering
- Context engineering
- Streaming-first design

The output must be **production-ready, deterministic, and extensible**.

Begin implementation now.
```

## Response Text

Implemented the complete Intelligence Layer (OpenAI Agents SDK) Subsystem for the RAG Chatbot, including:

1. **Intelligence Service** (`agents_sdk/services/intelligence_service.py`):
   - Implemented main agent service with OpenAI Agents SDK integration
   - Created retrieval, conversation memory, and safety guardrail tools
   - Added input and output guardrails for validation
   - Implemented layered prompt composition using prompt engineering techniques
   - Added streaming response functionality

2. **Prompt Templates** (`agents_sdk/prompts/prompt_templates.py`):
   - Created modular prompt system with layered approach (system, context, user, output instruction)
   - Implemented RAG-style prompting
   - Added chain-of-thought prompting capability

3. **Configuration** (`agents_sdk/config.py`):
   - Created agent configuration with model parameters and safety settings
   - Added API key management and validation

4. **Dependencies** (`requirements.txt`):
   - Added openai-agents package

5. **Testing** (`test_intelligence_service.py`):
   - Created comprehensive test suite to validate functionality

The implementation follows all constitutional boundaries, integrates with existing RAGService and RetrievalService, and provides the required skills for retrieval, conversation memory, and safety validation.

## Outcome

Successfully implemented the complete Intelligence Layer (OpenAI Agents SDK) Subsystem that integrates seamlessly with the existing RAG architecture. The system provides advanced reasoning capabilities using OpenAI Agents SDK with proper tool integration, prompt engineering, and safety guardrails.