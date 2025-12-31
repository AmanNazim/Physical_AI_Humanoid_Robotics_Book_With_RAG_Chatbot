---
id: 30
title: "RAG Chatbot Global Architecture Skeleton Creation"
stage: "general"
date_iso: "2025-12-23"
surface: "agent"
model: "claude-opus-4-5-20251101"
feature: "none"
branch: "main"
user: "user"
command: "/sp.implement"
labels: ["architecture", "skeleton", "global", "rag-chatbot"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - "rag_chatbot/backend/main.py"
  - "rag_chatbot/backend/routes/base.py"
  - "rag_chatbot/backend/routes/health.py"
  - "rag_chatbot/shared/config.py"
  - "rag_chatbot/shared/schemas/error.py"
  - "rag_chatbot/shared/schemas/chat.py"
  - "rag_chatbot/shared/schemas/retrieval.py"
  - "rag_chatbot/rag_core/rag_pipeline.py"
  - "rag_chatbot/rag_core/interfaces/retriever_interface.py"
  - "rag_chatbot/rag_core/interfaces/embedder_interface.py"
  - "rag_chatbot/rag_core/interfaces/database_interface.py"
  - "rag_chatbot/rag_core/interfaces/agent_interface.py"
  - "rag_chatbot/rag_core/utils/logger.py"
  - "rag_chatbot/rag_core/utils/text_utils.py"
  - "rag_chatbot/rag_core/utils/timing.py"
  - "rag_chatbot/docs/global_architecture.md"
  - "rag_chatbot/.env"
  - "rag_chatbot/pyproject.toml"
tests: []
outcome: "Successfully created the global architecture skeleton for the RAG Chatbot system"
evaluation: "All required components implemented according to specifications"
---

# RAG Chatbot Global Architecture Skeleton Creation

## Prompt Text

You are operating in **SDD Implementation Mode**.

Implement the complete **GLOBAL RAG CHATBOT SYSTEM** according to the following previously-created SDD files:

MANDATORY TO READ:
- constitution.md (Global RAG Chatbot)
- specification.md (Global RAG Chatbot)
- plan.md (Global RAG Chatbot)
- tasks.md (Global RAG Chatbot) (MUST FOLLOW COMPLETELY END TO END, NO HALLUCINATION)

### 🔥 GLOBAL IMPLEMENTATION GOAL
Generate the full **global architecture skeleton** for the RAG Chatbot system, without writing subsystem logic.
Only create the global structure, interfaces, base modules, shared utilities, and integration scaffolding required for all upcoming subsystems.

Do **NOT** generate subsystem internals (database, embeddings, fastapi backend, agents sdk, chatkit UI).
Instead, generate the **framework** they will plug into.

You must strictly follow the SDD tasks, but your output must align with real production engineering standards.

---

# ✅ GLOBAL ARCHITECTURE REQUIREMENTS

### 1. Create Top-Level Project Structure
Generate the following folder structure:

/rag_chatbot
/backend → FastAPI global integration folder
/frontend → ChatKit integration folder
/agents_sdk → Intelligence layer
/databases → Storage layer
/embedding_pipeline → embedding and chunking layer
/rag_core → Shared retrieval, routing, utils
/shared → Shared config, constants, schemas
/scripts → Admin scripts
/docs → Internal technical documentation


### 2. Initialize All Necessary Files (empty but structured)
Inside these directories, generate:

✔ `__init__.py` files
✔ placeholder module files
✔ global router files
✔ shared pydantic schemas
✔ shared configuration loader
✔ `.env` with all required variables
✔ pyproject.toml for uv (no dependencies yet)

### 3. Create Shared Interfaces for All Subsystems (global contracts)

Create interface definition files such as:

/rag_core/interfaces/retriever_interface.py
/rag_core/interfaces/embedder_interface.py
/rag_core/interfaces/database_interface.py
/rag_core/interfaces/agent_interface.py

These files must:

- Define abstract classes (protocols)
- Specify expected input/output shapes
- Specify future method responsibilities
- NOT contain implementation logic

### 4. Create Global Utility Modules

/rag_core/utils/logger.py
/rag_core/utils/text_utils.py
/rag_core/utils/timing.py

Include:
- A basic logger class with JSON logging format
- A timing decorator for performance monitoring
- Placeholder for error handling utilities

### 5. Global Config System

Create a production-ready configuration layer:

/shared/config.py

It must:

- Load environment variables
- Provide settings objects for:
  - Qdrant
  - Neon
  - Cohere embeddings
  - LLM provider (OpenRouter key)
  - Allowed origins for CORS
- Be dependency-injection friendly

### 6. Global FastAPI Router Shell

Inside:

/backend/main.py
/backend/routes/base.py
/backend/routes/health.py


Implement:

- Base FastAPI app with CORS
- Health check endpoint
- Placeholder routers for:
  `/chat`
  `/embed`
  `/retrieve`
  `/config` etc.

Routes must exist but contain only stubs.

### 7. Define Global RAG Pipeline Skeleton

Inside:

/rag_core/rag_pipeline.py

Define:

- High-level RAG flow contract
- No internal logic
- Placeholder functions:
  - `retrieve_chunks(query)`
  - `generate_answer(query, context_chunks)`
  - `format_response(answer, citations)`

### 8. Global Error Model + Response Schema

Create standard schemas:

/shared/schemas/error.py
/shared/schemas/chat.py
/shared/schemas/retrieval.py


### 9. Include Internal Documentation

Inside `/docs/global_architecture.md`, generate a short explanation of:

- Global flow
- Subsystem boundaries
- Responsibilities of each folder
- Integration rules

---

# 🚫 WHAT NOT TO DO
- Do NOT implement actual database connectors
- Do NOT implement embedding logic
- Do NOT implement retrieval logic
- Do NOT integrate LLMs
- Do NOT implement UI
- Do NOT write real business logic

This stage only produces the **skeleton + interfaces + global integration structure** exactly as an SDD global implementation should.

---

# 🎯 FINAL OUTPUT EXPECTED
Claude should generate:

- The full folder tree
- All file paths
- The initial contents of each file
- High-level architectural glue code
- No subsystem logic

---

## 🚀 Execute now:
**"Now First read the Global Files and then Implement the full global architecture skeleton exactly according to the tasks.md (every thing must follow task.md if contradicts then implement task.md one's) and the instructions above."**

## Response Text

Successfully implemented the global architecture skeleton for the RAG Chatbot system. The implementation includes:

1. Created the complete project structure with all specified directories
2. Initialized all necessary files with proper structure and content
3. Created shared interfaces for all subsystems (retriever, embedder, database, agent)
4. Implemented global utility modules (logger, text utilities, timing)
5. Created a comprehensive configuration system with support for Qdrant, Neon, Cohere, and OpenRouter
6. Built the FastAPI router shell with main application, base routes, and health checks
7. Defined the global RAG pipeline skeleton with proper contracts
8. Created comprehensive error models and response schemas
9. Added internal documentation explaining the global architecture

The architecture follows all specifications from the SDD files while maintaining production engineering standards. All components are structured as a framework that future subsystems can plug into without implementing the actual business logic.