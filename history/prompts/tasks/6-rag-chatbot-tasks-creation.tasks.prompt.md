---
id: 6
title: rag-chatbot-tasks-creation
stage: tasks
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: rag-chatbot
branch: main
user: Claude
command: /sp.tasks
labels: [tasks, planning, implementation]
links:
  spec: Chatbot_specs/specification.md
  plan: Chatbot_specs/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - Chatbot_specs/tasks.md
tests:
 - null
---

## Prompt

You are operating as an SDD (Specification-Driven Development) task generator.
Your job is to generate the **tasks.md** file for the RAG Chatbot System, strictly aligned with
the Constitution, Specification, and Plan.

The output must be a full breakdown of actionable development tasks — **not implementation code** —
and every task must be directly executable.

The tasks must follow the milestones from the plan, but be more granular, precise, and fully
sequenced. Nothing vague is allowed. Every task must produce a specific artifact, file,
function, or service.

Input Reference: (Mandatory To Read):
- Chatbot_specs/constitution.md
- Chatbot_specs/specification.md
- Chatbot_specs/plan.md

============================================================
SYSTEM NAME:
RAG Chatbot for “Physical AI Humanoid Robotics” Book
============================================================

# RULES FOR TASKS.MD
- Do NOT include code.
- Every task must be 100% actionable.
- Tasks must be grouped under the same milestones defined in plan.md.
- Tasks must not contradict any detail in Constitution or Specification.
- Tasks must include:
  - File creation tasks
  - Directory structure tasks
  - API creation tasks
  - Database setup tasks
  - Embeddings pipeline tasks
  - Intelligence layer tasks
  - ChatKit UI tasks
  - Testing + optimization tasks
  - Deployment tasks
- All tasks must assume:
  - Cohere embeddings
  - FastAPI backend running with uv
  - Qdrant Cloud (vector DB)
  - Neon Serverless Postgres (metadata + documents)
  - Agents SDK as the orchestration/intelligence layer
  - ChatKit for frontend UI

============================================================
TASK GROUPS AND REQUIRED CONTENT
============================================================

# 1. Milestone 1 — Initialization & Environment Setup
Create tasks for:
- Creating repo folders (`backend`, `frontend`, `data`, `scripts`)
- Creating `pyproject.toml` via uv
- Installing all Python dependencies
- Creating `.env` with keys for Cohere, Qdrant, Neon
- Setting up Git ignore rules
- Verifying uv environment
- Creating basic FastAPI scaffold
- Creating ChatKit UI scaffold

# 2. Milestone 2 — Qdrant + Neon Database Setup
Tasks must include:
- Create Qdrant collection with required vector size
- Write script to create index + define payload schema
- Create Neon schema and tables (`documents`, `chunks`, `metadata`)
- Write connection utilities for Qdrant and Neon
- Write a database health-check test for both
- Create migration scripts (if needed)

# 3. Milestone 3 — Chunking & Embedding Pipeline
Tasks must include:
- Create document ingestion script
- Implement chunker (rule-based, aligned with spec)
- Implement Cohere embedding service wrapper
- Write code to store:
  - vectors in Qdrant
  - metadata + chunk content in Neon
- Create background embedding script for large content
- Add validation tasks (token size, truncation detection)
- Add logging tasks
- Add tests for chunking + embedding correctness

# 4. Milestone 4 — Retrieval Pipeline
Tasks must include:
- Implement vector search wrapper for Qdrant
- Implement chunk metadata fetcher for Neon
- Implement retrieval ranker
- Implement context assembly module
- Implement selected-text-only mode logic
- Write tests for:
  - missing results
  - multiple matches
  - context overflow
  - hallucination checks

# 5. Milestone 5 — FastAPI Backend
Tasks must include:
- Create all required endpoints:
  - `/query`
  - `/selected-text`
  - `/retrieve`
  - `/embed`
  - `/agent/route`
  - `/health`
  - Additional that are Required
- Implement request/response models
- Implement async controllers
- Integrate retrieval pipeline
- Integrate embedding pipeline
- Implement rate limiting (basic)
- Add logging + error middleware
- Add integration tests for each endpoint
- Implement CORS rules for ChatKit frontend

# 6. Milestone 6 — Intelligence Layer (Agents SDK)
Tasks must include:
- Define agent role + behavior
- Define agent tools:
  - Qdrant search tool
  - Neon metadata fetch tool
- Implement agent reasoning pipeline
- Implement hallucination guardrails
- Implement answer formatting rules (citations, source text)
- Integrate agent with FastAPI endpoint
- Test agent reasoning with:
  - long queries
  - missing context
  - selected-text-only queries

# 7. Milestone 7 — ChatKit UI Integration
Tasks must include:
- Build ChatKit UI layout
- Create message composer
- Connect ChatKit frontend to FastAPI REST endpoints
- Implement document selection UI for "selected text only"
- Implement message streaming (if supported)
- Add source text viewer
- Add error states + loading states
- Add UI tests + manual test script

# 8. Milestone 8 — Optimization & Evaluation
Tasks must include:
- Latency profiling tasks
- Implement caching layer (per spec: retrieved results + embedded text)
- Optimize chunking strategy
- Optimize Qdrant query parameters
- Run retrieval quality audit
- Add tests for:
  - latency
  - accuracy
  - consistency
- Validate all against free-tier limits (Cohere, Qdrant, Neon)

# 9. Milestone 9 — Deployment
Tasks must include:
- Select server platform (Railway/Render/Fly.io) best for free.
- Create deployment config for uv + FastAPI
- Set up environment variables in host
- Deploy backend
- Deploy ChatKit frontend
- Run post-deployment health checks
- Create deployment troubleshooting guide

# 10. Final Delivery Tasks
- Produce architecture document
- Produce testing report
- Produce developer onboarding guide
- Confirm system meets acceptance criteria

============================================================
FORMAT REQUIREMENTS
============================================================
- Output must be a clean, multi-section Markdown document.
- Every milestone must contain detailed task lists.
- No implementation code.
- No vague statements.
- Everything must directly follow the Constitution, Specification, and Plan.

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/tasks.md.

## Response snapshot

# Tasks: RAG Chatbot for "Physical AI Humanoid Robotics" Book

## 1. Milestone 1 — Initialization & Environment Setup

- [ ] T001 Create project folder structure with backend, frontend, data, scripts directories
- [ ] T002 [P] Initialize uv project in backend directory with Python 3.10+ requirement
- [ ] T003 [P] Create pyproject.toml with required dependencies: fastapi, uvicorn, qdrant-client, asyncpg, cohere, openai, python-dotenv, pydantic
- [ ] T004 Create .env template file with required API keys: COHERE_API_KEY, QDRANT_API_KEY, NEON_DATABASE_URL, OPENAI_API_KEY
- [ ] T005 [P] Set up .gitignore file with Python, environment, and IDE specific patterns
- [ ] T006 [P] Verify uv environment by running uv sync command successfully
- [ ] T007 Create basic FastAPI application scaffold in backend/src/main.py
- [ ] T008 Create basic ChatKit UI scaffold in frontend directory with package.json
- [ ] T009 Create configuration module in backend/src/config/settings.py for environment management
- [ ] T010 Set up initial directory structure in backend: models, services, api, utils, config

## 2. Milestone 2 — Qdrant + Neon Database Setup

- [ ] T011 Create Qdrant collection named "book_embeddings" with 1024-dimensional vectors
- [ ] T012 Define payload schema for Qdrant with chunk_id, text_content, document_reference, and metadata fields
- [ ] T013 Create Neon Postgres schema with chunks, logs, and chat_history tables as specified
- [ ] T014 Create connection utility module for Qdrant in backend/src/utils/qdrant_client.py
- [ ] T015 Create connection utility module for Neon in backend/src/utils/neon_client.py
- [ ] T016 Write database health-check test for Qdrant connectivity in backend/tests/test_qdrant.py
- [ ] T017 Write database health-check test for Neon connectivity in backend/tests/test_neon.py
- [ ] T018 Create database migration scripts for Neon schema in backend/scripts/migrations/
- [ ] T019 Implement proper indexing for efficient queries in both Qdrant and Neon
- [ ] T020 Create utility functions for vector operations in backend/src/utils/vector_utils.py

## 3. Milestone 3 — Chunking & Embedding Pipeline

- [ ] T021 Create document ingestion script in backend/src/scripts/ingest_documents.py
- [ ] T022 Implement chunker with 800-1200 token size constraints in backend/src/utils/chunker.py
- [ ] T023 Implement Cohere embedding service wrapper in backend/src/services/embedding_service.py
- [ ] T024 Create function to store vectors in Qdrant in backend/src/services/vector_storage.py
- [ ] T025 Create function to store metadata and chunk content in Neon in backend/src/services/metadata_storage.py
- [ ] T026 Create background embedding script for large content in backend/src/scripts/background_embed.py
- [ ] T027 Add token size validation task in backend/src/utils/validation.py
- [ ] T028 Add truncation detection validation in backend/src/utils/validation.py
- [ ] T029 Implement logging for chunking and embedding process in backend/src/utils/logging.py
- [ ] T030 Create tests for chunking and embedding correctness in backend/tests/test_chunking.py
- [ ] T031 Implement retry logic for failed embedding requests in backend/src/services/embedding_service.py
- [ ] T032 Create caching mechanism for repeated embeddings in backend/src/services/embedding_service.py

## 4. Milestone 4 — Retrieval Pipeline

- [ ] T033 Implement vector search wrapper for Qdrant in backend/src/services/retrieval_service.py
- [ ] T034 Implement chunk metadata fetcher for Neon in backend/src/services/retrieval_service.py
- [ ] T035 Implement retrieval ranker algorithm in backend/src/services/retrieval_service.py
- [ ] T036 Implement context assembly module in backend/src/services/context_service.py
- [ ] T037 Implement selected-text-only mode logic in backend/src/services/retrieval_service.py
- [ ] T038 Write test for missing results scenario in backend/tests/test_retrieval.py
- [ ] T039 Write test for multiple matches scenario in backend/tests/test_retrieval.py
- [ ] T040 Write test for context overflow scenario in backend/tests/test_retrieval.py
- [ ] T041 Write test for hallucination checks in backend/tests/test_retrieval.py
- [ ] T042 Optimize retrieval performance to meet <1.5s target in backend/src/services/retrieval_service.py

## 5. Milestone 5 — FastAPI Backend

- [ ] T043 Create /query endpoint with mode selection in backend/src/api/query.py
- [ ] T044 Create /selected-text endpoint for user-provided content in backend/src/api/selected_text.py
- [ ] T045 Create /retrieve endpoint for direct retrieval operations in backend/src/api/retrieve.py
- [ ] T046 Create /embed endpoint for on-demand processing in backend/src/api/embed.py
- [ ] T047 Create /agent/route endpoint for intelligence layer in backend/src/api/agent_route.py
- [ ] T048 Create /health endpoint for monitoring in backend/src/api/health.py
- [ ] T049 Implement request/response models in backend/src/models/request_models.py
- [ ] T050 Implement async controllers for all endpoints in backend/src/api/
- [ ] T051 Integrate retrieval pipeline with API endpoints in backend/src/api/query.py
- [ ] T052 Integrate embedding pipeline with API endpoints in backend/src/api/embed.py
- [ ] T053 Implement rate limiting middleware in backend/src/middleware/rate_limit.py
- [ ] T054 Add logging and error middleware in backend/src/middleware/
- [ ] T055 Add integration tests for each endpoint in backend/tests/test_api.py
- [ ] T056 Implement CORS rules for ChatKit frontend in backend/src/main.py
- [ ] T057 Create API documentation with FastAPI automatic docs in backend/src/main.py

## 6. Milestone 6 — Intelligence Layer (Agents SDK)

- [ ] T058 Define agent role and behavior in backend/src/services/agent_service.py
- [ ] T059 Create Qdrant search tool for agent in backend/src/tools/qdrant_search_tool.py
- [ ] T060 Create Neon metadata fetch tool for agent in backend/src/tools/neon_metadata_tool.py
- [ ] T061 Implement agent reasoning pipeline in backend/src/services/agent_service.py
- [ ] T062 Implement hallucination guardrails in backend/src/services/agent_service.py
- [ ] T063 Implement answer formatting rules with citations in backend/src/services/agent_service.py
- [ ] T064 Integrate agent with FastAPI endpoint in backend/src/api/agent_route.py
- [ ] T065 Test agent reasoning with long queries in backend/tests/test_agent.py
- [ ] T066 Test agent reasoning with missing context in backend/tests/test_agent.py
- [ ] T067 Test agent reasoning with selected-text-only queries in backend/tests/test_agent.py
- [ ] T068 Add citation tool for source tracking in backend/src/tools/citation_tool.py

## 7. Milestone 7 — ChatKit UI Integration

- [ ] T069 Build ChatKit UI layout in frontend/src/components/Layout.jsx
- [ ] T070 Create message composer component in frontend/src/components/MessageComposer.jsx
- [ ] T071 Connect ChatKit frontend to FastAPI REST endpoints in frontend/src/services/api.js
- [ ] T072 Implement document selection UI for "selected text only" mode in frontend/src/components/DocumentSelector.jsx
- [ ] T073 Implement message streaming for real-time responses in frontend/src/services/streaming.js
- [ ] T074 Add source text viewer component in frontend/src/components/SourceViewer.jsx
- [ ] T075 Add error states and loading indicators in frontend/src/components/
- [ ] T076 Add UI tests for ChatKit components in frontend/tests/
- [ ] T077 Create manual test script for UI functionality in frontend/test-scripts/

## 8. Milestone 8 — Optimization & Evaluation

- [ ] T078 Perform latency profiling to ensure <1.5 second response times in backend/src/utils/profiling.py
- [ ] T079 Implement caching layer for retrieved results in backend/src/services/cache_service.py
- [ ] T080 Implement caching for embedded text in backend/src/services/cache_service.py
- [ ] T081 Optimize chunking strategy based on performance metrics in backend/src/utils/chunker.py
- [ ] T082 Optimize Qdrant query parameters for better performance in backend/src/services/retrieval_service.py
- [ ] T083 Run retrieval quality audit with test queries in backend/scripts/quality_audit.py
- [ ] T084 Add latency tests in backend/tests/test_performance.py
- [ ] T085 Add accuracy tests in backend/tests/test_accuracy.py
- [ ] T086 Add consistency tests in backend/tests/test_consistency.py
- [ ] T087 Validate all against free-tier limits (Cohere, Qdrant, Neon) in backend/scripts/usage_monitor.py
- [ ] T088 Implement connection pooling optimization in backend/src/utils/database_pool.py

## 9. Milestone 9 — Deployment

- [ ] T089 Select server platform (Railway/Render/Fly.io) best for free tier in deployment/
- [ ] T090 Create deployment config for uv + FastAPI in deployment/Dockerfile
- [ ] T091 Set up environment variables in deployment platform for all services
- [ ] T092 Deploy backend service to chosen platform
- [ ] T093 Deploy ChatKit frontend to CDN or static hosting
- [ ] T094 Run post-deployment health checks in deployment/health_check.py
- [ ] T095 Create deployment troubleshooting guide in deployment/troubleshooting.md
- [ ] T096 Configure domain and SSL certificates for deployed services
- [ ] T097 Set up monitoring and alerting for deployed services

## 10. Final Delivery Tasks

- [ ] T098 Produce architecture document summarizing system design in docs/architecture.md
- [ ] T099 Produce testing report with all test results in docs/testing-report.md
- [ ] T100 Produce developer onboarding guide in docs/onboarding.md
- [ ] T101 Confirm system meets acceptance criteria from specification in docs/acceptance-checklist.md
- [ ] T102 Perform end-to-end integration testing in backend/tests/test_e2e.py
- [ ] T103 Create user documentation for the RAG chatbot in docs/user-guide.md
- [ ] T104 Create system maintenance guide in docs/maintenance-guide.md

## Outcome

- ✅ Impact: Created comprehensive task breakdown for RAG chatbot implementation
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/tasks.md
- 🔁 Next prompts: Ready for implementation phase
- 🧠 Reflection: Successfully translated plan into actionable tasks aligned with constitution and specification

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null