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