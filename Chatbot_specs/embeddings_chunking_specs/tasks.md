# Tasks: Embeddings & Chunking Pipeline for Global RAG Chatbot System

## 1. PROJECT SETUP TASKS

- [ ] T001 Create directory structure for embeddings subsystem in `backend/src/embeddings/`
- [ ] T002 [P] Install required Python modules via uv including google-generativeai, qdrant-client, asyncpg
- [ ] T003 Create `.env.example` file with GEMINI_API_KEY, QDRANT_API_KEY, NEON_DATABASE_URL
- [ ] T004 Create configuration loader module in `backend/src/config/embedding_config.py`
- [ ] T005 Create constants module in `backend/src/embeddings/constants.py` for chunk size, overlap, etc.
- [ ] T006 Set up logging utilities in `backend/src/utils/embedding_logging.py`
- [ ] T007 Create error handling utilities in `backend/src/utils/embedding_errors.py`
- [ ] T008 Configure chunk size parameters (800-1200 tokens) in configuration
- [ ] T009 Configure overlap parameters (200 tokens) in configuration
- [ ] T010 Configure batching parameters (max 96 chunks per batch) in configuration

## 2. CLASS-BASED ARCHITECTURE FOUNDATION TASKS

- [ ] T011 Create base EmbeddingProcessor class in `backend/src/embeddings/base_processor.py`
- [ ] T012 Create base Chunker class in `backend/src/embeddings/base_chunker.py`
- [ ] T013 Create base EmbeddingGenerator class in `backend/src/embeddings/base_generator.py`
- [ ] T014 Create base DatabaseConnector class in `backend/src/embeddings/base_database.py`
- [ ] T015 Create FileProcessor class for document ingestion in `backend/src/embeddings/file_processor.py`
- [ ] T016 Create TextPreprocessor class for text normalization in `backend/src/embeddings/text_preprocessor.py`
- [ ] T017 Create ChunkProcessor class for chunking operations in `backend/src/embeddings/chunk_processor.py`
- [ ] T018 Create EmbeddingProcessor class for Google Gemini API integration in `backend/src/embeddings/embedding_processor.py`
- [ ] T019 Create DatabaseProcessor class for database operations in `backend/src/embeddings/database_processor.py`
- [ ] T020 Create PipelineManager class for end-to-end processing in `backend/src/embeddings/pipeline_manager.py`

## 4. DOCUMENT INGESTION TASKS

- [ ] T033 Create URL-based document crawler for sitemap.xml in `backend/src/embeddings/url_crawler.py`
- [ ] T034 Implement sitemap.xml parsing from https://amannazim.github.io/Physical_AI_Humanoid_Robotics_Book_With_RAG_Chatbot/sitemap.xml in `backend/src/embeddings/sitemap_parser.py`
- [ ] T035 Create individual URL processor for one-by-one processing in `backend/src/embeddings/url_processor.py`
- [ ] T036 Implement optimized code for fastest embedding generation and storage in `backend/src/embeddings/optimized_processor.py`

## 5. CHUNKING ENGINE TASKS

- [ ] T044 Implement dynamic chunking function with 800-1200 token range in `backend/src/embeddings/chunking_engine.py`
- [ ] T045 Implement chunk overlap logic with 200-token overlap in `backend/src/embeddings/chunking_engine.py`
- [ ] T046 Implement chunk hashing function in `backend/src/embeddings/chunk_hasher.py`
- [ ] T047 Implement deduplication detection using content hashes
- [ ] T048 Implement min/max token rules validation
- [ ] T049 Implement trimming and sanitation logic for chunk boundaries
- [ ] T050 Build chunk metadata structure (chunk_id, order, length, hash) in `backend/src/embeddings/chunk_metadata.py`
- [ ] T051 Create chunk validation tests for size, overlap, consistency in `backend/tests/test_chunking.py`
- [ ] T052 Implement sentence boundary preservation in chunking logic
- [ ] T053 Implement paragraph boundary preservation in chunking logic
- [ ] T054 Add character and token boundary tracking for each chunk

## 6. EMBEDDINGS GENERATION TASKS (Google Gemini API)

- [ ] T055 Write Google Gemini API client wrapper in `backend/src/embeddings/gemini_client.py`
- [ ] T056 Implement embedding batcher function for multiple chunks per request respecting Google Gemini API limits
- [ ] T057 Implement exponential-backoff retry logic (3 retries with 1s, 2s, 4s) in `backend/src/embeddings/retry_handler.py`
- [ ] T058 Implement failed batch requeue system in `backend/src/embeddings/failure_queue.py`
- [ ] T059 Generate embedding vectors using Google Gemini API in `backend/src/embeddings/embedding_generator.py`
- [ ] T060 Attach metadata to each embedding (doc_id, chunk_id, length, hash)
- [ ] T061 Validate vector shape (configurable dimensions: 768, 1536, or 3072) and type in `backend/src/embeddings/vector_validator.py`
- [ ] T062 Implement rate-limit handling for Google Gemini API in `backend/src/embeddings/rate_limiter.py`
- [ ] T063 Record logs for each embedding batch in `backend/src/embeddings/batch_logger.py`
- [ ] T064 Create vector generation tests in `backend/tests/test_embedding_generation.py`
- [ ] T065 Create retry reliability tests in `backend/tests/test_retry_logic.py`
- [ ] T066 Create failure recovery tests in `backend/tests/test_failure_recovery.py`

## 7. DATABASE STORAGE TASKS

- [ ] T067 Create chunks table in Neon Postgres with proper schema from specification
- [ ] T068 Create embeddings table structure for Qdrant with proper payload schema
- [ ] T069 Implement vector storage method for Qdrant in `backend/src/embeddings/vector_storage.py`
- [ ] T070 Write insertion function for Qdrant vectors in `backend/src/embeddings/vector_storage.py`
- [ ] T071 Write transaction-safe batch insertion for Neon in `backend/src/embeddings/metadata_storage.py`
- [ ] T072 Link embeddings to chunks through consistent chunk_id in both databases
- [ ] T073 Ensure indexing on doc_id and chunk_id in Neon Postgres
- [ ] T074 Add uniqueness constraints using chunk_hash in Neon Postgres
- [ ] T075 Create retrieval helper functions for both Qdrant and Neon in `backend/src/embeddings/retrieval_helpers.py`
- [ ] T076 Write database validation tests in `backend/tests/test_database_storage.py`
- [ ] T077 Implement cross-database consistency validation between Qdrant and Neon

## 8. END-TO-END PIPELINE TASKS

### 8.1 — Document → Text → Chunks

- [ ] T078 Build document ingestion pipeline function in `backend/src/embeddings/pipeline.py`
- [ ] T079 Build text extraction pipeline function in `backend/src/embeddings/pipeline.py`
- [ ] T080 Build chunking pipeline function in `backend/src/embeddings/pipeline.py`

### 8.2 — Chunks → Embeddings

- [ ] T081 Build embedding generation pipeline function in `backend/src/embeddings/pipeline.py`

### 8.3 — Embeddings → DB

- [ ] T082 Build database storage pipeline function in `backend/src/embeddings/pipeline.py`

### 8.4 — Pipeline-level error handling

- [ ] T083 Implement pipeline error handling in `backend/src/embeddings/pipeline.py`

### 8.5 — Pipeline progress tracking

- [ ] T084 Implement pipeline progress tracking in `backend/src/embeddings/pipeline_tracker.py`

### 8.6 — Pipeline logging

- [ ] T085 Implement pipeline logging in `backend/src/embeddings/pipeline_logger.py`

### 8.7 — Pipeline time measurement

- [ ] T086 Implement pipeline timing measurement in `backend/src/embeddings/pipeline_profiler.py`

### Additional pipeline tasks:

- [ ] T087 Create pipeline entry function `generate_embeddings_for_document()` in `backend/src/embeddings/main_pipeline.py`
- [ ] T088 Create pipeline integration tests in `backend/tests/test_pipeline_integration.py`

## 9. RE-EMBEDDING & UPDATE TASKS

- [ ] T089 Implement content-diff algorithm to detect document changes in `backend/src/embeddings/diff_algorithm.py`
- [ ] T090 Detect changed chunks using hash comparison in `backend/src/embeddings/change_detector.py`
- [ ] T091 Re-embed modified chunks only in `backend/src/embeddings/selective_reembedder.py`
- [ ] T092 Delete outdated embeddings from both Qdrant and Neon in `backend/src/embeddings/embedding_cleaner.py`
- [ ] T093 Update version metadata for re-embedded content in `backend/src/embeddings/version_updater.py`
- [ ] T094 Create batch reprocessing function for multiple documents in `backend/src/embeddings/batch_reprocessor.py`
- [ ] T095 Write validation and consistency checks for re-embedding in `backend/src/embeddings/reembed_validator.py`
- [ ] T096 Write tests for re-embedding correctness in `backend/tests/test_reembedding.py`
- [ ] T097 Implement document-level update pipeline in `backend/src/embeddings/document_updater.py`

## 10. PERFORMANCE OPTIMIZATION TASKS

- [ ] T098 Implement chunk caching using content hash keys in `backend/src/embeddings/chunk_cache.py`
- [ ] T099 Implement embedding caching to avoid redundant generation in `backend/src/embeddings/embedding_cache.py`
- [ ] T100 Enable parallel chunking within resource limits in `backend/src/embeddings/parallel_chunker.py`
- [ ] T101 Enable batch parallelization for compute-bound operations in `backend/src/embeddings/parallel_batcher.py`
- [ ] T102 Minimize database roundtrips through bulk operations in `backend/src/embeddings/bulk_operations.py`
- [ ] T103 Optimize insert performance for both Qdrant and Neon in `backend/src/embeddings/performance_optimizer.py`
- [ ] T104 Reduce memory overhead during processing in `backend/src/embeddings/memory_manager.py`
- [ ] T105 Define benchmark scripts for chunking speed in `backend/scripts/benchmark_chunking.py`
- [ ] T106 Define benchmark scripts for embedding speed in `backend/scripts/benchmark_embedding.py`
- [ ] T107 Implement connection pooling optimization for database operations

## 11. FAIL-SAFE & RECOVERY TASKS

- [ ] T108 Create failure-queue table in Neon Postgres for failed embeddings
- [ ] T109 Create reprocessing worker for failed batches in `backend/src/embeddings/reprocessing_worker.py`
- [ ] T110 Create corrupted-chunk fallback handler in `backend/src/embeddings/fallback_handler.py`
- [ ] T111 Create automatic retry scheduler for failed operations in `backend/src/embeddings/retry_scheduler.py`
- [ ] T112 Create exception-safe database writer in `backend/src/embeddings/safe_db_writer.py`
- [ ] T113 Create consistency-check utility in `backend/src/embeddings/consistency_checker.py`
- [ ] T114 Implement logging of all errors in `backend/src/embeddings/error_logger.py`
- [ ] T115 Create CLI tool for manually retrying failed batches in `backend/scripts/retry_failed_batches.py`
- [ ] T116 Implement state tracking for each chunk across operations in `backend/src/embeddings/state_tracker.py`
- [ ] T117 Create audit logging for troubleshooting and recovery in `backend/src/embeddings/audit_logger.py`

## 12. SECURITY & VALIDATION TASKS

- [ ] T118 Implement chunk hashing using SHA-256 in `backend/src/embeddings/security_hasher.py`
- [ ] T119 Verify embedding-chunk alignment between Qdrant and Neon in `backend/src/embeddings/alignment_verifier.py`
- [ ] T120 Detect duplicate chunks using hash comparison in `backend/src/embeddings/duplicate_detector.py`
- [ ] T121 Enforce chunk-size boundaries (800-1200 tokens) in validation
- [ ] T122 Validate API keys before making Google Gemini API requests in `backend/src/embeddings/api_validator.py`
- [ ] T123 Validate metadata integrity before database storage in `backend/src/embeddings/metadata_validator.py`
- [ ] T124 Write security-focused tests in `backend/tests/test_security.py`
- [ ] T125 Implement API key security in environment variables only
- [ ] T126 Create content sanitization to prevent injection attacks

## 13. DEVELOPER UTILITIES TASKS

- [ ] T127 Create CLI command for embedding documents in `backend/scripts/embed_document.py`
- [ ] T128 Create CLI command for re-embedding in `backend/scripts/reembed_document.py`
- [ ] T129 Create CLI command for checking embedding stats in `backend/scripts/embedding_stats.py`
- [ ] T130 Create CLI for re-running failed batches in `backend/scripts/retry_batches.py`
- [ ] T131 Create debugging utilities to print chunk tree in `backend/scripts/debug_chunk_tree.py`
- [ ] T132 Create debugging utilities to print embedding metadata in `backend/scripts/debug_metadata.py`
- [ ] T133 Document each utility in `backend/docs/embedding_utilities.md`
- [ ] T134 Create health check script for embeddings subsystem in `backend/scripts/health_check.py`
- [ ] T135 Create validation script to verify subsystem integrity in `backend/scripts/validate_integrity.py`

## 14. FINAL VERIFICATION TASKS

- [ ] T136 Run full pipeline tests in `backend/tests/test_full_pipeline.py`
- [ ] T137 Run load tests for large document processing in `backend/tests/test_load.py`
- [ ] T138 Run embedding correctness tests in `backend/tests/test_embedding_correctness.py`
- [ ] T139 Run metadata validity tests in `backend/tests/test_metadata_validity.py`
- [ ] T140 Run performance benchmarks in `backend/tests/test_performance.py`
- [ ] T141 Generate final verification report in `backend/reports/final_verification.md`
- [ ] T142 Ensure all tasks meet Constitution requirements
- [ ] T143 Ensure all tasks meet Specification requirements
- [ ] T144 Create acceptance criteria validation script in `backend/scripts/validate_acceptance.py`
- [ ] T145 Perform end-to-end integration test with database subsystem
- [ ] T146 Perform integration test with Google Gemini API compliance
- [ ] T147 Document any deviations from original plan in `backend/docs/deviations.md`
