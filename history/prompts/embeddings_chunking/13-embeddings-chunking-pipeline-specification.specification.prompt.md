---
id: 13
title: embeddings-chunking-pipeline-specification
stage: spec
date: 2025-12-22
surface: agent
model: claude-opus-4-5-20251101
feature: embeddings_chunking
branch: main
user: Claude
command: /sp.specify
labels: [embeddings, chunking, specification, architecture]
links:
  constitution: Chatbot_specs/embeddings_chunking_specs/constitution.md
  plan: Chatbot_specs/embeddings_chunking_specs/plan.md
  ticket: null
  adr: null
  pr: null
files:
 - Chatbot_specs/embeddings_chunking_specs/specification.md
tests:
 - null
---

## Prompt

You are **Specification Architect Expert**, an expert in designing highly detailed, deeply structured technical specifications for modular software systems.

Input Reference: (Mandatory To Read):
- Chatbot_specs/constitution.md
- Chatbot_specs/specification.md
- Chatbot_specs/plan.md
- Chatbot_specs/tasks.md
- Chatbot_specs/database_specs/constitution.md
- Chatbot_specs/database_specs/specification.md
- Chatbot_specs/database_specs/plan.md
- Chatbot_specs/database_specs/tasks.md
- Chatbot_specs/embeddings_chunking_specs/constitution.md

Your task is to generate the **specification.md** file for the subsystem:

# Subsystem: Embeddings & Chunking Pipeline
# Parent System: RAG Chatbot for "Physical AI Humanoid Robotics" Book

You must produce the **exact, detailed, engineering-level specifications** that implement the responsibilities defined in the embeddings constitution.md.

This document is NOT implementation.
It must be a full, formal, deeply detailed SDD specification.

-----------------------------------------------
STRUCTURE & REQUIREMENTS FOR THE SPECIFICATION
-----------------------------------------------

Your output MUST include the following sections exactly:

1. **Subsystem Overview**
2. **Inputs & Input Validation Rules**
3. **Outputs**
4. **Core Responsibilities**
5. **Non-Responsibilities**
6. **Data Model Specifications**
7. **Preprocessing Specifications**
8. **Chunking Specifications**
9. **Chunk Overlap Specifications** (Best Chunking overlap)
10. **Metadata Schema Specifications**
11. **Embedding Generation Specifications (Cohere Free Tier)**
12. **Vector Packaging Specifications**
13. **Database Connectivity Specifications**
     - Qdrant Vector DB
     - Neon Postgres
14. **Vector Upsert Pipeline Specifications**
15. **Error Handling Specifications**
16. **Performance & Optimization Specifications**
17. **Security & Safety Specifications**
18. **Testing Specifications**
19. **Completion Criteria**

You MUST strictly define the behavior of each subsystem stage.

===========================================
1. SUBSYSTEM OVERVIEW
===========================================
Define a clear technical overview of how the subsystem:
- receives text (full book, module, chapter, lesson, or selected text)
- preprocesses it
- chunks it
- applies overlap rules
- generates embeddings using Cohere
- constructs metadata
- sends vectors + metadata to Qdrant
- sends relational metadata to Neon
- returns storage confirmation

This section must be **precise and architectural**.

===========================================
2. INPUTS & VALIDATION RULES
===========================================
Define EXACT allowed inputs:
- full book text
- module text
- chapter/lesson text
- arbitrary user-selected text snippet
- file formats allowed (md, txt, html, docx converted to text)
Define strict validation:
- must not contain binary data
- must be UTF-8 plain text after preprocessing
- maximum text length limits (based on Cohere free tier)
- required fields for metadata

===========================================
3. OUTPUTS
===========================================
Explicitly define:
- vector payloads for Qdrant
- metadata payloads for Neon
- confirmation object returned to FastAPI

===========================================
4. CORE RESPONSIBILITIES
===========================================
Detail responsibilities across:
- Text preprocessing
- Chunk segmentation
- Overlap handling
- Metadata enrichment
- Cohere embedding generation
- Qdrant vector preparation
- Neon metadata preparation

===========================================
5. NON-RESPONSIBILITIES
===========================================
Explicit things this subsystem must not do:
- Must NOT run retrieval
- Must NOT run inference or answer questions
- Must NOT perform RAG ranking
- Must NOT expose any API endpoints
- Must NOT manage user sessions or UI
- Must NOT modify database schemas

===========================================
6. DATA MODEL SPECIFICATIONS
===========================================
Define strict structures for:
- Preprocessed text
- Chunk objects
- Overlapped chunk ranges
- Vector embedding objects
- Qdrant payload schema (JSON)
- Neon relational record schema

===========================================
7. PREPROCESSING SPECIFICATIONS
===========================================
Define:
- text normalization rules
  (Unicode normalization, trimming, removing control chars)
- whitespace normalization
- header/footer stripping rules
- markdown cleaning rules
- removal of irrelevant tokens (HTML tags, scripts)
- sentence boundary rules (if used)
- maximum character limits

===========================================
8. CHUNKING SPECIFICATIONS
===========================================
Define:
- Chunking mode: by tokens or characters
  (default: token-based with fallback to char-based)
- Target chunk size (e.g., 500 tokens or 1500 chars)
- Minimum chunk size
- Maximum chunk size
- Rules for mid-sentence breaks
- Handling oversized paragraphs
- Multi-level granularity (chapter → section → chunk)

===========================================
9. CHUNK OVERLAP SPECIFICATIONS
===========================================
Define a **precise overlap strategy**:
- Overlap size (e.g., 30–35% of chunk size)
- Boundary handling
- Prevent embedding duplication
- Metadata linking parent chunk IDs
- Rules for when overlap is reduced or disabled

===========================================
10. METADATA SCHEMA SPECIFICATIONS
===========================================
Define mandatory metadata:
- chunk_id (UUID)
- parent_structure (book/module/chapter/lesson)
- source_path (hierarchical)
- source_url
- character_start, character_end
- token_start, token_end
- overlap_before, overlap_after
- timestamp_created
- document_version
- content_hash (SHA-256)

Define optional metadata:
- semantic tags
- user selection flag
- language code

Metadata MUST be identical across Qdrant and Neon.

===========================================
11. EMBEDDING GENERATION SPECIFICATIONS
===========================================
Define the exact embedding flow:
- embedding model: Cohere (free tier)
- API call structure
- batching strategy
- max batch size
- rate limit handling
- retry logic
- fallback embeddings (zeros if error)

===========================================
12. VECTOR PACKAGING SPECIFICATIONS
===========================================
Define how chunk text + embed vector + metadata are packaged into:
- Qdrant "points"
- Neon "records"

Ensure perfect alignment:
- same chunk_id
- same metadata
- deterministic ordering

===========================================
13. DATABASE CONNECTIVITY SPECIFICATIONS
===========================================
Define:
- Qdrant Cloud connection rules
  (URL, API key, collection naming, index setup)
- Neon Postgres connection rules
  (connection string, table names, field mappings)

Define strict "connection contract":
- subsystem never alters schema
- subsystem never manages migration
- subsystem must fail fast on invalid schema

===========================================
14. VECTOR UPSERT PIPELINE SPECIFICATIONS
===========================================
Define the end-to-end data flow:
1. Receive preprocessed chunks
2. Generate embeddings
3. Prepare Qdrant vectors
4. Upsert into Qdrant
5. Insert metadata into Neon
6. Return success response

Define atomicity rules:
- If Qdrant fails → abort Neon insert
- If Neon fails → roll back vector insertion

===========================================
15. ERROR HANDLING SPECIFICATIONS
===========================================
Define:
- retry rules for Cohere API
- retry rules for Qdrant + Neon
- malformed input handling
- corrupted chunk detection
- metadata mismatch handling
- logging + reporting format

===========================================
16. PERFORMANCE OPTIMIZATION SPECIFICATIONS
===========================================
Define:
- batching embeddings
- parallel chunk processing
- concurrency limits
- caching frequently accessed sections
- avoiding redundant re-embeddings (hash-based)

===========================================
17. SECURITY & SAFETY SPECIFICATIONS
===========================================
Define:
- API key protection rules
- data integrity hashing
- sanitization flow
- separation of user data vs book content
- safe failure modes

===========================================
18. TESTING SPECIFICATIONS
===========================================
Define tests for:
- preprocessing correctness
- chunking boundaries
- overlap correctness
- metadata consistency across DBs
- embedding generation validity
- Qdrant + Neon integration tests
- load tests for large embeddings

===========================================
19. COMPLETION CRITERIA
===========================================
Explicitly define when this subsystem is considered complete.

-----------------------------------------------

REQUIREMENTS:
- Use polished, professional SDD language.
- NO code.
- NO implementation details.
- NO pseudo-code.
- Produce only the specification.md content.

Output in Chatbot_specs Folder:
- File Type: Markdown
- File Path: Chatbot_specs/embeddings_chunking_specs/specification.md

Generate the specification.md..

## Response snapshot

# Specification: Embeddings & Chunking Pipeline for Global RAG Chatbot System

## 1. Subsystem Overview

The Embeddings & Chunking Pipeline serves as the knowledge ingestion engine of the Global RAG Chatbot System, transforming raw textual content into vector representations suitable for semantic search and retrieval. This subsystem operates as a specialized preprocessing component that receives text content (full book, module, chapter, lesson, or selected text), applies normalization and segmentation, generates vector embeddings using Cohere API, and coordinates storage of vectors and metadata with the Database Subsystem.

The pipeline executes the following architectural workflow:
1. Receives text content through the FastAPI Backend (either book content for ingestion or user-selected text for on-demand processing)
2. Applies preprocessing normalization to clean and standardize the input text
3. Segments the text into appropriately-sized chunks with overlap handling for context preservation
4. Generates metadata for each chunk including source location, boundaries, and content hashes
5. Requests vector embeddings from Cohere API for each chunk
6. Packages embeddings with metadata for Qdrant vector database storage
7. Packages relational metadata for Neon Postgres storage
8. Coordinates the upsert operation with the Database Subsystem to ensure consistency
9. Returns storage confirmation to the requesting component

This subsystem maintains strict separation of concerns by focusing exclusively on knowledge transformation while deferring all storage, retrieval, and reasoning decisions to other subsystems. It ensures deterministic processing results that support the system's constitutional requirement for grounded, non-hallucinated responses.

## 2. Inputs & Input Validation Rules

### Allowed Inputs
- **Full book text**: Complete book content for initial ingestion
- **Module text**: Specific module content from the book
- **Chapter/lesson text**: Individual chapter or lesson content
- **User-selected text snippet**: Arbitrary text provided by users for "selected text only" mode
- **File formats**: Plain text content (md, txt, html, docx converted to text via preprocessing tools)

### Validation Requirements
- **Binary data check**: Input MUST NOT contain binary data; only UTF-8 plain text is acceptable
- **Text format validation**: Content MUST be UTF-8 plain text after preprocessing with no embedded objects
- **Maximum length limits**: Text length MUST NOT exceed Cohere API free tier limits (approximately 4000 tokens per request)
- **Required metadata fields**: For book content, document reference and source location MUST be provided
- **Content integrity**: Text MUST NOT contain personal data, sensitive information, or inappropriate content
- **Encoding validation**: All text MUST be properly encoded in UTF-8 without character corruption
- **Structure validation**: Document structure (if provided) MUST be parseable and coherent

### Validation Procedures
- Text content is validated for UTF-8 compliance before processing
- Content is scanned for binary data signatures and rejected if detected
- Token count is calculated to ensure compliance with Cohere API limits
- Document structure is validated for coherence and proper formatting
- Content is sanitized to remove any potentially harmful elements

## 3. Outputs

### Vector Payloads for Qdrant
- **chunk_id**: UUID identifier matching Neon record
- **text_content**: The actual text content of the chunk
- **document_reference**: Reference to the source document/chapter
- **metadata**: Additional metadata including page numbers, section titles, and processing version
- **vector**: 1024-dimensional embedding vector from Cohere API

### Metadata Payloads for Neon
- **chunk_id**: UUID identifier matching Qdrant record
- **document_reference**: Source document identifier
- **page_reference**: Page number or location within the document
- **section_title**: Section or chapter title for context
- **chunk_text**: The actual text content (for metadata-only queries)
- **embedding_id**: Reference to the corresponding vector in Qdrant
- **created_at**: Timestamp of processing
- **updated_at**: Timestamp of last update
- **processing_version**: Version of the embedding pipeline used

### Confirmation Object for FastAPI
- **status**: Success or failure indicator
- **chunk_ids**: Array of processed chunk identifiers
- **processing_time**: Duration of the processing operation
- **error_details**: Any error information if processing failed
- **validation_results**: Summary of validation performed

## 4. Core Responsibilities

### Text Preprocessing
- Normalize input text by removing extraneous whitespace and special characters
- Apply Unicode normalization to ensure consistent character encoding
- Remove HTML tags, scripts, and other non-content elements
- Apply language-specific text normalization rules
- Validate text for proper structure and coherence

### Chunk Segmentation
- Divide documents into 800-1200 token chunks as specified in system constitution
- Preserve semantic boundaries and document structure during segmentation
- Apply appropriate overlap to maintain context across chunk boundaries
- Generate chunk-level metadata including document references and location tracking

### Overlap Handling
- Implement overlap strategy to maintain context across chunk boundaries
- Ensure overlap does not result in duplicate embeddings
- Maintain parent-child relationships between overlapping chunks
- Track overlap boundaries for proper context assembly

### Metadata Enrichment
- Generate UUID identifiers for each chunk that align with database records
- Extract document reference information for source tracking
- Capture location information (page numbers, section titles)
- Calculate token and character boundaries for each chunk
- Generate content hashes for integrity verification

### Cohere Embedding Generation
- Request 1024-dimensional vector embeddings from Cohere API
- Maintain consistent embedding parameters across all processing
- Handle API rate limiting and retry logic appropriately
- Manage batch processing for efficiency optimization

### Qdrant Vector Preparation
- Package embeddings with appropriate metadata for Qdrant storage
- Ensure vector dimensionality matches 1024-dimensional requirement
- Maintain ID alignment with corresponding Neon records
- Prepare payload schema according to Qdrant requirements

### Neon Metadata Preparation
- Package metadata according to Neon Postgres schema requirements
- Ensure relational integrity with proper foreign key relationships
- Maintain consistency with Qdrant vector records
- Prepare for ACID-compliant storage operations

## 5. Non-Responsibilities

### Explicitly Prohibited Actions
- **Must NOT run retrieval**: This subsystem does not perform vector similarity searches or retrieve information
- **Must NOT run inference or answer questions**: No LLM interaction or response generation occurs in this subsystem
- **Must NOT perform RAG ranking**: Ranking of retrieved results is handled by the Intelligence Layer
- **Must NOT expose any API endpoints**: All interfaces are handled by the FastAPI Backend
- **Must NOT manage user sessions or UI**: User interaction management is handled by the ChatKit UI
- **Must NOT modify database schemas**: Schema management is handled by the Database Subsystem
- **Must NOT store processed content persistently**: All storage coordination is delegated to the Database Subsystem
- **Must NOT perform quality assessment**: Content quality evaluation is outside the scope of this subsystem

## 6. Data Model Specifications

### Preprocessed Text Structure
- **original_content**: Raw text before preprocessing
- **normalized_content**: Text after normalization and cleaning
- **encoding_type**: Character encoding of the content
- **language_code**: ISO language code for the content
- **preprocessing_steps**: Array of applied preprocessing operations

### Chunk Object Structure
- **chunk_id**: UUID identifier for the chunk
- **content**: The actual text content of the chunk
- **token_count**: Number of tokens in the chunk
- **character_start**: Starting character position in original document
- **character_end**: Ending character position in original document
- **token_start**: Starting token position in original document
- **token_end**: Ending token position in original document
- **parent_chunk_id**: Reference to parent chunk if overlapping
- **overlap_type**: Type of overlap (before, after, none)

### Overlapped Chunk Ranges
- **primary_chunk_id**: ID of the main chunk
- **overlap_chunk_id**: ID of the overlapping chunk segment
- **overlap_start**: Starting position of overlap
- **overlap_end**: Ending position of overlap
- **overlap_size**: Size of the overlap in characters/tokens
- **overlap_direction**: Direction of overlap (preceding, succeeding)

### Vector Embedding Object
- **vector_id**: UUID identifier matching chunk record
- **embedding_vector**: Array of 1024 floating-point numbers
- **embedding_model**: Identifier for the embedding model used
- **embedding_parameters**: Parameters used for embedding generation
- **creation_timestamp**: Time of embedding generation
- **quality_score**: Quality assessment of the embedding

### Qdrant Payload Schema (JSON)
- **id**: Chunk identifier (UUID)
- **vector**: 1024-dimensional embedding array
- **payload**: Object containing:
  - **chunk_id**: Chunk identifier
  - **text_content**: Text content preview
  - **document_reference**: Source document identifier
  - **page_reference**: Page or location reference
  - **section_title**: Section title
  - **processing_version**: Pipeline version
  - **content_hash**: SHA-256 hash of content

### Neon Relational Record Schema
- **chunk_id**: UUID primary key
- **document_reference**: VARCHAR - Source document identifier
- **page_reference**: INTEGER - Page or location reference
- **section_title**: VARCHAR - Section title
- **chunk_text**: TEXT - Full text content
- **embedding_id**: UUID - Reference to Qdrant vector
- **created_at**: TIMESTAMP - Creation time
- **updated_at**: TIMESTAMP - Last update time
- **processing_version**: VARCHAR - Pipeline version used

## 7. Preprocessing Specifications

### Text Normalization Rules
- **Unicode normalization**: Apply NFC (Canonical Decomposition followed by Canonical Composition) normalization to ensure consistent character representation
- **Whitespace normalization**: Replace multiple consecutive whitespace characters with a single space, preserve line breaks for document structure
- **Control character removal**: Remove non-printable control characters except standard whitespace characters
- **Special character handling**: Preserve essential punctuation while removing formatting-specific characters
- **Case normalization**: Maintain original case to preserve semantic meaning

### Header/Footer Stripping
- **Document header detection**: Identify and remove standard document headers (page numbers, titles, author information)
- **Footer content removal**: Remove footers containing page numbers, copyright information, or navigation elements
- **Content boundary preservation**: Ensure document content boundaries are maintained after header/footer removal

### Markdown Cleaning
- **Markdown syntax removal**: Remove markdown formatting symbols while preserving content
- **Link preservation**: Extract link text while removing markdown link syntax
- **Code block handling**: Preserve code content while removing formatting
- **List structure preservation**: Maintain list structure while removing formatting symbols

### Irrelevant Token Removal
- **HTML tag removal**: Strip HTML tags while preserving content text
- **Script content removal**: Remove JavaScript and other script content
- **Style information removal**: Remove CSS and other styling information
- **Comment removal**: Strip document comments while preserving content

### Maximum Character Limits
- **Processing unit limits**: Individual processing units MUST NOT exceed 4000 tokens to comply with Cohere API limits
- **Memory usage limits**: Preprocessing operations MUST maintain memory usage under 100MB per processing unit
- **Processing time limits**: Preprocessing operations MUST complete within 30 seconds per processing unit

## 8. Chunking Specifications

### Chunking Mode
- **Primary mode**: Token-based chunking using standard tokenization
- **Fallback mode**: Character-based chunking for non-standard content
- **Adaptive mode**: Switch between token and character modes based on content characteristics

### Chunk Size Parameters
- **Target chunk size**: 1000 tokens (within 800-1200 range specified in constitution)
- **Minimum chunk size**: 800 tokens
- **Maximum chunk size**: 1200 tokens
- **Size tolerance**: ±10% to accommodate semantic boundaries

### Boundary Handling Rules
- **Sentence boundary preservation**: Avoid breaking sentences when possible
- **Paragraph boundary respect**: Preserve paragraph integrity when possible
- **Code block preservation**: Keep code blocks within single chunks when possible
- **Table preservation**: Keep table elements within single chunks when possible

### Oversized Content Handling
- **Oversized paragraph handling**: Split paragraphs that exceed maximum chunk size at semantic boundaries
- **Oversized code handling**: Break large code blocks at logical function or class boundaries
- **Fallback splitting**: Use character-based splitting for content that cannot be logically segmented

### Multi-Level Granularity
- **Document level**: Process entire documents as input units
- **Chapter/section level**: Process chapters or sections when provided as discrete units
- **Chunk level**: Generate chunks within the specified size constraints
- **Overlap level**: Create overlap segments for context preservation

## 9. Chunk Overlap Specifications

### Overlap Strategy
- **Overlap size**: 200 tokens (approximately 20% of target chunk size of 1000 tokens)
- **Overlap position**: Applied to the end of one chunk and beginning of the next
- **Overlap preservation**: Maintain context across semantic boundaries while avoiding duplicate embeddings

### Boundary Handling
- **Semantic boundary respect**: Adjust overlap boundaries to preserve semantic coherence
- **Sentence preservation**: Avoid breaking sentences within overlap regions when possible
- **Context continuity**: Ensure smooth transition of context across overlapping chunks

### Duplicate Prevention
- **Embedding duplication prevention**: Do not generate embeddings for overlap regions to avoid redundancy
- **Metadata linking**: Link overlap chunks to their parent chunks through metadata relationships
- **Query handling**: Handle overlap regions appropriately during retrieval to avoid redundant results

### Parent-Child Relationships
- **Primary chunk identification**: Identify the main chunk that contains the primary content
- **Overlap reference**: Create references from overlap segments to their parent chunks
- **Boundary tracking**: Track overlap boundaries for proper context assembly

### Conditional Overlap Rules
- **Disable for small chunks**: Overlap MAY be reduced or disabled for chunks smaller than 600 tokens
- **Content-dependent adjustment**: Adjust overlap size based on content type (code vs. prose)
- **Boundary proximity**: Reduce overlap when approaching document boundaries

## 10. Metadata Schema Specifications

### Mandatory Metadata Fields

#### Core Identification
- **chunk_id**: UUID - Unique identifier for the chunk, matching across Qdrant and Neon
- **document_reference**: VARCHAR - Reference to the source document (book, chapter, section)
- **source_path**: VARCHAR - Hierarchical path to the source content
- **source_url**: VARCHAR - URL reference to the source document (if applicable)

#### Position Information
- **character_start**: INTEGER - Starting character position in original document
- **character_end**: INTEGER - Ending character position in original document
- **token_start**: INTEGER - Starting token position in original document
- **token_end**: INTEGER - Ending token position in original document

#### Overlap Information
- **overlap_before**: UUID - Reference to preceding overlap chunk (if applicable)
- **overlap_after**: UUID - Reference to succeeding overlap chunk (if applicable)
- **overlap_size**: INTEGER - Size of overlap in tokens/characters

#### Temporal Information
- **timestamp_created**: TIMESTAMP - Time of chunk creation
- **document_version**: VARCHAR - Version identifier of the source document

#### Integrity Information
- **content_hash**: VARCHAR - SHA-256 hash of the chunk content for integrity verification

### Optional Metadata Fields

#### Semantic Information
- **semantic_tags**: JSONB - Array of semantic tags or categories for the content
- **language_code**: VARCHAR - ISO language code for the content
- **content_type**: VARCHAR - Type of content (text, code, table, etc.)

#### User Information
- **user_selection_flag**: BOOLEAN - Indicates if this chunk was selected by user (for selected-text mode)
- **user_id**: UUID - User identifier if content is user-provided (for selected-text mode)

### Metadata Consistency Requirements
- **Cross-database alignment**: All mandatory metadata MUST be identical across Qdrant and Neon
- **UUID consistency**: Chunk IDs MUST match exactly between vector and relational records
- **Timestamp synchronization**: Creation timestamps MUST be consistent across databases
- **Hash verification**: Content hashes MUST match the actual content in both databases

## 11. Embedding Generation Specifications (Cohere Free Tier)

### Embedding Model Configuration
- **Model identifier**: Cohere embed-multilingual-v2.0 (or equivalent free tier model)
- **Vector dimensions**: 1024-dimensional vectors as required by system specifications
- **Input format**: Plain text chunks within API token limits
- **Output format**: Normalized 1024-dimensional floating-point vector array

### API Call Structure
- **Endpoint**: Cohere embedding API endpoint
- **Method**: POST request with JSON payload
- **Authentication**: Bearer token authentication using COHERE_API_KEY
- **Content type**: application/json

### Batching Strategy
- **Batch size**: Up to 96 text chunks per API request (within free tier limits)
- **Batch grouping**: Group chunks by document or processing session for efficiency
- **Batch optimization**: Optimize batch composition to maximize API efficiency while respecting limits

### Rate Limiting Handling
- **Rate limit detection**: Monitor API responses for rate limit errors
- **Exponential backoff**: Implement exponential backoff for rate limit scenarios
- **Retry logic**: Retry failed requests with appropriate delays
- **Queue management**: Queue requests during rate limit periods

### Retry Logic
- **Retry conditions**: Network errors, API errors, rate limit responses
- **Maximum retries**: 3 attempts with exponential backoff (1s, 2s, 4s)
- **Failure handling**: Log failed embeddings and continue with remaining chunks
- **Fallback strategy**: Use zero vectors for failed embeddings if required (with appropriate flagging)

### Error Handling
- **API error responses**: Handle specific Cohere API error codes appropriately
- **Content validation**: Validate content before API submission to avoid content-related errors
- **Token limit enforcement**: Ensure chunks do not exceed API token limits
- **Fallback embeddings**: Generate appropriate fallback values for failed requests

## 12. Vector Packaging Specifications

### Qdrant Point Structure
- **Point ID**: Use the chunk_id as the Qdrant point ID to ensure alignment with Neon
- **Vector data**: 1024-dimensional array from Cohere embedding API
- **Payload structure**: JSON object containing all required metadata fields
- **Payload validation**: Validate payload structure before upsert to ensure schema compliance

### Neon Record Structure
- **Primary key**: Use the same chunk_id as the primary key for alignment with Qdrant
- **Field mapping**: Map metadata fields according to Neon schema requirements
- **Data type compliance**: Ensure all data types match Neon schema specifications
- **Foreign key relationships**: Establish proper relationships with other tables

### Alignment Requirements
- **ID consistency**: Chunk IDs MUST be identical in both Qdrant and Neon
- **Metadata synchronization**: All metadata fields MUST match between databases
- **Temporal alignment**: Creation and update timestamps SHOULD be synchronized
- **Content verification**: Text content references MUST be consistent between systems

### Packaging Validation
- **Schema validation**: Validate all packaging against database schemas before upsert
- **Integrity checks**: Verify content hashes and other integrity measures
- **Cross-reference validation**: Ensure all cross-references are valid and consistent
- **Dimensionality verification**: Verify vector dimensions match expected values

## 13. Database Connectivity Specifications

### Qdrant Cloud Connection Rules

#### Connection Parameters
- **URL**: QDRANT_HOST environment variable specifying the Qdrant Cloud endpoint
- **API key**: QDRANT_API_KEY environment variable for authentication
- **Collection name**: "book_embeddings" as specified in system requirements
- **Timeout settings**: 30-second timeout for connection and request operations

#### Index Configuration
- **Vector dimensions**: Configure for 1024-dimensional vectors
- **Distance metric**: Cosine distance for similarity search
- **Index type**: Auto-index on metadata fields for efficient filtering
- **Payload schema**: Configure to accept all required metadata fields

#### Connection Contract
- **Schema immutability**: Subsystem MUST NOT modify Qdrant collection schema
- **Index management**: Subsystem MUST NOT modify index configurations
- **Collection management**: Subsystem MUST NOT create or delete collections
- **Error handling**: Fail fast if Qdrant schema does not match expectations

### Neon Postgres Connection Rules

#### Connection Parameters
- **Connection string**: NEON_DATABASE_URL environment variable
- **Connection pooling**: Use connection pooling with appropriate limits
- **SSL configuration**: Enforce SSL connections for security
- **Timeout settings**: 30-second timeout for connection and query operations

#### Table and Field Mappings
- **Table names**: Use schema-compliant table names (chunks, logs, chat_history)
- **Field mappings**: Map metadata fields according to Neon schema specifications
- **Data types**: Use appropriate PostgreSQL data types for each field
- **Index specifications**: Ensure proper indexing for query performance

#### Connection Contract
- **Schema immutability**: Subsystem MUST NOT modify Neon table schemas
- **Migration exclusion**: Subsystem MUST NOT perform database migrations
- **Permission limitations**: Use only granted permissions for read/write operations
- **Error handling**: Fail fast if Neon schema does not match expectations

## 14. Vector Upsert Pipeline Specifications

### End-to-End Data Flow
1. **Receive preprocessed chunks**: Accept chunks from preprocessing stage with validated content and metadata
2. **Generate embeddings**: Request vector embeddings from Cohere API for each chunk
3. **Prepare Qdrant vectors**: Package embeddings with metadata for Qdrant storage
4. **Upsert into Qdrant**: Store vectors in Qdrant with proper error handling
5. **Insert metadata into Neon**: Store relational metadata in Neon Postgres
6. **Return success response**: Confirm successful storage to requesting component

### Atomicity Rules
- **Qdrant failure handling**: If Qdrant upsert fails, abort Neon insert and return error
- **Neon failure handling**: If Neon insert fails, attempt to remove corresponding Qdrant vector (if possible) and return error
- **Partial failure management**: Handle scenarios where one database operation succeeds and the other fails
- **Consistency maintenance**: Maintain cross-database consistency even during partial failures

### Transaction Management
- **Two-phase commit simulation**: Implement application-level consistency checks
- **Rollback procedures**: Define procedures for rolling back operations when failures occur
- **State tracking**: Track the state of each chunk across both databases
- **Recovery procedures**: Implement recovery mechanisms for failed operations

### Error Recovery
- **Consistency verification**: Verify consistency between databases after operations
- **Repair procedures**: Implement procedures to repair inconsistencies when detected
- **Retry mechanisms**: Retry failed operations with appropriate backoff strategies
- **Logging**: Log all operations and errors for troubleshooting and recovery

## 15. Error Handling Specifications

### Cohere API Retry Rules
- **Retryable errors**: Network timeouts, rate limit errors, server errors (5xx)
- **Non-retryable errors**: Client errors (4xx), invalid content, authentication failures
- **Retry strategy**: Exponential backoff with jitter (1s, 2s, 4s with random variation)
- **Maximum attempts**: 3 retry attempts before permanent failure

### Database Retry Rules
- **Qdrant retry conditions**: Connection timeouts, server errors, temporary unavailability
- **Neon retry conditions**: Connection timeouts, deadlocks, temporary unavailability
- **Retry strategy**: Linear backoff (1s, 2s, 3s) with maximum 3 attempts
- **Circuit breaker**: Implement circuit breaker pattern for persistent failures

### Malformed Input Handling
- **Input validation**: Validate all inputs before processing with detailed error messages
- **Content sanitization**: Sanitize content to remove potentially harmful elements
- **Format validation**: Validate text format and encoding before processing
- **Error reporting**: Provide detailed error information for debugging

### Corrupted Chunk Detection
- **Hash verification**: Verify content integrity using SHA-256 hashes
- **Size validation**: Validate chunk sizes against specified constraints
- **Content validation**: Check for binary content or other invalid data
- **Error isolation**: Isolate and report corrupted chunks without affecting others

### Metadata Mismatch Handling
- **ID validation**: Verify that chunk IDs match between systems
- **Schema validation**: Validate metadata against expected schema
- **Consistency checks**: Verify metadata consistency across databases
- **Error correction**: Attempt to correct minor mismatches when possible

### Logging and Reporting Format
- **Operation logging**: Log all major operations with timestamps and identifiers
- **Error categorization**: Categorize errors by type and severity
- **Performance metrics**: Track performance metrics for monitoring
- **Audit trail**: Maintain audit trail for compliance and debugging

## 16. Performance & Optimization Specifications

### Batching Embeddings
- **Batch size optimization**: Process up to 96 chunks per Cohere API request
- **Batch composition**: Group chunks by document or processing session for efficiency
- **Memory management**: Manage memory usage during batch processing to avoid overflow
- **Parallel processing**: Process multiple batches in parallel when possible

### Parallel Chunk Processing
- **Concurrency limits**: Limit concurrent processing to prevent resource exhaustion
- **Thread safety**: Ensure thread-safe operations during parallel processing
- **Resource allocation**: Allocate resources appropriately for parallel operations
- **Load balancing**: Distribute processing load across available resources

### Concurrency Limits
- **API rate limiting**: Respect Cohere API rate limits and implement appropriate queuing
- **Database connection limits**: Respect database connection pool limits
- **Memory usage limits**: Monitor and limit memory usage during processing
- **Processing throughput**: Optimize for maximum throughput within resource constraints

### Caching Strategies
- **Frequently accessed content**: Cache preprocessed content that is accessed repeatedly
- **Embedding caching**: Cache embeddings for content that is processed multiple times
- **Metadata caching**: Cache frequently accessed metadata to reduce database queries
- **Cache invalidation**: Implement appropriate cache invalidation strategies

### Redundant Embedding Prevention
- **Content hashing**: Use SHA-256 hashes to identify duplicate content
- **Hash comparison**: Compare hashes before generating embeddings to avoid redundancy
- **Cache lookup**: Check for existing embeddings before generating new ones
- **Update tracking**: Track content updates to determine when re-embedding is necessary

## 17. Security & Safety Specifications

### API Key Protection Rules
- **Environment storage**: Store API keys only in environment variables
- **Memory protection**: Avoid logging API keys or storing in memory longer than necessary
- **Access controls**: Restrict access to API keys to authorized components only
- **Rotation procedures**: Implement procedures for API key rotation

### Data Integrity Hashing
- **Content hashing**: Generate SHA-256 hashes for all processed content
- **Integrity verification**: Verify content integrity using hashes during operations
- **Hash storage**: Store hashes alongside content for verification
- **Validation procedures**: Validate hashes during retrieval and processing

### Content Sanitization Flow
- **Input sanitization**: Sanitize all input content to remove potentially harmful elements
- **Output sanitization**: Sanitize output content before storage or transmission
- **Format validation**: Validate content formats to prevent injection attacks
- **Content filtering**: Filter content to remove inappropriate or sensitive information

### User Data Separation
- **Content isolation**: Isolate user-provided content from book content
- **Access controls**: Implement appropriate access controls for different content types
- **Privacy protection**: Ensure user privacy is maintained for user-provided content
- **Data segregation**: Segregate user data from system data appropriately

### Safe Failure Modes
- **Graceful degradation**: Implement graceful degradation when components fail
- **Error containment**: Contain errors to prevent system-wide failures
- **Fallback mechanisms**: Provide fallback mechanisms for critical operations
- **Recovery procedures**: Implement recovery procedures for failed operations

## 18. Testing Specifications

### Preprocessing Correctness Tests
- **Normalization validation**: Test that text normalization is applied correctly
- **Character encoding**: Validate proper handling of different character encodings
- **Content sanitization**: Verify that content sanitization removes harmful elements
- **Format validation**: Test that different input formats are handled correctly

### Chunking Boundary Tests
- **Size compliance**: Verify that chunks comply with size constraints (800-1200 tokens)
- **Boundary preservation**: Test that semantic boundaries are preserved during chunking
- **Overlap correctness**: Validate that overlap is applied correctly and doesn't create duplicates
- **Edge cases**: Test chunking of very small or very large inputs

### Overlap Correctness Tests
- **Overlap size**: Verify that overlap sizes match specifications
- **Context preservation**: Test that context is preserved across chunk boundaries
- **Duplicate prevention**: Validate that overlap doesn't result in duplicate embeddings
- **Boundary handling**: Test overlap at document boundaries and special cases

### Metadata Consistency Tests
- **Cross-database alignment**: Verify that metadata is consistent between Qdrant and Neon
- **ID matching**: Test that chunk IDs match between databases
- **Field mapping**: Validate that all metadata fields are correctly mapped
- **Integrity verification**: Test that content hashes are correctly generated and verified

### Embedding Generation Validity Tests
- **Vector dimensions**: Verify that all embeddings have 1024 dimensions
- **API compliance**: Test that API calls comply with Cohere requirements
- **Batch processing**: Validate batch processing functionality and limits
- **Error handling**: Test embedding generation error scenarios

### Qdrant and Neon Integration Tests
- **Upsert operations**: Test the complete upsert pipeline across both databases
- **Atomicity**: Validate atomicity rules and failure handling
- **Consistency**: Test cross-database consistency maintenance
- **Performance**: Measure performance under various load conditions

### Load Tests for Large Embeddings
- **Throughput testing**: Test processing throughput for large document sets
- **Memory usage**: Monitor memory usage during large processing operations
- **API utilization**: Test API usage efficiency and rate limiting compliance
- **System limits**: Test behavior at system resource limits

## 19. Completion Criteria

### Functional Completion
- [ ] All text preprocessing operations execute correctly according to specifications
- [ ] Chunking operations produce chunks within 800-1200 token range consistently
- [ ] Overlap handling maintains context without creating duplicate embeddings
- [ ] Metadata generation produces consistent data for both Qdrant and Neon
- [ ] Cohere embedding API integration operates within free tier limits
- [ ] Vector packaging aligns correctly between Qdrant and Neon databases
- [ ] Database connectivity operates reliably with both Qdrant and Neon
- [ ] Upsert pipeline maintains atomicity and consistency across databases

### Quality Completion
- [ ] All error handling scenarios are properly managed according to specifications
- [ ] Performance optimization strategies are implemented and effective
- [ ] Security and safety measures are in place and validated
- [ ] All testing specifications are satisfied with acceptable pass rates
- [ ] Cross-database consistency is maintained under all operational conditions
- [ ] API rate limits and resource constraints are properly respected

### Integration Completion
- [ ] The subsystem integrates properly with FastAPI Backend for receiving content
- [ ] The subsystem coordinates correctly with Database Subsystem for storage
- [ ] The subsystem provides appropriate feedback to requesting components
- [ ] All interface contracts with other subsystems are satisfied
- [ ] The subsystem operates within constitutional constraints and boundaries

### Operational Completion
- [ ] The subsystem operates reliably under expected load conditions
- [ ] Performance metrics meet system requirements (<1.5s response time)
- [ ] Resource utilization stays within acceptable bounds
- [ ] Logging and monitoring capabilities are operational
- [ ] The subsystem produces output that meets Intelligence Layer requirements

## Outcome

- ✅ Impact: Created comprehensive specification for Embeddings & Chunking Pipeline subsystem with all 19 required sections
- 🧪 Tests: null
- 📁 Files: Created Chatbot_specs/embeddings_chunking_specs/specification.md
- 🔁 Next prompts: Ready for planning phase (/sp.plan)
- 🧠 Reflection: Successfully defined detailed technical specifications implementing constitutional responsibilities

## Evaluation notes (flywheel)

- Failure modes observed: null
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): null