from typing import List, Dict, Any, Optional
import uuid


class PostgresQueries:
    """
    Contains prepared SQL statements for all PostgreSQL operations.
    """

    # Chunks table queries
    INSERT_CHUNK = """
    INSERT INTO chunks (
        chunk_id, document_reference, page_reference, section_title,
        chunk_text, embedding_id, processing_version, metadata
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    ON CONFLICT (chunk_id) DO UPDATE SET
        document_reference = EXCLUDED.document_reference,
        page_reference = EXCLUDED.page_reference,
        section_title = EXCLUDED.section_title,
        chunk_text = EXCLUDED.chunk_text,
        embedding_id = EXCLUDED.embedding_id,
        processing_version = EXCLUDED.processing_version,
        metadata = EXCLUDED.metadata,
        updated_at = NOW();
    """

    INSERT_CHUNK_MANY = """
    INSERT INTO chunks (
        chunk_id, document_reference, page_reference, section_title,
        chunk_text, embedding_id, processing_version, metadata
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    ON CONFLICT (chunk_id) DO UPDATE SET
        document_reference = EXCLUDED.document_reference,
        page_reference = EXCLUDED.page_reference,
        section_title = EXCLUDED.section_title,
        chunk_text = EXCLUDED.chunk_text,
        embedding_id = EXCLUDED.embedding_id,
        processing_version = EXCLUDED.processing_version,
        metadata = EXCLUDED.metadata,
        updated_at = NOW();
    """

    SELECT_CHUNK_BY_ID = """
    SELECT * FROM chunks WHERE chunk_id = $1;
    """

    SELECT_CHUNKS_BY_DOCUMENT = """
    SELECT * FROM chunks WHERE document_reference = $1 ORDER BY created_at;
    """

    SELECT_ALL_CHUNKS = """
    SELECT * FROM chunks ORDER BY created_at LIMIT $1 OFFSET $2;
    """

    UPDATE_CHUNK = """
    UPDATE chunks SET
        document_reference = $2,
        page_reference = $3,
        section_title = $4,
        chunk_text = $5,
        embedding_id = $6,
        processing_version = $7,
        metadata = $8,
        updated_at = NOW()
    WHERE chunk_id = $1;
    """

    DELETE_CHUNK = """
    DELETE FROM chunks WHERE chunk_id = $1;
    """

    DELETE_CHUNKS_BY_DOCUMENT = """
    DELETE FROM chunks WHERE document_reference = $1;
    """

    # Chat history table queries
    INSERT_CHAT_MESSAGE = """
    INSERT INTO chat_history (
        chat_id, user_id, query, response, source_chunks
    ) VALUES ($1, $2, $3, $4, $5);
    """

    INSERT_CHAT_MESSAGE_MANY = """
    INSERT INTO chat_history (
        chat_id, user_id, query, response, source_chunks
    ) VALUES ($1, $2, $3, $4, $5);
    """

    SELECT_CHAT_HISTORY_BY_USER = """
    SELECT * FROM chat_history
    WHERE user_id = $1
    ORDER BY timestamp DESC
    LIMIT $2 OFFSET $3;
    """

    SELECT_CHAT_HISTORY_ALL = """
    SELECT * FROM chat_history
    ORDER BY timestamp DESC
    LIMIT $1 OFFSET $2;
    """

    SELECT_CHAT_HISTORY_BY_ID = """
    SELECT * FROM chat_history WHERE chat_id = $1;
    """

    DELETE_CHAT_HISTORY_BY_USER = """
    DELETE FROM chat_history WHERE user_id = $1;
    """

    DELETE_CHAT_HISTORY_BY_ID = """
    DELETE FROM chat_history WHERE chat_id = $1;
    """

    # Logs table queries
    INSERT_LOG = """
    INSERT INTO logs (
        log_id, user_query, retrieved_chunks, response, retrieval_mode
    ) VALUES ($1, $2, $3, $4, $5);
    """

    SELECT_LOGS_BY_DATE_RANGE = """
    SELECT * FROM logs
    WHERE timestamp BETWEEN $1 AND $2
    ORDER BY timestamp DESC;
    """

    SELECT_LOGS_BY_RETRIEVAL_MODE = """
    SELECT * FROM logs
    WHERE retrieval_mode = $1
    ORDER BY timestamp DESC
    LIMIT $2 OFFSET $3;
    """

    SELECT_LOG_BY_ID = """
    SELECT * FROM logs WHERE log_id = $1;
    """

    # Users table queries
    INSERT_USER = """
    INSERT INTO users (
        user_id, email, profile_metadata, preferences
    ) VALUES ($1, $2, $3, $4)
    ON CONFLICT (user_id) DO UPDATE SET
        email = EXCLUDED.email,
        profile_metadata = EXCLUDED.profile_metadata,
        preferences = EXCLUDED.preferences,
        is_active = TRUE;
    """

    SELECT_USER_BY_ID = """
    SELECT * FROM users WHERE user_id = $1;
    """

    SELECT_USER_BY_EMAIL = """
    SELECT * FROM users WHERE email = $1;
    """

    UPDATE_USER_PREFERENCES = """
    UPDATE users SET
        preferences = $2,
        updated_at = NOW()
    WHERE user_id = $1;
    """

    UPDATE_USER_LAST_LOGIN = """
    UPDATE users SET
        last_login = NOW()
    WHERE user_id = $1;
    """

    DELETE_USER = """
    UPDATE users SET is_active = FALSE WHERE user_id = $1;
    """

    # Settings table queries
    INSERT_SETTING = """
    INSERT INTO settings (
        setting_id, setting_key, setting_value, setting_type, scope
    ) VALUES ($1, $2, $3, $4, $5)
    ON CONFLICT (setting_key) DO UPDATE SET
        setting_value = EXCLUDED.setting_value,
        setting_type = EXCLUDED.setting_type,
        scope = EXCLUDED.scope,
        updated_at = NOW();
    """

    SELECT_SETTING_BY_KEY = """
    SELECT * FROM settings WHERE setting_key = $1 AND scope = $2;
    """

    SELECT_SETTINGS_BY_SCOPE = """
    SELECT * FROM settings WHERE scope = $1;
    """

    UPDATE_SETTING = """
    UPDATE settings SET
        setting_value = $2,
        updated_at = NOW()
    WHERE setting_key = $1 AND scope = $2;
    """

    DELETE_SETTING = """
    DELETE FROM settings WHERE setting_key = $1 AND scope = $2;
    """

    # Audit logs table queries
    INSERT_AUDIT_LOG = """
    INSERT INTO audit_logs (
        log_id, operation_type, resource_type, resource_id, user_id, details, ip_address, user_agent
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
    """

    SELECT_AUDIT_LOGS_BY_USER = """
    SELECT * FROM audit_logs
    WHERE user_id = $1
    ORDER BY operation_timestamp DESC
    LIMIT $2 OFFSET $3;
    """

    SELECT_AUDIT_LOGS_BY_OPERATION = """
    SELECT * FROM audit_logs
    WHERE operation_type = $1
    ORDER BY operation_timestamp DESC
    LIMIT $2 OFFSET $3;
    """

    SELECT_AUDIT_LOGS_BY_DATE_RANGE = """
    SELECT * FROM audit_logs
    WHERE operation_timestamp BETWEEN $1 AND $2
    ORDER BY operation_timestamp DESC;
    """

    # Embedding jobs table queries
    INSERT_EMBEDDING_JOB = """
    INSERT INTO embedding_jobs (
        job_id, document_reference, chunk_count, priority
    ) VALUES ($1, $2, $3, $4);
    """

    SELECT_EMBEDDING_JOB_BY_ID = """
    SELECT * FROM embedding_jobs WHERE job_id = $1;
    """

    SELECT_EMBEDDING_JOBS_BY_STATUS = """
    SELECT * FROM embedding_jobs
    WHERE status = $1
    ORDER BY priority DESC, created_at;
    """

    SELECT_EMBEDDING_JOBS_BY_DOCUMENT = """
    SELECT * FROM embedding_jobs
    WHERE document_reference = $1
    ORDER BY created_at DESC;
    """

    UPDATE_EMBEDDING_JOB_STATUS = """
    UPDATE embedding_jobs SET
        status = $2,
        processed_count = $3,
        retry_count = $4,
        failure_reason = $5,
        completed_at = $6,
        updated_at = NOW()
    WHERE job_id = $1;
    """

    UPDATE_EMBEDDING_JOB_PROGRESS = """
    UPDATE embedding_jobs SET
        processed_count = $2,
        updated_at = NOW()
    WHERE job_id = $1;
    """

    DELETE_EMBEDDING_JOB = """
    DELETE FROM embedding_jobs WHERE job_id = $1;
    """

    @staticmethod
    def get_insert_chunk_params(
        chunk_id: str,
        document_reference: str,
        chunk_text: str,
        embedding_id: str,
        processing_version: str,
        page_reference: Optional[int] = None,
        section_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """Get parameters for insert chunk query"""
        return (
            uuid.UUID(chunk_id) if isinstance(chunk_id, str) else chunk_id,
            document_reference,
            page_reference,
            section_title,
            chunk_text,
            uuid.UUID(embedding_id) if isinstance(embedding_id, str) else embedding_id,
            processing_version,
            metadata or {}
        )

    @staticmethod
    def get_insert_chat_message_params(
        chat_id: str,
        query: str,
        response: str,
        user_id: Optional[str] = None,
        source_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:
        """Get parameters for insert chat message query"""
        return (
            uuid.UUID(chat_id) if isinstance(chat_id, str) else chat_id,
            uuid.UUID(user_id) if user_id and isinstance(user_id, str) else user_id,
            query,
            response,
            source_chunks or []
        )

    @staticmethod
    def get_insert_log_params(
        log_id: str,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        response: str,
        retrieval_mode: str
    ) -> tuple:
        """Get parameters for insert log query"""
        return (
            uuid.UUID(log_id) if isinstance(log_id, str) else log_id,
            user_query,
            retrieved_chunks,
            response,
            retrieval_mode
        )

    @staticmethod
    def get_insert_user_params(
        user_id: str,
        email: Optional[str] = None,
        profile_metadata: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> tuple:
        """Get parameters for insert user query"""
        return (
            uuid.UUID(user_id) if isinstance(user_id, str) else user_id,
            email,
            profile_metadata or {},
            preferences or {}
        )

    @staticmethod
    def get_insert_setting_params(
        setting_id: str,
        setting_key: str,
        setting_value: Dict[str, Any],
        setting_type: str,
        scope: str = "global"
    ) -> tuple:
        """Get parameters for insert setting query"""
        return (
            uuid.UUID(setting_id) if isinstance(setting_id, str) else setting_id,
            setting_key,
            setting_value,
            setting_type,
            scope
        )

    @staticmethod
    def get_pagination_offset(page: int, page_size: int) -> int:
        """Calculate offset for pagination"""
        return (page - 1) * page_size