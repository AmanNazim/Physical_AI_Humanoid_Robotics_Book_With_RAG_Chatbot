from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from rag_core.utils.logger import rag_logger
from rag_core.utils.timing import timing_decorator
from .pg_models import PostgresChunk, PostgresChatHistory, PostgresLog


class PostgresUtils:
    """
    Utility functions for PostgreSQL operations including row-to-model mapping and pagination.
    """

    @staticmethod
    @timing_decorator
    def map_row_to_chunk(row: Dict[str, Any]) -> PostgresChunk:
        """
        Convert a database row to a PostgresChunk model.
        """
        try:
            chunk = PostgresChunk(
                chunk_id=str(row['chunk_id']),
                document_reference=row['document_reference'],
                page_reference=row.get('page_reference'),
                section_title=row.get('section_title'),
                chunk_text=row['chunk_text'],
                embedding_id=str(row['embedding_id']),
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                processing_version=row['processing_version'],
                metadata=row.get('metadata', {})
            )
            return chunk
        except Exception as e:
            rag_logger.error(f"Error mapping row to chunk: {str(e)}")
            raise

    @staticmethod
    @timing_decorator
    def map_row_to_chat_history(row: Dict[str, Any]) -> PostgresChatHistory:
        """
        Convert a database row to a PostgresChatHistory model.
        """
        try:
            chat_history = PostgresChatHistory(
                chat_id=str(row['chat_id']),
                user_id=str(row['user_id']) if row['user_id'] else None,
                query=row['query'],
                response=row['response'],
                source_chunks=row.get('source_chunks', []),
                timestamp=row['timestamp']
            )
            return chat_history
        except Exception as e:
            rag_logger.error(f"Error mapping row to chat history: {str(e)}")
            raise

    @staticmethod
    @timing_decorator
    def map_row_to_log(row: Dict[str, Any]) -> PostgresLog:
        """
        Convert a database row to a PostgresLog model.
        """
        try:
            log = PostgresLog(
                log_id=str(row['log_id']),
                user_query=row['user_query'],
                retrieved_chunks=row['retrieved_chunks'],
                response=row['response'],
                timestamp=row['timestamp'],
                retrieval_mode=row['retrieval_mode']
            )
            return log
        except Exception as e:
            rag_logger.error(f"Error mapping row to log: {str(e)}")
            raise

    @staticmethod
    @timing_decorator
    def map_rows_to_chunks(rows: List[Dict[str, Any]]) -> List[PostgresChunk]:
        """
        Convert multiple database rows to a list of PostgresChunk models.
        """
        return [PostgresUtils.map_row_to_chunk(row) for row in rows]

    @staticmethod
    @timing_decorator
    def map_rows_to_chat_history(rows: List[Dict[str, Any]]) -> List[PostgresChatHistory]:
        """
        Convert multiple database rows to a list of PostgresChatHistory models.
        """
        return [PostgresUtils.map_row_to_chat_history(row) for row in rows]

    @staticmethod
    @timing_decorator
    def map_rows_to_logs(rows: List[Dict[str, Any]]) -> List[PostgresLog]:
        """
        Convert multiple database rows to a list of PostgresLog models.
        """
        return [PostgresUtils.map_row_to_log(row) for row in rows]

    @staticmethod
    @timing_decorator
    def format_datetime_for_db(dt: Optional[datetime] = None) -> str:
        """
        Format datetime for database storage.
        """
        if dt is None:
            dt = datetime.utcnow()
        return dt.isoformat()

    @staticmethod
    @timing_decorator
    def generate_uuid() -> str:
        """
        Generate a UUID string for database records.
        """
        return str(uuid.uuid4())

    @staticmethod
    @timing_decorator
    def validate_uuid(uuid_string: str) -> bool:
        """
        Validate that a string is a proper UUID.
        """
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False

    @staticmethod
    @timing_decorator
    def paginate_results(
        results: List[Any],
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """
        Paginate results and return metadata about pagination.
        """
        total = len(results)
        offset = (page - 1) * page_size
        end_index = offset + page_size

        paginated_data = results[offset:end_index]
        has_next = end_index < total
        has_prev = offset > 0
        total_pages = (total + page_size - 1) // page_size  # Ceiling division

        return {
            "data": paginated_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
                "next_page": page + 1 if has_next else None,
                "prev_page": page - 1 if has_prev else None
            }
        }

    @staticmethod
    @timing_decorator
    def build_where_clause(
        filters: Optional[Dict[str, Any]] = None,
        table_alias: Optional[str] = None
    ) -> tuple[str, List[Any]]:
        """
        Build a WHERE clause from a dictionary of filters.
        Returns the WHERE clause string and a list of parameters.
        """
        if not filters:
            return "", []

        conditions = []
        params = []
        param_index = 1

        table_prefix = f"{table_alias}." if table_alias else ""

        for field, value in filters.items():
            if value is None:
                conditions.append(f"{table_prefix}{field} IS NULL")
            elif isinstance(value, list):
                # Handle IN clause
                placeholders = ",".join([f"${param_index + i}" for i in range(len(value))])
                conditions.append(f"{table_prefix}{field} = ANY(ARRAY[{placeholders}])")
                params.extend(value)
                param_index += len(value)
            elif isinstance(value, dict) and "op" in value and "value" in value:
                # Handle operator-based conditions (e.g., {"op": ">", "value": 5})
                op = value["op"]
                val = value["value"]
                conditions.append(f"{table_prefix}{field} {op} ${param_index}")
                params.append(val)
                param_index += 1
            else:
                # Handle simple equality
                conditions.append(f"{table_prefix}{field} = ${param_index}")
                params.append(value)
                param_index += 1

        where_clause = "WHERE " + " AND ".join(conditions)
        return where_clause, params

    @staticmethod
    @timing_decorator
    def validate_chunk_data(
        document_reference: str,
        chunk_text: str,
        processing_version: str
    ) -> bool:
        """
        Validate chunk data before database insertion.
        """
        if not document_reference or not isinstance(document_reference, str):
            rag_logger.error("Invalid document_reference provided")
            return False

        if not chunk_text or not isinstance(chunk_text, str):
            rag_logger.error("Invalid chunk_text provided")
            return False

        if not processing_version or not isinstance(processing_version, str):
            rag_logger.error("Invalid processing_version provided")
            return False

        if len(chunk_text) > 100000:  # 100KB limit
            rag_logger.warning("Chunk text exceeds recommended size limit")

        return True

    @staticmethod
    @timing_decorator
    def validate_chat_history_data(
        query: str,
        response: str
    ) -> bool:
        """
        Validate chat history data before database insertion.
        """
        if not query or not isinstance(query, str):
            rag_logger.error("Invalid query provided")
            return False

        if not response or not isinstance(response, str):
            rag_logger.error("Invalid response provided")
            return False

        if len(query) > 10000:  # 10KB limit
            rag_logger.warning("Query exceeds recommended size limit")

        if len(response) > 50000:  # 50KB limit
            rag_logger.warning("Response exceeds recommended size limit")

        return True

    @staticmethod
    @timing_decorator
    def format_metadata_for_db(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format metadata for database storage, ensuring it's compatible with JSONB.
        """
        if metadata is None:
            return {}

        # Ensure all keys are strings
        formatted_metadata = {}
        for key, value in metadata.items():
            str_key = str(key)
            # Convert non-JSON-serializable values to strings
            if isinstance(value, (datetime, uuid.UUID)):
                formatted_metadata[str_key] = str(value)
            elif value is None or isinstance(value, (str, int, float, bool, list, dict)):
                formatted_metadata[str_key] = value
            else:
                formatted_metadata[str_key] = str(value)

        return formatted_metadata

    @staticmethod
    @timing_decorator
    def build_full_text_search_query(
        search_term: str,
        table_name: str,
        text_columns: List[str]
    ) -> str:
        """
        Build a full-text search query for PostgreSQL.
        """
        if not text_columns:
            raise ValueError("At least one text column must be specified")

        # Create the search condition for each column
        search_conditions = []
        for col in text_columns:
            search_conditions.append(f"LOWER({col}) LIKE LOWER('%' || $1 || '%')")

        search_clause = " OR ".join(search_conditions)
        query = f"SELECT * FROM {table_name} WHERE {search_clause};"
        return query