import asyncio
import asyncpg
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from shared.config import settings
from rag_core.utils.logger import rag_logger
from rag_core.utils.timing import timing_decorator


class PostgresClient:
    """
    Async PostgreSQL client with connection pooling and retry logic.
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._connection_string = settings.neon_settings.database_url
        self._pool_size = settings.neon_settings.pool_size
        self._pool_timeout = settings.neon_settings.pool_timeout

    async def connect(self):
        """
        Initialize the PostgreSQL connection pool with retry logic.
        """
        max_retries = 5
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                self._pool = await asyncpg.create_pool(
                    dsn=self._connection_string,
                    min_size=2,
                    max_size=self._pool_size,
                    command_timeout=self._pool_timeout,
                    server_settings={
                        "application_name": "rag_chatbot",
                        "idle_in_transaction_session_timeout": "30000",  # 30 seconds
                    }
                )
                rag_logger.info("PostgreSQL pool initialized successfully")
                return
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    rag_logger.error(f"Failed to connect to PostgreSQL after {max_retries} attempts: {str(e)}")
                    raise
                else:
                    rag_logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed, retrying in {retry_delay}s: {str(e)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

    async def disconnect(self):
        """
        Close the PostgreSQL connection pool.
        """
        if self._pool:
            await self._pool.close()
            rag_logger.info("PostgreSQL pool closed")

    @property
    def pool(self) -> asyncpg.Pool:
        """
        Get the PostgreSQL connection pool.
        """
        if not self._pool:
            raise RuntimeError("PostgreSQL pool not initialized. Call connect() first.")
        return self._pool

    @asynccontextmanager
    async def get_connection(self):
        """
        Get a connection from the pool using async context manager.
        """
        if not self.pool:
            raise RuntimeError("PostgreSQL pool not initialized.")

        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)

    @timing_decorator
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as list of dictionaries.
        """
        async with self.get_connection() as conn:
            try:
                rows = await conn.fetch(query, *args)
                # Convert asyncpg records to dictionaries
                result = [dict(row) for row in rows]
                rag_logger.info(f"Executed SELECT query, returned {len(result)} rows")
                return result
            except Exception as e:
                rag_logger.error(f"Error executing SELECT query: {str(e)}")
                raise

    @timing_decorator
    async def execute_command(self, command: str, *args) -> int:
        """
        Execute an INSERT/UPDATE/DELETE command and return affected row count.
        """
        async with self.get_connection() as conn:
            try:
                result = await conn.fetchval(command, *args)
                # fetchval returns the result of the command (e.g., count)
                if result is None:
                    # For commands that don't return a value, get the status message
                    result = await conn.execute(command, *args)
                    # Extract the number of affected rows from the result
                    if "UPDATE" in result.upper():
                        affected = int(result.split()[-1])
                    elif "DELETE" in result.upper():
                        affected = int(result.split()[-1])
                    elif "INSERT" in result.upper():
                        affected = 1  # For simple insert
                    else:
                        affected = 0
                else:
                    # If fetchval returned a value, it's likely the count
                    affected = result
                rag_logger.info(f"Executed command, affected {affected} rows")
                return affected
            except Exception as e:
                rag_logger.error(f"Error executing command: {str(e)}")
                raise

    @timing_decorator
    async def execute_many(self, command: str, args_list: List[tuple]) -> int:
        """
        Execute a command multiple times with different arguments.
        """
        async with self.get_connection() as conn:
            try:
                result = await conn.executemany(command, args_list)
                affected_count = len(args_list)  # For batch operations
                rag_logger.info(f"Executed command {len(args_list)} times")
                return affected_count
            except Exception as e:
                rag_logger.error(f"Error executing command multiple times: {str(e)}")
                raise

    @timing_decorator
    async def health_check(self) -> bool:
        """
        Perform a health check on the PostgreSQL connection.
        """
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            rag_logger.error(f"PostgreSQL health check failed: {str(e)}")
            return False

    @timing_decorator
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test the PostgreSQL connection and return detailed information.
        """
        try:
            async with self.get_connection() as conn:
                # Get basic connection info
                version = await conn.fetchval("SELECT version()")
                active_connections = await conn.fetchval("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")

                return {
                    "connected": True,
                    "version": version,
                    "active_connections": active_connections,
                    "pool_status": {
                        "min_size": self.pool.get_min_size(),
                        "max_size": self.pool.get_max_size(),
                        "current_size": self.pool.get_size(),
                        "idle_size": self.pool.get_idle_size()
                    }
                }
        except Exception as e:
            rag_logger.error(f"PostgreSQL connection test failed: {str(e)}")
            return {
                "connected": False,
                "error": str(e)
            }

    @timing_decorator
    async def create_table_if_not_exists(self, table_name: str, schema_sql: str) -> bool:
        """
        Create a table if it doesn't exist using provided schema.
        """
        try:
            # Check if table exists first
            exists_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = $1
                );
            """
            async with self.get_connection() as conn:
                table_exists = await conn.fetchval(exists_query, table_name)

                if not table_exists:
                    await conn.execute(schema_sql)
                    rag_logger.info(f"Created table: {table_name}")
                    return True
                else:
                    rag_logger.info(f"Table already exists: {table_name}")
                    return False
        except Exception as e:
            rag_logger.error(f"Error creating table {table_name}: {str(e)}")
            raise

    @timing_decorator
    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        """
        try:
            exists_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = $1
                );
            """
            async with self.get_connection() as conn:
                result = await conn.fetchval(exists_query, table_name)
                return result
        except Exception as e:
            rag_logger.error(f"Error checking if table {table_name} exists: {str(e)}")
            return False

    @timing_decorator
    async def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table.
        """
        try:
            query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position;
            """
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, table_name)
                return [dict(row) for row in rows]
        except Exception as e:
            rag_logger.error(f"Error getting columns for table {table_name}: {str(e)}")
            return []