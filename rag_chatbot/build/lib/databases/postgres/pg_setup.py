"""
Database Schema Setup Module
This module provides functions to initialize the database schema by executing
the SQL schema file against the connected PostgreSQL database.
"""
import os
from pathlib import Path
from typing import List
from databases.postgres.pg_client import PostgresClient
from rag_core.utils.logger import rag_logger


async def execute_schema_file(client: PostgresClient, schema_file_path: str) -> bool:
    """
    Execute a SQL schema file against the database.

    Args:
        client: PostgresClient instance
        schema_file_path: Path to the SQL schema file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the schema file
        with open(schema_file_path, 'r', encoding='utf-8') as file:
            schema_sql = file.read()

        # For this schema file, execute it as a whole since it contains complex structures like functions
        # that shouldn't be split on semicolons
        async with client.get_connection() as conn:
            await conn.execute(schema_sql)
            rag_logger.info(f"Executed schema file: {schema_file_path}")

        rag_logger.info(f"Successfully executed schema from {schema_file_path}")
        return True

    except Exception as e:
        rag_logger.error(f"Error executing schema file {schema_file_path}: {str(e)}")
        return False


async def ensure_database_schema(client: PostgresClient) -> bool:
    """
    Ensure that the database schema exists by checking for key tables
    and creating them if they don't exist.

    Args:
        client: PostgresClient instance

    Returns:
        True if schema is properly set up, False otherwise
    """
    try:
        # Define the key tables that should exist
        required_tables = [
            'chunks',
            'logs',
            'chat_history',
            'users',
            'audit_logs',
            'settings',
            'embedding_jobs'
        ]

        missing_tables = []

        # Check which tables are missing
        for table in required_tables:
            exists = await client.table_exists(table)
            if not exists:
                missing_tables.append(table)
                rag_logger.info(f"Missing table: {table}")

        # If there are missing tables, execute the schema file
        if missing_tables:
            rag_logger.info(f"Found {len(missing_tables)} missing tables, executing schema...")

            # Get the path to the schema file
            schema_file = os.path.join(
                Path(__file__).parent,
                'pg_schema.sql'
            )

            if not os.path.exists(schema_file):
                rag_logger.error(f"Schema file not found: {schema_file}")
                return False

            # Execute the schema file
            success = await execute_schema_file(client, schema_file)

            if success:
                rag_logger.info("Database schema setup completed successfully")

                # Verify that all required tables now exist
                for table in required_tables:
                    exists = await client.table_exists(table)
                    if not exists:
                        rag_logger.error(f"Table {table} still doesn't exist after schema execution")
                        return False

                rag_logger.info("All required tables verified after schema execution")
                return True
            else:
                rag_logger.error("Failed to execute schema file")
                return False
        else:
            rag_logger.info("All required tables already exist")
            return True

    except Exception as e:
        rag_logger.error(f"Error in ensure_database_schema: {str(e)}")
        return False


async def initialize_database_schema() -> bool:
    """
    Initialize the database schema by connecting to the database
    and ensuring all required tables exist.

    Returns:
        True if successful, False otherwise
    """
    client = PostgresClient()

    try:
        # Connect to the database
        await client.connect()
        rag_logger.info("Connected to PostgreSQL database")

        # Ensure the schema exists
        success = await ensure_database_schema(client)

        return success

    except Exception as e:
        rag_logger.error(f"Error initializing database schema: {str(e)}")
        return False
    finally:
        try:
            await client.disconnect()
            rag_logger.info("Disconnected from PostgreSQL database")
        except:
            pass