"""Snowflake database connector for executing queries and fetching schema metadata."""

import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()


class SnowflakeClient:
    """Manages connection and query execution on Snowflake."""

    def __init__(self) -> None:
        self.account = os.getenv("SNOWFLAKE_ACCOUNT", "")
        self.user = os.getenv("SNOWFLAKE_USER", "")
        self.password = os.getenv("SNOWFLAKE_PASSWORD", "")
        self.warehouse = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        self.database = os.getenv("SNOWFLAKE_DATABASE", "DEMO_DB")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        self.role = os.getenv("SNOWFLAKE_ROLE", "PUBLIC")
        self._conn: Optional[snowflake.connector.SnowflakeConnection] = None

    def connect(self) -> snowflake.connector.SnowflakeConnection:
        """Establish or reuse an active Snowflake connection."""
        if self._conn is None or self._conn.is_closed():
            self._conn = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                role=self.role,
                client_session_keep_alive=False,
            )
        return self._conn

    def execute_query(self, sql: str, max_rows: int = 100) -> Dict[str, Any]:
        """Execute a read-only SQL query and return results with Snowflake query ID."""
        conn = self.connect()
        cursor = conn.cursor()
        start_time = time.perf_counter()

        try:
            cursor.execute(sql)
            query_id = cursor.sfqid or "unknown"
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return {
                "columns": columns,
                "rows": rows,
                "query_id": query_id,
                "row_count": len(rows),
                "duration_ms": duration_ms,
            }
        finally:
            cursor.close()

    def list_tables(self) -> List[str]:
        """Fetch list of available tables in the active schema."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("SHOW TABLES")
            return [row[1] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def describe_table(self, table_name: str) -> List[Dict[str, str]]:
        """Fetch column names and data types for a given table."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute(f"DESCRIBE TABLE {table_name}")
            return [{"column": row[0], "type": row[1]} for row in cursor.fetchall()]
        finally:
            cursor.close()

    def introspect_schema(self, refresh: bool = False) -> Dict[str, List[Dict[str, str]]]:
        """Discover all tables and columns in the active schema, caching results."""
        if hasattr(self, "_schema_cache") and self._schema_cache and not refresh:
            return self._schema_cache

        schema_data: Dict[str, List[Dict[str, str]]] = {}
        tables = self.list_tables()
        for tbl in tables:
            try:
                schema_data[tbl] = self.describe_table(tbl)
            except Exception:
                schema_data[tbl] = []

        self._schema_cache = schema_data
        return schema_data

    def format_schema_context(self, max_tables: int = 12) -> str:
        """Format introspected schema into concise context string for the LLM."""
        schema = self.introspect_schema()
        if not schema:
            return ""

        lines = [f"Database: {self.database} | Schema: {self.schema}"]
        for tbl, cols in list(schema.items())[:max_tables]:
            col_strs = [f"{c['column']} ({c['type']})" for c in cols[:15]]
            if len(cols) > 15:
                col_strs.append(f"... +{len(cols) - 15} more")
            lines.append(f"- Table `{tbl}`: {', '.join(col_strs)}")
        return "\n".join(lines)

    def close(self) -> None:
        """Close connection if open."""
        if self._conn and not self._conn.is_closed():
            self._conn.close()
            self._conn = None
