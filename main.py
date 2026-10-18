"""DataMind CLI: Ask questions in natural language and get answers from Snowflake."""

import re
import sys
from tabulate import tabulate
from db import SnowflakeClient
from llm import GemmaClient

BLOCKED_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXECUTE", "CALL", "REPLACE",
}


def is_safe_query(sql: str) -> tuple[bool, str]:
    """Ensure generated SQL is read-only and free of destructive statements."""
    cleaned = sql.strip()
    upper = cleaned.upper()

    # Must start with SELECT or WITH
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "Query must start with SELECT or WITH."

    # Reject dangerous SQL manipulation keywords
    tokens = set(re.findall(r"\b[A-Z_]+\b", upper))
    found_blocked = tokens.intersection(BLOCKED_KEYWORDS)
    if found_blocked:
        return False, f"Query contains blocked non-read-only operation(s): {', '.join(found_blocked)}"

    # Disallow multiple semicolons / command chaining
    clean_no_strings = re.sub(r"'[^']*'", "", cleaned)
    if clean_no_strings.count(";") > 1:
        return False, "Multi-statement execution is disallowed."

    return True, ""


def run_pipeline(question: str, db: SnowflakeClient, llm: GemmaClient) -> None:
    print(f"\nQuestion: {question}\n")

    # Step 1: Introspect detailed database schema for Gemma
    print("[1/3] Introspecting database schema context...")
    try:
        schema_context = db.format_schema_context()
        if schema_context:
            print(f"      Loaded context for active tables in {db.database}.{db.schema}")
        else:
            print("      (No tables found in active schema)")
    except Exception as e:
        schema_context = ""
        print(f"      (Schema introspection skipped: {e})")

    # Step 2: Generate SQL via Google Gemma
    print("[2/3] Generating SQL query via Google Gemma...")
    sql = llm.generate_sql(question, schema_context=schema_context)
    print("\nGenerated SQL:")
    print("-" * 60)
    print(sql)
    print("-" * 60 + "\n")

    # Step 3: Safety check & live Snowflake execution
    safe, reason = is_safe_query(sql)
    if not safe:
        print(f"Error: Query blocked by safety guard -> {reason}")
        return

    print("[3/3] Executing query on Snowflake...")
    result = db.execute_query(sql)

    print("\nExecution Summary:")
    print(f"  Snowflake Query ID: {result['query_id']}")
    print(f"  Duration:           {result['duration_ms']} ms")
    print(f"  Rows Returned:      {result['row_count']}\n")

    if result["rows"]:
        print("Query Results:")
        print(tabulate(result["rows"], headers=result["columns"], tablefmt="psql"))
    else:
        print("Query executed successfully, but returned 0 rows.")


def interactive_loop() -> None:
    """Run interactive question-answer REPL session."""
    print("\n" + "=" * 60)
    print("DataMind — Interactive Snowflake Query Assistant")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'schema' to inspect discovered database tables.")
    print("=" * 60 + "\n")

    db = SnowflakeClient()
    llm = GemmaClient()

    try:
        while True:
            try:
                question = input("\nDataMind> ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if question.lower() == "schema":
                print("\nDiscovered Schema:")
                print(db.format_schema_context() or "No tables discovered.")
                continue

            try:
                run_pipeline(question, db, llm)
            except Exception as err:
                print(f"Query error: {err}")
    finally:
        db.close()


def main() -> None:
    args = sys.argv[1:]

    if not args or "--interactive" in args or "-i" in args:
        interactive_loop()
        return

    if "--schema" in args:
        db = SnowflakeClient()
        try:
            print(db.format_schema_context() or "No tables discovered.")
        finally:
            db.close()
        return

    question = " ".join(args).strip()
    db = SnowflakeClient()
    llm = GemmaClient()
    try:
        run_pipeline(question, db, llm)
    finally:
        db.close()


if __name__ == "__main__":
    main()
