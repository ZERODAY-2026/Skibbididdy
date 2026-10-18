"""DataMind CLI prototype: Ask questions in natural language and get answers from Snowflake."""

import sys
from tabulate import tabulate
from db import SnowflakeClient
from llm import GemmaClient


def is_safe_query(sql: str) -> bool:
    """Ensure generated SQL is read-only."""
    cleaned = sql.strip().upper()
    return cleaned.startswith("SELECT") or cleaned.startswith("WITH")


def run_pipeline(question: str) -> None:
    print("\n" + "=" * 60)
    print("DataMind — Natural Language Query Assistant")
    print("=" * 60)
    print(f"Question: {question}\n")

    db = SnowflakeClient()
    llm = GemmaClient()

    try:
        # Step 1: Discover available tables for schema context
        print("[1/3] Inspecting database tables...")
        try:
            tables = db.list_tables()
            schema_context = f"Available tables in database: {', '.join(tables[:15])}"
        except Exception as e:
            schema_context = ""
            print(f"      (Table discovery skipped: {e})")

        # Step 2: Generate SQL with Google Gemma
        print("[2/3] Generating SQL query via Google Gemma...")
        sql = llm.generate_sql(question, schema_context=schema_context)
        print("\nGenerated SQL:")
        print("------------------------------------------------------------")
        print(sql)
        print("------------------------------------------------------------\n")

        # Step 3: Safety verification & execution
        if not is_safe_query(sql):
            print("Error: Generated query was blocked because it is not a read-only SELECT statement.")
            return

        print("[3/3] Executing query on Snowflake...")
        result = db.execute_query(sql)

        print(f"\nExecution Summary:")
        print(f"  Snowflake Query ID: {result['query_id']}")
        print(f"  Duration:           {result['duration_ms']} ms")
        print(f"  Rows Returned:      {result['row_count']}\n")

        if result["rows"]:
            print("Query Results:")
            print(tabulate(result["rows"], headers=result["columns"], tablefmt="psql"))
        else:
            print("Query executed successfully, but returned 0 rows.")

    except Exception as err:
        print(f"\nExecution failed: {err}")
    finally:
        db.close()
        print("\n" + "=" * 60 + "\n")


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your question: ").strip()

    if not question:
        print("Please provide a question to query.")
        sys.exit(1)

    run_pipeline(question)


if __name__ == "__main__":
    main()
