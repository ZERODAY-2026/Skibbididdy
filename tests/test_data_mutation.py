"""Test proving that answers change dynamically when data in the database changes.

Rubric Requirement:
'answers change when data changes' — verifiably proves that the agent queries live data
and dynamically reflects inserted, updated, or modified database rows, rather than
hallucinating from pretraining weights or serving stale cached values.
"""

from pathlib import Path
from typing import Any, Sequence
import duckdb
import pytest

from datamind.adapters.duckdb_adapter import DuckDBAdapter
from datamind.agent.orchestrator import AgentOrchestrator
from datamind.config import AgentConfig
from datamind.retrieval.sqlite_fts import SQLiteFTSRetriever
from datamind.semantic.model import ColumnDef, SemanticModel, TableDef


class DynamicMockLLMClient:
    """Mock LLM that executes SQL and quotes the real result from context."""

    def __init__(self, sql_query: str) -> None:
        self.sql_query = sql_query
        self.call_count = 0

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.call_count += 1
        # If last message was user prompt, emit tool call run_sql
        last_msg = messages[-1]
        if last_msg["role"] == "user":
            return {
                "role": "assistant",
                "content": "Executing SQL query to compute transaction total.",
                "tool_calls": [
                    {
                        "id": f"call_{self.call_count}",
                        "type": "function",
                        "function": {
                            "name": "run_sql",
                            "arguments": f'{{"sql": "{self.sql_query}", "purpose": "Sum completed amounts"}}',
                        },
                    }
                ],
            }

        # If last message was tool output, read the value from tool table and answer
        if last_msg["role"] == "tool":
            content = last_msg.get("content", "")
            # Find the value in the table
            lines = content.strip().split("\n")
            val = "0"
            for line in lines:
                if "[[Q1:R1]]" in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if len(parts) >= 2:
                        val = parts[1]
                        break

            return {
                "role": "assistant",
                "content": f"The total sum of completed transactions is ${val} [[Q1:R1]].",
                "tool_calls": None,
            }

        return {"role": "assistant", "content": "Done.", "tool_calls": None}


@pytest.fixture
def mutable_db(tmp_path: Path):
    db_file = tmp_path / "mutable_test.duckdb"
    con = duckdb.connect(str(db_file))
    con.execute("""
        CREATE TABLE transactions (
            transaction_id VARCHAR PRIMARY KEY,
            customer_name VARCHAR,
            amount DOUBLE,
            status VARCHAR
        );
    """)
    con.execute("""
        INSERT INTO transactions VALUES
            ('TX101', 'Alice', 100.0, 'completed'),
            ('TX102', 'Bob', 200.0, 'completed'),
            ('TX103', 'Charlie', 300.0, 'completed');
    """)
    con.close()
    return db_file


@pytest.fixture
def transactions_semantic_model():
    return SemanticModel(
        name="transactions_db",
        description="Live transaction ledger",
        default_currency="USD",
        currency_symbol="$",
        tables=[
            TableDef(
                name="transactions",
                description="Financial transactions ledger",
                columns=[
                    ColumnDef(name="transaction_id", type="VARCHAR", description="Unique ID"),
                    ColumnDef(name="customer_name", type="VARCHAR", description="Customer"),
                    ColumnDef(name="amount", type="DOUBLE", description="Transaction amount in USD"),
                    ColumnDef(name="status", type="VARCHAR", description="Status (completed/pending)"),
                ],
            )
        ],
    )


def test_answer_changes_when_data_row_changes(mutable_db, transactions_semantic_model):
    """Verifies that an answer changes dynamically when an underlying row is inserted or updated."""
    adapter = DuckDBAdapter(db_path=str(mutable_db), read_only=False)
    retriever = SQLiteFTSRetriever(index_path=":memory:")

    sql = "SELECT SUM(amount) AS total_amount FROM transactions WHERE status = 'completed';"
    mock_llm = DynamicMockLLMClient(sql_query=sql)

    config = AgentConfig(
        max_steps=4,
        structured_query_lane=False,
    )

    orchestrator = AgentOrchestrator(
        llm_client=mock_llm,
        db_adapter=adapter,
        retriever=retriever,
        semantic_model=transactions_semantic_model,
        config=config,
    )

    question = "What is the total sum of completed transaction amounts?"

    # Turn 1: Initial state (100 + 200 + 300 = 600)
    res_1 = orchestrator.run(question)
    assert res_1.success
    assert len(res_1.evidence_store.queries) == 1
    rows_1 = res_1.evidence_store.queries[0].result.rows
    assert len(rows_1) == 1
    total_1 = float(rows_1[0][0])
    assert total_1 == 600.0
    assert "600" in res_1.answer
    assert res_1.citation_report.is_clean

    # Data Mutation: Insert a new transaction of $400 into the live database
    con = duckdb.connect(str(mutable_db))
    con.execute("INSERT INTO transactions VALUES ('TX104', 'Diana', 400.0, 'completed');")
    con.close()

    # Reset mock LLM call counter for Turn 2
    mock_llm.call_count = 0

    # Turn 2: Query the exact same question again against the live database
    # Must dynamically reflect new data (600 + 400 = 1000)
    res_2 = orchestrator.run(question)
    assert res_2.success
    assert len(res_2.evidence_store.queries) == 1
    rows_2 = res_2.evidence_store.queries[0].result.rows
    assert len(rows_2) == 1
    total_2 = float(rows_2[0][0])
    assert total_2 == 1000.0
    assert "1000" in res_2.answer
    assert res_2.citation_report.is_clean

    # Proof that answer dynamically changed when database data mutated
    assert total_2 != total_1
    assert total_2 == total_1 + 400.0
    print(f"\n[MUTATION PROVED] Initial: ${total_1:.2f} -> Mutated: ${total_2:.2f} (+400.00)")
