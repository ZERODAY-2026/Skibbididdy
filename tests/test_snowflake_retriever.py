import pytest
from unittest.mock import MagicMock
from datamind.retrieval.snowflake_retriever import SnowflakeRetriever
from datamind.adapters.base import QueryResult


def test_snowflake_retriever_olist_search():
    adapter = MagicMock()
    adapter.list_tables.return_value = ["OLIST_ORDER_REVIEWS", "OLIST_ORDERS"]
    adapter.execute.return_value = QueryResult(
        columns=["REVIEW_ID", "ORDER_ID", "REVIEW_SCORE", "REVIEW_COMMENT_TITLE", "REVIEW_COMMENT_MESSAGE"],
        rows=[
            ("rev_123", "ord_456", 1, "Atraso", "O produto atrasou muito na entrega"),
            ("rev_789", "ord_101", 1, "", "Demorou demais e veio com defeito"),
        ],
        query_id="sfqid-test-12345",
        elapsed_ms=120.0,
        truncated=False,
    )

    retriever = SnowflakeRetriever(adapter=adapter)
    chunks = retriever.search("What are the delay complaints in reviews?", k=5)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "D:OLIST_ORDER_REVIEWS:rev_123"
    assert chunks[0].doc_name == "OLIST_ORDER_REVIEWS"
    assert "Score: 1/5" in chunks[0].text
    assert chunks[0].metadata["snowflake_query_id"] == "sfqid-test-12345"

    # Verify SQL execution was called with ILIKE
    assert adapter.execute.called
    executed_sql = adapter.execute.call_args[0][0]
    assert "OLIST_ORDER_REVIEWS" in executed_sql
    assert "ILIKE" in executed_sql


def test_snowflake_retriever_support_tickets_search():
    adapter = MagicMock()
    adapter.list_tables.return_value = ["SUPPORT_TICKETS", "PRODUCTS"]
    adapter.execute.return_value = QueryResult(
        columns=["ticket_id", "created_at", "product_name", "category", "priority", "body"],
        rows=[
            ("TK-101", "2026-01-15", "Widget A", "Hardware", "High", "Widget broke on day 2"),
        ],
        query_id="sfqid-test-67890",
        elapsed_ms=90.0,
        truncated=False,
    )

    retriever = SnowflakeRetriever(adapter=adapter)
    chunks = retriever.search("broken widget hardware complaints", k=3)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "D:SUPPORT_TICKETS:TK-101"
    assert chunks[0].metadata["product"] == "Widget A"
    assert chunks[0].metadata["snowflake_query_id"] == "sfqid-test-67890"


def test_snowflake_retriever_empty_adapter():
    retriever = SnowflakeRetriever(adapter=None)
    chunks = retriever.search("anything", k=3)
    assert chunks == []
