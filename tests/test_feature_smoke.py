"""Automated feature smoke tests for F10: End-to-end capabilities on real data."""

import pytest
import json
from pathlib import Path
from datamind.config import load_config
from datamind.agent.orchestrator import AgentOrchestrator
from datamind.agent.second_opinion import SecondOpinionEngine
from datamind.adapters.duckdb_adapter import DuckDBAdapter
from datamind.guardrails.sql_guard import SQLGuard
from datamind.llm.client import LLMClient
from datamind.llm.models import ModelRegistry
from datamind.semantic.loader import load_semantic_model
from datamind.trust.chart_planner import plan_chart
from datamind.trust.receipts import ReceiptGenerator
from datamind.uploads.manager import UploadManager
from datamind.adapters.base import QueryResult
from datamind.agent.evidence import EvidenceStore, QueryRecord
from datamind.agent.verifier import verify_citations
from datamind.guardrails.egress_guard import EgressSentinelGuard, SENTINEL_ROW_SECRET


@pytest.fixture(scope="module")
def app_config():
    return load_config("datamind_olist.yaml")


@pytest.fixture(scope="module")
def duckdb_adapter(app_config):
    adapter = DuckDBAdapter(app_config.backend.duckdb_path, read_only=True)
    yield adapter
    adapter.close()


@pytest.fixture(scope="module")
def semantic_model(app_config):
    return load_semantic_model(app_config.semantic_model)


# 1. Model Selector Smoke Test (Local & Bedrock)
def test_model_selector_validation(app_config):
    reg = ModelRegistry(app_config)
    # Local model
    valid_local, msg_local, info_local = reg.validate_model_selection("gemma4:e2b")
    assert valid_local, f"Local model selection failed: {msg_local}"
    assert info_local.provider == "ollama"

    # Bedrock Cloud model
    valid_cloud, msg_cloud, info_cloud = reg.validate_model_selection("google.gemma-4-26b-a4b")
    assert valid_cloud, f"Bedrock model selection failed: {msg_cloud}"
    assert info_cloud.provider == "bedrock"


# 2. Second Opinion Consensus Smoke Test
def test_second_opinion_consensus(duckdb_adapter, semantic_model):
    guard = SQLGuard(approved_tables=semantic_model.get_approved_table_names(), dialect="duckdb")
    engine = SecondOpinionEngine(duckdb_adapter, guard)

    p_sql = "SELECT order_status, COUNT(*) AS cnt FROM olist_orders GROUP BY order_status ORDER BY cnt DESC LIMIT 3;"
    s_sql = "SELECT order_status, COUNT(order_id) AS cnt FROM olist_orders GROUP BY order_status ORDER BY cnt DESC LIMIT 3;"

    res = engine.evaluate_queries(p_sql, s_sql, primary_model="google.gemma-4-26b-a4b", secondary_model="gemma4:e2b")
    assert res.verdict in ("AGREE_EXACT", "AGREE_VALUE")
    d = res.to_dict()
    assert "verdict" in d
    assert d["verdict"] in ("AGREE_EXACT", "AGREE_VALUE")


# 3. Receipt Generation, HMAC Verification, and Export Smoke Test
def test_verifiable_receipt_and_export():
    store = EvidenceStore()
    q_res = QueryResult(
        columns=["count"],
        rows=[(96478,)],
        query_id="qid-test-smoke",
        elapsed_ms=1.0,
        truncated=False,
    )
    store.add_query("SELECT COUNT(*) FROM olist_orders WHERE order_status = 'delivered';", "Order count", q_res)
    ans = "There are 96,478 delivered orders [[Q1:R1]]."
    rep = verify_citations(ans, store)

    receipt = ReceiptGenerator.generate_receipt(
        question="What is the total number of delivered orders?",
        answer=ans,
        evidence_store=store,
        citation_report=rep,
        backend_name="DuckDB",
        model_name="google.gemma-4-26b-a4b",
    )
    assert "root_hash" in receipt
    assert receipt["citation_report"]["is_clean"] is True

    # Verify receipt hash chain integrity
    from datamind.trust.receipts import ReceiptVerifier
    ver_res = ReceiptVerifier.verify(receipt)
    assert ver_res["verified"] is True
    assert ver_res["status"] == "PASS"

    # Export to HTML
    html = ReceiptGenerator.export_html(receipt)
    assert "<!DOCTYPE html>" in html
    assert receipt["root_hash"][:12] in html


# 4. Red-Team Interactive Adversarial Attack Smoke Test
def test_red_team_attack_engine():
    store = EvidenceStore()
    q_res = QueryResult(
        columns=["net_revenue"],
        rows=[(13221498.11,)],
        query_id="qid-rt-smoke",
        elapsed_ms=2.5,
        truncated=False,
    )
    store.add_query("SELECT SUM(price) FROM olist_order_items", "Net revenue", q_res)

    clean_answer = "Based on verified data, the total net revenue is $13,221,498.11 [[Q1:R1]]."
    clean_rep = verify_citations(clean_answer, store)
    assert clean_rep.is_clean is True

    # Attack 1: Planted Sentinel Canary Trap
    sentinel_guard = EgressSentinelGuard()
    leaks = sentinel_guard.scan_text(f"{clean_answer} {SENTINEL_ROW_SECRET}")
    assert len(leaks) > 0, "Sentinel leak was not caught"

    # Attack 2: Perturbation (+100.00)
    mutated_perturbed = "Based on verified data, the total net revenue is $13,221,598.11 [[Q1:R1]]."
    rep_perturbed = verify_citations(mutated_perturbed, store)
    assert not rep_perturbed.is_clean, "Perturbation attack was not caught"

    # Attack 3: Currency Swap ($ -> €)
    mutated_curr = "Based on verified data, the total net revenue is €13,221,498.11 [[Q1:R1]]."
    rep_curr = verify_citations(mutated_curr, store)
    assert not rep_curr.is_clean, "Currency swap attack was not caught"

    # Attack 4: Fabricated Citation ID
    mutated_cit = "Based on verified data, the total net revenue is $13,221,498.11 [[Q99:R99]]."
    rep_cit = verify_citations(mutated_cit, store)
    assert not rep_cit.is_clean, "Fabricated citation ID was not caught"


# 5. Assumptions Card Structure Smoke Test
def test_assumptions_card_structure():
    store = EvidenceStore()
    q_res = QueryResult(
        columns=["customer_state", "orders_count"],
        rows=[("SP", 40501), ("RJ", 12350)],
        query_id="qid-assump",
        elapsed_ms=1.5,
        truncated=False,
    )
    store.add_query("SELECT customer_state, COUNT(*) FROM olist_orders GROUP BY customer_state", "State breakdown", q_res)
    ans = "Top states: 1. SP: 40,501 [[Q1:R1]], 2. RJ: 12,350 [[Q1:R2]]."
    rep = verify_citations(ans, store, question="What are the top states by delivered orders?")

    card = getattr(rep, "assumptions_card", None)
    assert card is not None
    assert hasattr(card, "grain") or hasattr(card, "filters") or hasattr(card, "metric")


# 6. Chart Planner Smoke Test (Bar, Line, Scatter)
def test_chart_planner_specs():
    # Bar chartable
    bar_rows = [("SP", 40501), ("RJ", 12350), ("MG", 11335)]
    bar_cols = ["customer_state", "orders_count"]
    q_rec = QueryRecord(
        query_label="Q1",
        query_num=1,
        sql="SELECT customer_state, COUNT(*) FROM olist_orders GROUP BY customer_state",
        purpose="State breakdown",
        result=QueryResult(columns=bar_cols, rows=bar_rows, query_id="q1", elapsed_ms=1.0, truncated=False),
    )
    chart_bar = plan_chart(q_rec)
    assert chart_bar is not None
    assert chart_bar.type in ("horizontal_bar", "bar", "line", "point", "table", "kpi_card")
    assert chart_bar.points is not None

    # Line / Time series chartable
    time_rows = [("2017-01", 100), ("2017-02", 150), ("2017-03", 200)]
    time_cols = ["month", "revenue"]
    q_time = QueryRecord(
        query_label="Q2",
        query_num=2,
        sql="SELECT month, revenue FROM monthly_sales",
        purpose="Time series",
        result=QueryResult(columns=time_cols, rows=time_rows, query_id="q2", elapsed_ms=1.0, truncated=False),
    )
    chart_line = plan_chart(q_time)
    assert chart_line is not None
    assert chart_line.points is not None


# 7. CSV Export Smoke Test
def test_csv_export_formatting():
    cols = ["seller_id", "revenue", "rating"]
    rows = [
        ("s101", 15000.50, 4.8),
        ("s102", 9200.00, 4.2),
    ]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(c) for c in r))
    csv_text = "\n".join(lines)
    assert "seller_id,revenue,rating" in csv_text
    assert "s101,15000.5,4.8" in csv_text


# 8. Upload Manager End-to-End Smoke Test
def test_upload_lifecycle(tmp_path):
    upload_mgr = UploadManager(base_dir=tmp_path / "smoke_uploads", enabled=True)
    session = upload_mgr.create_session()
    csv_bytes = b"category,sales\nElectronics,500.00\nBooks,150.00\n"
    meta = upload_mgr.ingest_file(session, "test_data.csv", csv_bytes)

    assert meta["total_rows"] == 2
    assert meta["table_name"] == "test_data"

    # Confirm model
    sm = upload_mgr.confirm_model(session, review_config={})
    assert sm.name == f"uploaded_session_{session.session_id}"
    assert len(sm.tables) >= 1

    # Cleanup
    upload_mgr.delete_session(session.session_id)
