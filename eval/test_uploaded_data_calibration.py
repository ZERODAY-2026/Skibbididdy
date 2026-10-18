"""Test uploaded data ingestion, semantic drafting, and calibration query accuracy (W8 / F7)."""

import os
import time
import json
import duckdb
from pathlib import Path
from datamind.uploads.manager import UploadManager
from datamind.config import load_config
from datamind.llm.client import LLMClient


def build_calibration_suite():
    temp_dir = Path(".datamind/calibration_uploads")
    temp_dir.mkdir(parents=True, exist_ok=True)

    upload_mgr = UploadManager(
        base_dir=temp_dir,
        enabled=True,
        max_file_size_bytes=100 * 1024 * 1024,  # 100 MB for Online_Retail.csv
        max_rows=1_000_000,
        max_columns=100,
        session_ttl_seconds=3600,
    )

    session = upload_mgr.create_session()
    print(f"Created isolated upload session: {session.session_id}")

    # File 1: UCI Online Retail CSV
    f1_path = Path("examples/secondary/online_retail/Online_Retail.csv")
    if not f1_path.exists():
        raise FileNotFoundError(f"Missing {f1_path}")
    t0_f1 = time.time()
    meta1 = upload_mgr.ingest_file(session, "online_retail.csv", f1_path.read_bytes())
    t1_f1 = time.time() - t0_f1
    print(f"File 1 (Online Retail): {meta1['total_rows']} rows ingested in {t1_f1:.2f}s")

    # File 2: Multi-table relational sample (payments and sellers)
    f2_path = Path("data/olist/olist_order_payments_dataset.csv")
    meta2 = upload_mgr.ingest_file(session, "order_payments.csv", f2_path.read_bytes())
    print(f"File 2 (Order Payments): {meta2['total_rows']} rows ingested")

    # File 3: Messy CSV (BOM, semicolon, currency symbols, dirty rows)
    messy_csv = (
        "\ufeffTransactionID;CustomerName;ItemCategory;AmountPaid;Rating;Notes\n"
        "TX101;Alice;Electronics;$450.00;5;Excellent condition\n"
        "TX102;Bob;Books;$25.50;4;Arrived on time\n"
        'TX103;Charlie;Electronics;$1,200.00;1;"Defective unit; returned"\n'
        "TX104;Diana;Clothing;$85.20;5;Great quality\n"
        "TX105;Evan;Books;$15.00;3;Average packaging\n"
        "TX106;Fiona;Clothing;$120.00;4;Fits well\n"
        "TX107;George;Electronics;$310.50;5;Fast shipping\n"
    )
    meta3 = upload_mgr.ingest_file(session, "messy_store_records.csv", messy_csv.encode("utf-8-sig"))
    print(f"File 3 (Messy CSV): {meta3['total_rows']} rows ingested")

    # Finalize session draft model
    t0_draft = time.time()
    upload_mgr.confirm_model(session, review_config={})
    t_ready = time.time() - t0_f1
    print(f"Session semantic model drafted. Time from upload to ready: {t_ready:.2f}s (Threshold: < 180s)")

    # Compute Ground Truth via DuckDB directly on session database
    db_con = duckdb.connect(str(session.db_path))

    # Define calibration questions covering: counts, sums, distinct counts, top-1 category, min and max, simple filtered counts
    calibration_cases = [
        # File 1 (Online Retail)
        {
            "id": "calib_01",
            "file": "online_retail",
            "question": "What is the total number of records in the online retail dataset?",
            "ground_truth_sql": "SELECT COUNT(*) FROM online_retail;",
        },
        {
            "id": "calib_02",
            "file": "online_retail",
            "question": "What is the total quantity of items sold across all transactions?",
            "ground_truth_sql": "SELECT SUM(Quantity) FROM online_retail WHERE Quantity > 0;",
        },
        {
            "id": "calib_03",
            "file": "online_retail",
            "question": "How many distinct customer IDs made purchases in online retail?",
            "ground_truth_sql": "SELECT COUNT(DISTINCT CustomerID) FROM online_retail WHERE CustomerID IS NOT NULL;",
        },
        {
            "id": "calib_04",
            "file": "online_retail",
            "question": "Which country had the single highest number of transactions?",
            "ground_truth_sql": "SELECT Country FROM online_retail GROUP BY Country ORDER BY COUNT(*) DESC LIMIT 1;",
        },
        {
            "id": "calib_05",
            "file": "online_retail",
            "question": "What is the maximum unit price recorded for an item in online retail?",
            "ground_truth_sql": "SELECT ROUND(MAX(UnitPrice), 2) FROM online_retail;",
        },
        # File 2 (Order Payments)
        {
            "id": "calib_06",
            "file": "order_payments",
            "question": "What is the total payment value across all payment records?",
            "ground_truth_sql": "SELECT ROUND(SUM(payment_value), 2) FROM order_payments;",
        },
        {
            "id": "calib_07",
            "file": "order_payments",
            "question": "How many total payment transactions were recorded in order payments?",
            "ground_truth_sql": "SELECT COUNT(*) FROM order_payments;",
        },
        {
            "id": "calib_08",
            "file": "order_payments",
            "question": "What is the total payment amount processed specifically via credit card?",
            "ground_truth_sql": "SELECT ROUND(SUM(payment_value), 2) FROM order_payments WHERE payment_type = 'credit_card';",
        },
        {
            "id": "calib_09",
            "file": "order_payments",
            "question": "What is the maximum number of installments used in any payment transaction?",
            "ground_truth_sql": "SELECT MAX(payment_installments) FROM order_payments;",
        },
        # File 3 (Messy CSV)
        {
            "id": "calib_10",
            "file": "messy_store",
            "question": "What is the total number of transactions in the messy store records?",
            "ground_truth_sql": "SELECT COUNT(*) FROM messy_store_records;",
        },
        {
            "id": "calib_11",
            "file": "messy_store",
            "question": "What is the average customer rating in the messy store records?",
            "ground_truth_sql": "SELECT ROUND(AVG(Rating), 2) FROM messy_store_records;",
        },
        {
            "id": "calib_12",
            "file": "messy_store",
            "question": "How many transactions received a 5 star rating in messy store records?",
            "ground_truth_sql": "SELECT COUNT(*) FROM messy_store_records WHERE Rating = 5;",
        },
        {
            "id": "calib_13",
            "file": "messy_store",
            "question": "Which item category had the highest number of transactions in messy store records?",
            "ground_truth_sql": "SELECT ItemCategory FROM messy_store_records GROUP BY ItemCategory ORDER BY COUNT(*) DESC LIMIT 1;",
        },
    ]

    print("\nComputing ground truth values via DuckDB...")
    for item in calibration_cases:
        res = db_con.execute(item["ground_truth_sql"]).fetchall()
        val = res[0][0]
        item["ground_truth_value"] = round(float(val), 2) if isinstance(val, (int, float)) else str(val).strip()
        print(f"  {item['id']}: {item['ground_truth_value']}")

    db_con.close()

    # Get isolated orchestrator
    cfg = load_config("datamind_olist.yaml")
    orchestrator, adapter, retriever = upload_mgr.create_session_orchestrator(session.session_id, config=cfg)

    print("\nEvaluating calibration questions against uploaded session...")
    passes = 0
    total = len(calibration_cases)
    first_answer_time = 0.0

    results = []
    for idx, item in enumerate(calibration_cases, 1):
        t0 = time.time()
        agent_res = orchestrator.run(item["question"])
        elapsed = time.time() - t0
        if idx == 1:
            first_answer_time = t_ready + elapsed

        def norm_val(v):
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return round(float(v), 2)
            s = str(v).strip().replace("$", "").replace(",", "")
            try:
                return round(float(s), 2)
            except ValueError:
                return str(v).strip().lower()

        gt_norm = norm_val(item["ground_truth_value"])
        matched = False
        for q in agent_res.evidence_store.queries:
            for r in (q.result.rows if q.result else []):
                for c in r:
                    if norm_val(c) == gt_norm:
                        matched = True
                        break
        if not matched and isinstance(gt_norm, (int, float)):
            import re
            nums = re.findall(r"[\d,]+(?:\.\d+)?", agent_res.answer)
            for n in nums:
                if norm_val(n) == gt_norm:
                    matched = True
                    break
        elif not matched and isinstance(gt_norm, str):
            if gt_norm in agent_res.answer.lower():
                matched = True

        if matched and agent_res.citation_report.is_clean:
            passes += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  [{status}] [{idx:>2}/{total}] {item['id']}: expected={gt_norm} ({elapsed:.2f}s)")
        results.append({
            "id": item["id"],
            "question": item["question"],
            "ground_truth": item["ground_truth_value"],
            "status": status,
            "answer": agent_res.answer,
            "elapsed_s": round(elapsed, 2),
        })

    pass_rate = round(passes / total * 100, 1)
    print("\n" + "=" * 60)
    print(f"CALIBRATION SCORE: {passes}/{total} Passed ({pass_rate}%)")
    print(f"Time from upload to first answer: {first_answer_time:.2f}s (Threshold: < 180s)")
    print("=" * 60)

    # Save artifact
    out_file = Path("eval/results/upload_calibration_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "passed": passes,
            "total": total,
            "pass_rate_pct": pass_rate,
            "time_to_ready_s": round(t_ready, 2),
            "time_to_first_answer_s": round(first_answer_time, 2),
            "results": results,
        }, f, indent=2)

    # Clean up upload session
    adapter.close()
    if retriever:
        retriever.close()
    upload_mgr.delete_session(session.session_id)
    return passes, total, pass_rate, first_answer_time


if __name__ == "__main__":
    build_calibration_suite()
