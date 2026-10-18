"""Evaluation harness for W4 / F4: Cross-language English-question to Portuguese-review retrieval.

Evaluates 40 diverse English customer questions against the indexed Olist reviews
with ground-truth review IDs derived from deterministic keyword and score rules.
Reports Recall@5 (pass threshold: >= 80%).
"""

import sys
sys.path.insert(0, ".")

import duckdb
from datamind.retrieval.sqlite_fts import SQLiteFTSRetriever

QUERIES_AND_RULES = [
    # Delivery delay / late fulfillment (Negative, 1-2 stars)
    ("customers complaining about late delivery or delays", "review_score <= 2 AND (review_comment_message LIKE '%atras%' OR review_comment_message LIKE '%demor%')"),
    ("shipments that arrived after the promised delivery deadline", "review_score <= 2 AND (review_comment_message LIKE '%prazo%' OR review_comment_message LIKE '%atras%')"),
    ("reviews mentioning severe delivery delays", "review_score <= 2 AND (review_comment_message LIKE '%demor%' OR review_comment_message LIKE '%atras%')"),
    ("complaints regarding late package arrivals", "review_score <= 2 AND (review_comment_message LIKE '%atras%' OR review_comment_message LIKE '%chegou tarde%')"),
    ("buyers whose orders took too long to arrive", "review_score <= 2 AND (review_comment_message LIKE '%demorou%' OR review_comment_message LIKE '%espera%')"),

    # Damaged / broken / defective items
    ("customers reporting defective or broken merchandise", "review_score <= 2 AND (review_comment_message LIKE '%quebrad%' OR review_comment_message LIKE '%defeito%')"),
    ("orders arriving with damaged packaging or broken items", "review_score <= 2 AND (review_comment_message LIKE '%danificad%' OR review_comment_message LIKE '%amassad%')"),
    ("products received with scratches or physical defects", "review_score <= 2 AND (review_comment_message LIKE '%defeito%' OR review_comment_message LIKE '%avariad%')"),
    ("damaged goods upon box opening", "review_score <= 2 AND (review_comment_message LIKE '%avariad%' OR review_comment_message LIKE '%quebrad%')"),
    ("faulty products that stopped working", "review_score <= 2 AND (review_comment_message LIKE '%defeito%' OR review_comment_message LIKE '%funciona%')"),

    # Wrong or missing items
    ("customers who received the wrong product entirely", "review_score <= 2 AND (review_comment_message LIKE '%errad%' OR review_comment_message LIKE '%diferente%')"),
    ("orders with missing parts or incomplete items", "review_score <= 2 AND (review_comment_message LIKE '%falt%' OR review_comment_message LIKE '%incomplet%')"),
    ("received different model than what was ordered", "review_score <= 2 AND (review_comment_message LIKE '%diferente%' OR review_comment_message LIKE '%errad%')"),
    ("incomplete parcel with missing accessories", "review_score <= 2 AND (review_comment_message LIKE '%falt%' OR review_comment_message LIKE '%veio faltando%')"),
    ("wrong item delivered in package", "review_score <= 2 AND (review_comment_message LIKE '%veio errad%' OR review_comment_message LIKE '%produto errad%')"),

    # Non-delivery / never received
    ("customers who never received their purchased items", "review_score <= 2 AND (review_comment_message LIKE '%nao recebi%' OR review_comment_message LIKE '%nao foi entregue%')"),
    ("orders that were never delivered to destination", "review_score <= 2 AND (review_comment_message LIKE '%nunca recebi%' OR review_comment_message LIKE '%nao chegou%')"),
    ("items lost in transit with no delivery", "review_score <= 2 AND (review_comment_message LIKE '%nao recebi%' OR review_comment_message LIKE '%extravio%')"),
    ("complaints that parcel never showed up", "review_score <= 2 AND (review_comment_message LIKE '%nao entregue%' OR review_comment_message LIKE '%nao chegou%')"),
    ("missing package never delivered by post", "review_score <= 2 AND (review_comment_message LIKE '%nao recebi%' OR review_comment_message LIKE '%correios%')"),

    # Cancellation & refund requests
    ("customers asking for order cancellation and refund", "review_score <= 2 AND (review_comment_message LIKE '%cancel%' OR review_comment_message LIKE '%estorno%')"),
    ("requests for money back and return", "review_score <= 2 AND (review_comment_message LIKE '%devolu%' OR review_comment_message LIKE '%reembols%')"),
    ("customers demanding refund for unfulfilled purchases", "review_score <= 2 AND (review_comment_message LIKE '%estorno%' OR review_comment_message LIKE '%dinheiro%')"),
    ("cancellation due to merchant delays", "review_score <= 2 AND (review_comment_message LIKE '%cancel%' OR review_comment_message LIKE '%atras%')"),
    ("returns and chargeback complaints", "review_score <= 2 AND (review_comment_message LIKE '%devol%' OR review_comment_message LIKE '%estorno%')"),

    # Positive reviews: Fast delivery (5 stars)
    ("customers praising extremely fast delivery before deadline", "review_score = 5 AND (review_comment_message LIKE '%antes do prazo%' OR review_comment_message LIKE '%rapido%')"),
    ("orders delivered much earlier than expected", "review_score = 5 AND (review_comment_message LIKE '%antes do prazo%' OR review_comment_message LIKE '%chegou antes%')"),
    ("super fast shipping and prompt arrival", "review_score = 5 AND (review_comment_message LIKE '%muito rapid%' OR review_comment_message LIKE '%entrega rapida%')"),
    ("speedy fulfillment and delivery compliments", "review_score = 5 AND (review_comment_message LIKE '%rapida%' OR review_comment_message LIKE '%agil%')"),
    ("express parcel delivery compliments", "review_score = 5 AND (review_comment_message LIKE '%rapido%' OR review_comment_message LIKE '%antes%')"),

    # Positive reviews: Product quality & recommendation (5 stars)
    ("customers highly recommending the product quality", "review_score = 5 AND (review_comment_message LIKE '%recomendo%' OR review_comment_message LIKE '%qualidade%')"),
    ("reviews stating product is excellent and beautiful", "review_score = 5 AND (review_comment_message LIKE '%excelente%' OR review_comment_message LIKE '%lindo%')"),
    ("buyers expressing total satisfaction with their purchase", "review_score = 5 AND (review_comment_message LIKE '%satisfeit%' OR review_comment_message LIKE '%otimo%')"),
    ("compliments about great product materials and build", "review_score = 5 AND (review_comment_message LIKE '%qualidade%' OR review_comment_message LIKE '%perfeito%')"),
    ("customers who loved their new purchase", "review_score = 5 AND (review_comment_message LIKE '%adorei%' OR review_comment_message LIKE '%amei%')"),
    ("five star reviews praising seller service", "review_score = 5 AND (review_comment_message LIKE '%parabens%' OR review_comment_message LIKE '%otimo vendedor%')"),
    ("great product as described in the catalog", "review_score = 5 AND (review_comment_message LIKE '%conforme o anuncio%' OR review_comment_message LIKE '%recomendo%')"),
    ("satisfied buyers recommending the store", "review_score = 5 AND (review_comment_message LIKE '%super recomendo%' OR review_comment_message LIKE '%loja%')"),
    ("top quality goods with flawless condition", "review_score = 5 AND (review_comment_message LIKE '%perfeito%' OR review_comment_message LIKE '%excelente produto%')"),
    ("delighted customer feedback on delivered item", "review_score = 5 AND (review_comment_message LIKE '%muito bom%' OR review_comment_message LIKE '%adorei%')"),
]


def evaluate_retrieval_recall():
    print("=" * 70)
    print("CROSS-LANGUAGE RETRIEVAL RECALL@5 BENCHMARK (English -> Portuguese Reviews)")
    print("=" * 70)

    duck_con = duckdb.connect("data/olist/olist.duckdb", read_only=True)
    retriever = SQLiteFTSRetriever(".datamind/fts_olist.sqlite", "data/olist/olist.duckdb")

    hits = 0
    total = len(QUERIES_AND_RULES)

    for idx, (en_query, sql_rule) in enumerate(QUERIES_AND_RULES, start=1):
        # 1. Fetch ground-truth review IDs from duckdb using rule (all matches, not arbitrary 200 limit)
        gt_query = f"SELECT review_id FROM olist_order_reviews WHERE {sql_rule}"
        gt_rows = duck_con.execute(gt_query).fetchall()
        gt_ids = {r[0] for r in gt_rows}

        # 2. Run retrieval search
        chunks = retriever.search(en_query, k=5)
        retrieved_ids = [c.chunk_id.replace("D:order_reviews:", "") for c in chunks]

        # 3. Calculate Hit@5
        hit = any(r_id in gt_ids for r_id in retrieved_ids)
        if hit:
            hits += 1
            badge = "[HIT ]"
        else:
            badge = "[MISS]"

        print(f"  {badge} [{idx:02d}/{total}] '{en_query[:55]}...' -> {len(retrieved_ids)} retrieved, {len(gt_ids)} GT matches")

    duck_con.close()
    retriever.close()

    recall_at_5 = (hits / total) * 100.0
    print("-" * 70)
    print(f"RESULTS: {hits}/{total} Hits ({recall_at_5:.1f}% Recall@5)")
    print(f"Threshold: >= 80.0% -> {'PASS' if recall_at_5 >= 80.0 else 'FAIL'}")
    print("=" * 70)

    return recall_at_5


if __name__ == "__main__":
    rec = evaluate_retrieval_recall()
    sys.exit(0 if rec >= 80.0 else 1)
