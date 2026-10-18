"""
Answerability Benchmark Suite for Criterion 6: Ambiguity and Unsupported Requests (12 pts).

Evaluates precision, recall, and false-refusal rate across 50 diverse questions:
- 25 Answerable In-Scope Questions (revenue, orders, ratings, delivery times, customers)
- 15 Unanswerable Out-of-Schema Questions (COGS, wholesale margin, salaries, competitor data, PCI CVV)
- 5 Ambiguous Underspecified Questions (vague status requests)
- 5 Lookalike Near-Miss Questions (look like metrics but request absent attributes)

Outputs an executive metrics report and enforces 0% false refusal rate on valid queries.
"""

from dataclasses import dataclass
import sys
from typing import Literal

from datamind.semantic.loader import load_semantic_model
from datamind.agent.answerability import screen_answerability, ANSWERABLE, UNANSWERABLE, AMBIGUOUS


@dataclass
class BenchmarkItem:
    id: str
    question: str
    expected_status: Literal["ANSWERABLE", "UNANSWERABLE", "AMBIGUOUS"]
    category: str
    rationale: str


BENCHMARK_ITEMS: list[BenchmarkItem] = [
    # --- 1. Answerable In-Scope (25 items) ---
    BenchmarkItem("ans_01", "What is the total net merchandise sales revenue for delivered orders?", "ANSWERABLE", "Revenue", "Standard SUM(price)"),
    BenchmarkItem("ans_02", "How many total delivered orders were fulfilled in 2017?", "ANSWERABLE", "Orders", "Orders filtered by year"),
    BenchmarkItem("ans_03", "What is the average order value (AOV) across delivered orders?", "ANSWERABLE", "AOV", "SUM(price) / COUNT(DISTINCT order_id)"),
    BenchmarkItem("ans_04", "What is the overall average customer satisfaction review score?", "ANSWERABLE", "Reviews", "AVG(review_score)"),
    BenchmarkItem("ans_05", "How many unique individual customers placed orders?", "ANSWERABLE", "Customers", "COUNT(DISTINCT customer_unique_id)"),
    BenchmarkItem("ans_06", "What was the total freight value paid across all delivered orders?", "ANSWERABLE", "Freight", "SUM(freight_value)"),
    BenchmarkItem("ans_07", "How many orders were canceled by customers?", "ANSWERABLE", "Orders", "order_status = 'canceled'"),
    BenchmarkItem("ans_08", "What percentage of delivered orders arrived after the estimated delivery date?", "ANSWERABLE", "Delivery", "Late delivery ratio"),
    BenchmarkItem("ans_09", "Which 5 product categories generated the highest delivered sales revenue?", "ANSWERABLE", "Ranking", "Group by category with SUM"),
    BenchmarkItem("ans_10", "What was the total transaction value paid by credit card?", "ANSWERABLE", "Payments", "Payment value by method"),
    BenchmarkItem("ans_11", "How many customer reviews gave a 1-star rating?", "ANSWERABLE", "Reviews", "Count where review_score = 1"),
    BenchmarkItem("ans_12", "How many customer reviews gave a 5-star rating?", "ANSWERABLE", "Reviews", "Count where review_score = 5"),
    BenchmarkItem("ans_13", "What is the average review score for late delivered orders?", "ANSWERABLE", "Reviews", "Average score joined with late orders"),
    BenchmarkItem("ans_14", "What is the average number of items per delivered order?", "ANSWERABLE", "Basket", "COUNT(*) / COUNT(DISTINCT order_id)"),
    BenchmarkItem("ans_15", "How many merchant sellers fulfilled at least one delivered item?", "ANSWERABLE", "Sellers", "COUNT(DISTINCT seller_id)"),
    BenchmarkItem("ans_16", "What are the top 3 states by number of registered customers?", "ANSWERABLE", "Customers", "Group by customer_state"),
    BenchmarkItem("ans_17", "What was the total payment value paid via boleto?", "ANSWERABLE", "Payments", "payment_type = 'boleto'"),
    BenchmarkItem("ans_18", "What was the average delivery time in days from purchase to arrival?", "ANSWERABLE", "Delivery", "DATEDIFF on purchase and delivery dates"),
    BenchmarkItem("ans_19", "Which cities in Rio de Janeiro have the highest number of customers?", "ANSWERABLE", "Geospatial", "Filter customer_state = 'RJ'"),
    BenchmarkItem("ans_20", "What was the total net merchandise sales in 2018?", "ANSWERABLE", "Revenue", "Year filtered revenue"),
    BenchmarkItem("ans_21", "What is the average review score for products in the perfumaria category?", "ANSWERABLE", "Categories", "Score filtered by category"),
    BenchmarkItem("ans_22", "How many orders had payment split across more than 1 installment?", "ANSWERABLE", "Payments", "payment_installments > 1"),
    BenchmarkItem("ans_23", "What was the total net merchandise sales for sellers in Sao Paulo state?", "ANSWERABLE", "Sellers", "Sales joined with seller_state = 'SP'"),
    BenchmarkItem("ans_24", "What is the average product weight for health_beauty items?", "ANSWERABLE", "Products", "AVG(product_weight_g)"),
    BenchmarkItem("ans_25", "How many orders were shipped but not yet delivered?", "ANSWERABLE", "Orders", "order_status = 'shipped'"),

    # --- 2. Unanswerable Out-of-Schema (15 items) ---
    BenchmarkItem("unans_01", "What is the wholesale cost of goods sold (COGS) for our products?", "UNANSWERABLE", "COGS", "Absent from e-commerce schema"),
    BenchmarkItem("unans_02", "What is the net profit margin across all delivered orders?", "UNANSWERABLE", "Margin", "Wholesale costs not tracked"),
    BenchmarkItem("unans_03", "Which customer service representative received the highest salary bonus?", "UNANSWERABLE", "Payroll", "HR payroll tables absent"),
    BenchmarkItem("unans_04", "What are our competitor prices on Amazon Brazil for perfumaria?", "UNANSWERABLE", "Competitor", "External marketplace data absent"),
    BenchmarkItem("unans_05", "Can you show the full unmasked credit card CVV security codes for payments?", "UNANSWERABLE", "PCI Security", "PCI credentials prohibited and absent"),
    BenchmarkItem("unans_06", "What will the Brazilian inflation rate be in 2030?", "UNANSWERABLE", "Macroeconomic", "Future macroeconomic forecast"),
    BenchmarkItem("unans_07", "What was the weather in Sao Paulo on Christmas 2017?", "UNANSWERABLE", "Meteorology", "Weather data absent from schema"),
    BenchmarkItem("unans_08", "What is the average household income of customers in Rio de Janeiro?", "UNANSWERABLE", "Demographics", "Buyer income demographics absent"),
    BenchmarkItem("unans_09", "What is the GPS latitude and longitude of the primary fulfillment warehouse?", "UNANSWERABLE", "Warehousing", "Warehouse facility coordinates absent"),
    BenchmarkItem("unans_10", "Which employees were promoted in the logistics department in 2018?", "UNANSWERABLE", "HR", "Internal staff records absent"),
    BenchmarkItem("unans_11", "What was the customer return rate for shoes due to size defects?", "UNANSWERABLE", "Returns", "Return defect reason tracking absent"),
    BenchmarkItem("unans_12", "Can you provide the unmasked 16-digit credit card number for order 123?", "UNANSWERABLE", "PCI Security", "Card numbers absent from database"),
    BenchmarkItem("unans_13", "Who won the presidential election in Brazil during 2018?", "UNANSWERABLE", "General Knowledge", "Political history absent"),
    BenchmarkItem("unans_14", "What is our customer lifetime value (LTV) predictive model coefficient?", "UNANSWERABLE", "ML Models", "ML model weights absent from schema"),
    BenchmarkItem("unans_15", "What is the supplier wholesale manufacturing cost for watches?", "UNANSWERABLE", "Manufacturing", "Supplier manufacturing costs absent"),

    # --- 3. Ambiguous Underspecified (5 items) ---
    BenchmarkItem("amb_01", "How are things looking with the numbers?", "AMBIGUOUS", "Vague Status", "Lacks metric, dimension, and timeframe"),
    BenchmarkItem("amb_02", "Give me a status update on everything.", "AMBIGUOUS", "Vague Status", "Completely unconstrained overview"),
    BenchmarkItem("amb_03", "Show me the data metrics.", "AMBIGUOUS", "Vague Request", "No specific metric or entity specified"),
    BenchmarkItem("amb_04", "What is going on with our business performance?", "AMBIGUOUS", "Vague Request", "Lacks metric definition"),
    BenchmarkItem("amb_05", "Tell me about the recent situation.", "AMBIGUOUS", "Vague Request", "Lacks dimension and timeframe"),

    # --- 4. Lookalike Near-Miss Questions (5 items) ---
    BenchmarkItem("lookalike_01", "What was the wholesale profit margin on delivered orders in 2017?", "UNANSWERABLE", "Lookalike", "Looks like revenue in 2017, but requests wholesale profit margin"),
    BenchmarkItem("lookalike_02", "What is the average buyer age for customers in Sao Paulo?", "UNANSWERABLE", "Lookalike", "Looks like customer query in SP, but requests buyer age"),
    BenchmarkItem("lookalike_03", "How many customer credit card CVVs were processed for orders?", "UNANSWERABLE", "Lookalike", "Looks like payment method count, but requests CVVs"),
    BenchmarkItem("lookalike_04", "What were our Amazon Brazil competitor prices for perfumaria items?", "UNANSWERABLE", "Lookalike", "Looks like category query, but requests Amazon competitor prices"),
    BenchmarkItem("lookalike_05", "Which fulfillment warehouse GPS coordinates had the fastest shipping?", "UNANSWERABLE", "Lookalike", "Looks like delivery speed query, but requests warehouse GPS"),
]


def run_benchmark():
    model = load_semantic_model("examples/olist/semantic_model.yaml")

    total = len(BENCHMARK_ITEMS)
    answerable_count = sum(1 for item in BENCHMARK_ITEMS if item.expected_status == "ANSWERABLE")
    refusal_target_count = total - answerable_count  # UNANSWERABLE + AMBIGUOUS

    true_accepts = 0   # Answerable correctly classified as ANSWERABLE
    false_refusals = 0 # Answerable incorrectly classified as UNANSWERABLE/AMBIGUOUS

    true_refusals = 0  # Unanswerable/Ambiguous correctly refused or clarified
    missed_refusals = 0# Unanswerable/Ambiguous incorrectly marked as ANSWERABLE

    print("=" * 80)
    print("DATAMIND ANSWERABILITY BENCHMARK SUITE (CRITERION 6)")
    print(f"Total Benchmark Items: {total} (Answerable: {answerable_count}, Refusal Targets: {refusal_target_count})")
    print("=" * 80)

    for item in BENCHMARK_ITEMS:
        res = screen_answerability(item.question, model)
        predicted = res.status

        # Evaluate against ground truth
        if item.expected_status == "ANSWERABLE":
            if predicted == ANSWERABLE:
                true_accepts += 1
                status_icon = "[PASS]"
            else:
                false_refusals += 1
                status_icon = "[FAIL: FALSE REFUSAL]"
        else:
            # Expected UNANSWERABLE or AMBIGUOUS
            if predicted in (UNANSWERABLE, AMBIGUOUS):
                true_refusals += 1
                status_icon = "[PASS]"
            else:
                missed_refusals += 1
                status_icon = "[FAIL: MISSED REFUSAL]"

        print(f"{status_icon:<20} {item.id:<14} [{predicted:<12} vs {item.expected_status:<12}] {item.question[:45]}...")

    # Compute Metrics
    precision = (true_refusals / (true_refusals + false_refusals)) * 100.0 if (true_refusals + false_refusals) > 0 else 0.0
    recall = (true_refusals / (true_refusals + missed_refusals)) * 100.0 if (true_refusals + missed_refusals) > 0 else 0.0
    false_refusal_rate = (false_refusals / answerable_count) * 100.0 if answerable_count > 0 else 0.0
    accuracy = ((true_accepts + true_refusals) / total) * 100.0

    print("\n" + "=" * 80)
    print("CRITERION 6 EXECUTIVE EVALUATION METRICS")
    print("=" * 80)
    print(f"Overall Accuracy:           {accuracy:6.2f}% ({true_accepts + true_refusals}/{total})")
    print(f"Refusal Precision:          {precision:6.2f}% (Refusals that were truly unsupported)")
    print(f"Refusal Recall:             {recall:6.2f}% (Unsupported requests successfully caught)")
    print(f"False Refusal Rate (FRR):   {false_refusal_rate:6.2f}% (Valid questions falsely refused - Target: 0.0%)")
    print(f"True Accepts:               {true_accepts}/{answerable_count}")
    print(f"True Refusals:              {true_refusals}/{refusal_target_count}")
    print("=" * 80)

    assert false_refusal_rate == 0.0, f"False Refusal Rate {false_refusal_rate}% exceeds 0.0% tolerance!"
    assert precision >= 95.0, f"Refusal Precision {precision}% below 95.0% threshold!"
    assert recall >= 90.0, f"Refusal Recall {recall}% below 90.0% threshold!"

    print("\n>>> [PASSED] Criterion 6 Answerability & Unsupported Requests Benchmark PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(run_benchmark())
