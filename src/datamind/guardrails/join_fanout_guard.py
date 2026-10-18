"""Join fan-out protection guardrail for DataMind.

Detects multi-table queries where 1:N relationships inflate parent entity counts
or aggregates (e.g. counting orders after joining to line items without DISTINCT).
"""

from __future__ import annotations

import logging
from typing import Any
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

# Known 1:N parent-child relationships where naive joins duplicate parent rows
KNOWN_FANOUT_RELATIONSHIPS = [
    ("olist_orders", "olist_order_items"),
    ("olist_orders", "olist_order_payments"),
    ("olist_orders", "olist_order_reviews"),
    ("orders", "order_items"),
    ("orders", "line_items"),
    ("orders", "payments"),
    ("orders", "reviews"),
]


class JoinFanoutGuard:
    """AST guard inspecting SQL queries for dangerous join fan-out inflation."""

    def __init__(self, dialect: str = "snowflake") -> None:
        self.dialect = dialect

    def check_query(self, sql: str) -> tuple[bool, str | None]:
        """Inspect SQL for join fan-out risk.
        
        Returns (is_valid, error_or_warning_message).
        """
        try:
            expression = sqlglot.parse_one(sql, read=self.dialect)
        except Exception:
            try:
                expression = sqlglot.parse_one(sql)
            except Exception:
                return True, None

        # Find all table names referenced in the query
        tables = set()
        for t in expression.find_all(exp.Table):
            if t.this and hasattr(t.this, "name"):
                tables.add(t.this.name.lower())

        if len(tables) < 2:
            return True, None

        # Check for multi-detail join (e.g. order_items JOIN order_payments)
        has_items = any("item" in t for t in tables)
        has_payments = any("payment" in t for t in tables)
        has_reviews = any("review" in t for t in tables)

        detail_count = sum([has_items, has_payments, has_reviews])
        if detail_count >= 2:
            return False, (
                "Join Fan-out Exception: Query simultaneously joins multiple 1:N detail tables "
                "(e.g. items and payments/reviews) on the same order, creating a multiplicative Cartesian fan-out. "
                "Aggregate each child table in separate CTEs before joining."
            )

        # Check for 1:N join with un-DISTINCT parent COUNT
        for parent, child in KNOWN_FANOUT_RELATIONSHIPS:
            if parent in tables and child in tables:
                for count_expr in expression.find_all(exp.Count):
                    # Check if DISTINCT was used
                    if not count_expr.args.get("distinct"):
                        arg = count_expr.this
                        # If counting * or parent ID, this is a known fan-out inflation bug
                        if isinstance(arg, exp.Star) or (arg and hasattr(arg, "name") and "order" in arg.name.lower()):
                            return False, (
                                f"Join Fan-out Exception: Query joins parent '{parent}' with child '{child}' "
                                f"and aggregates using COUNT({arg}) without DISTINCT. "
                                f"Use COUNT(DISTINCT {parent}.order_id) to avoid row inflation."
                            )

        return True, None
