from __future__ import annotations

import logging
import re
from typing import Any

from datamind.adapters.snowflake_adapter import SnowflakeAdapter
from datamind.retrieval.base import Chunk, Retriever
from datamind.retrieval.sqlite_fts import CROSS_LANG_EXPANSIONS

logger = logging.getLogger(__name__)


class SnowflakeRetriever(Retriever):
    """Document retriever that queries text and reviews directly from Snowflake cloud data warehouse.

    Eliminates local file dependencies, executing live queries with real Snowflake query IDs (sfqid).
    """

    def __init__(
        self,
        adapter: SnowflakeAdapter | None = None,
        table_name: str = "OLIST_ORDER_REVIEWS",
    ) -> None:
        self.adapter = adapter
        self.table_name = table_name

    def build_index(self) -> None:
        """No-op for Snowflake: documents are queried live at runtime from the cloud warehouse."""
        pass

    def search(self, query: str, k: int = 5, filters: dict[str, Any] | None = None) -> list[Chunk]:
        """Search text documents live in Snowflake using multilingual ILIKE clauses."""
        if self.adapter is None:
            return []

        # Extract search terms and Portuguese cross-language expansions
        raw_words = re.findall(r"[a-zA-Z0-9_\u00C0-\u00FF]+", query.lower())
        stop_words = {
            "the", "a", "an", "is", "in", "for", "of", "and", "or", "what",
            "are", "tell", "me", "about", "show", "give", "top", "find", "why",
            "how", "do", "does", "did", "review", "reviews", "customer", "customers"
        }
        search_terms: set[str] = set()

        for w in raw_words:
            if w not in stop_words and len(w) > 2:
                search_terms.add(w)
                if w in CROSS_LANG_EXPANSIONS:
                    search_terms.update(CROSS_LANG_EXPANSIONS[w])

        if not search_terms:
            search_terms = {w for w in raw_words if len(w) > 1} or {"reclamacao"}

        # Sanitize terms for SQL literal ILIKE matching
        safe_terms = [re.sub(r"[';\-]", "", t).strip() for t in search_terms if t.strip()]

        try:
            active_tables = [t.upper() for t in self.adapter.list_tables()]
        except Exception as e:
            logger.warning(f"Could not list Snowflake tables: {e}")
            active_tables = ["OLIST_ORDER_REVIEWS"]

        chunks: list[Chunk] = []

        if "OLIST_ORDER_REVIEWS" in active_tables:
            like_conditions = []
            for t in safe_terms[:8]:
                like_conditions.append(f"REVIEW_COMMENT_MESSAGE ILIKE '%{t}%'")
                like_conditions.append(f"REVIEW_COMMENT_TITLE ILIKE '%{t}%'")

            where_clause = " OR ".join(like_conditions) if like_conditions else "1=1"

            extra_filters = []
            if filters and "review_score" in filters:
                extra_filters.append(f"REVIEW_SCORE = {int(filters['review_score'])}")
            elif any(w in query.lower() for w in ["1-star", "negative", "bad", "worst", "complaint", "dissatisfied"]):
                extra_filters.append("REVIEW_SCORE <= 2")
            elif any(w in query.lower() for w in ["5-star", "positive", "good", "praise", "great", "best"]):
                extra_filters.append("REVIEW_SCORE >= 4")

            filter_str = f" AND ({' AND '.join(extra_filters)})" if extra_filters else ""

            sql = f"""
            SELECT REVIEW_ID, ORDER_ID, REVIEW_SCORE, REVIEW_COMMENT_TITLE, REVIEW_COMMENT_MESSAGE
            FROM OLIST_ORDER_REVIEWS
            WHERE REVIEW_COMMENT_MESSAGE IS NOT NULL
              AND ({where_clause})
              {filter_str}
            LIMIT {k}
            """
            res = self.adapter.execute(sql)
            for r in res.rows:
                rev_id = str(r[0])
                ord_id = str(r[1]) if r[1] else ""
                score = r[2]
                title = str(r[3]) if r[3] else ""
                msg = str(r[4]) if r[4] else ""

                content = f"Review ID: {rev_id} (Score: {score}/5)\n"
                if title:
                    content += f"Title: {title}\n"
                content += f"Customer Comment: {msg}"

                chunks.append(
                    Chunk(
                        chunk_id=f"D:OLIST_ORDER_REVIEWS:{rev_id}",
                        doc_name="OLIST_ORDER_REVIEWS",
                        text=content,
                        metadata={
                            "review_id": rev_id,
                            "order_id": ord_id,
                            "score": score,
                            "snowflake_query_id": res.query_id,
                        },
                        score=1.0,
                    )
                )

        elif "SUPPORT_TICKETS" in active_tables:
            like_conditions = [f"body ILIKE '%{t}%'" for t in safe_terms[:8]]
            where_clause = " OR ".join(like_conditions) if like_conditions else "1=1"
            sql = f"""
            SELECT ticket_id, created_at, product_name, category, priority, body
            FROM SUPPORT_TICKETS
            WHERE {where_clause}
            LIMIT {k}
            """
            res = self.adapter.execute(sql)
            for r in res.rows:
                tid = str(r[0])
                created = str(r[1])
                prod = str(r[2])
                cat = str(r[3])
                prio = str(r[4])
                body = str(r[5])
                content = (
                    f"Ticket {tid} (Date: {created})\n"
                    f"Product: {prod}\n"
                    f"Category: {cat} | Priority: {prio}\n"
                    f"Details: {body}"
                )
                chunks.append(
                    Chunk(
                        chunk_id=f"D:SUPPORT_TICKETS:{tid}",
                        doc_name="SUPPORT_TICKETS",
                        text=content,
                        metadata={"ticket_id": tid, "product": prod, "snowflake_query_id": res.query_id},
                        score=1.0,
                    )
                )

        return chunks

    def close(self) -> None:
        """Close retriever resources (adapter connection lifecycle managed by caller)."""
        pass
