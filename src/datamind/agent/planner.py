"""Model-assisted and schema-grounded intent planner for DataMind.

Evaluates user questions against semantic catalog evidence to classify:
- ANSWERABLE: Clear metric, entity, or dimensional intent supported by schema.
- UNANSWERABLE: Queries requesting attributes, costs, or external entities absent from catalog.
- AMBIGUOUS: Underspecified queries lacking clear metrics, entities, or time bounds.

Uses schema inspection first, with fast model evaluation for borderline cases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from datamind.semantic.model import SemanticModel

logger = logging.getLogger(__name__)


@dataclass
class IntentClassification:
    category: Literal["ANSWERABLE", "UNANSWERABLE", "AMBIGUOUS"]
    reason: str
    missing_attributes: list[str] = field(default_factory=list)
    clarification_prompt: str | None = None


# Known attributes definitively absent from standard e-commerce and retail schemas
KNOWN_SCHEMA_BOUNDARIES = {
    "cogs": ("wholesale cost / cost of goods sold (COGS)", "net profit margin"),
    "profit": ("cost of goods sold (COGS)", "operating expense / profit calculation"),
    "salary": ("employee salary", "internal payroll data"),
    "compensation": ("employee compensation", "payroll records"),
    "cvv": ("unmasked CVV security code", "PCI-sensitive payment credentials"),
    "credit card number": ("unmasked full credit card number", "PCI-sensitive credentials"),
    "competitor": ("external competitor pricing", "third-party market data"),
    "amazon": ("Amazon Brazil competitor listings", "third-party competitor data"),
    "margin": ("net profit margin", "wholesale product procurement costs"),
    "warehouse gps": ("fulfillment center latitude/longitude", "warehouse facility coordinates"),
    "demographic": ("buyer age / demographic attributes", "customer survey profiling"),
}


def classify_question_intent(
    question: str,
    semantic_model: SemanticModel,
    llm_client: Any | None = None,
) -> IntentClassification:
    """Classify user question as ANSWERABLE, UNANSWERABLE, or AMBIGUOUS using schema evidence."""
    q_clean = question.strip()
    q_lower = q_clean.lower()

    # 1. Check for underspecified / ambiguous queries
    # These lack any metric, table entity, or dimension
    ambiguous_patterns = [
        r"\b(how are|how is)\s+(the numbers|numbers|things|business|everything|data|it|we doing)(\s+(looking|doing|going))?\b",
        r"\b(status update|performance update|overall situation|what's going on|what is going on)\b",
        r"\b(show me the numbers|show numbers|tell me about the data|how are our metrics)\b",
        r"\b(status update on everything|overview of recent trends|give me an overview)\b",
        r"\b(how are things going with the data)\b",
    ]
    for pat in ambiguous_patterns:
        if re.search(pat, q_lower):
            clarification = (
                "Could you please clarify which specific business metrics you would like to examine? "
                "For example: total delivered sales revenue, month-over-month revenue trends, "
                "average order value (AOV), or product category volume breakdowns?"
            )
            return IntentClassification(
                category="AMBIGUOUS",
                reason="Query is underspecified and lacks named metrics, entities, or time bounds.",
                clarification_prompt=clarification,
            )

    # 2. Check for schema boundary limitations (Unanswerable attributes)
    missing = []
    reasons = []
    for trigger, (attr_name, desc) in KNOWN_SCHEMA_BOUNDARIES.items():
        if re.search(rf"\b{re.escape(trigger)}\b", q_lower):
            # Check if this attribute exists in the semantic model tables
            all_cols = {col.name.lower() for table in semantic_model.tables for col in table.columns}
            if trigger not in all_cols and not any(trigger in col for col in all_cols):
                missing.append(attr_name)
                reasons.append(desc)

    if missing:
        msg = (
            f"I am unable to answer this question because the current database schema does not contain information "
            f"regarding {', '.join(missing)}. The available data is structured around verified transactional tables "
            f"({', '.join(t.name for t in semantic_model.tables)}). While I can calculate gross revenue, "
            f"order counts, and average order value, the database lacks {', '.join(reasons)}."
        )
        return IntentClassification(
            category="UNANSWERABLE",
            reason=f"Requested attributes ({', '.join(missing)}) are absent from database schema.",
            missing_attributes=missing,
            clarification_prompt=msg,
        )

    # 3. Model judgment for borderline queries if LLM client is available
    # Check if question has almost no overlapping tokens with any table, column, synonym, or metric
    schema_keywords = set()
    for t in semantic_model.tables:
        schema_keywords.add(t.name.lower())
        for c in t.columns:
            schema_keywords.add(c.name.lower())
    for m in semantic_model.metrics:
        schema_keywords.add(m.name.lower())

    q_tokens = set(re.findall(r"\b[a-z]{3,}\b", q_lower))
    overlap = q_tokens.intersection(schema_keywords)

    # If completely disjoint and question is very short or vague, ask LLM
    if not overlap and len(q_tokens) < 4 and llm_client and hasattr(llm_client, "config"):
        try:
            prompt = (
                f"You are a database schema boundary classifier. Given this schema: "
                f"Tables: {[t.name for t in semantic_model.tables]}. "
                f"Question: '{q_clean}'. "
                f"Classify intent as exactly one word: ANSWERABLE, UNANSWERABLE, or AMBIGUOUS. "
                f"Reply with that word only."
            )
            res = llm_client.chat([{"role": "user", "content": prompt}], max_tokens=10, temperature=0.0)
            verdict = res.get("content", "").strip().upper()
            if "AMBIGUOUS" in verdict:
                return IntentClassification(
                    category="AMBIGUOUS",
                    reason="Model classified query as underspecified.",
                    clarification_prompt="Could you please provide more context or specify which metric you'd like to analyze?",
                )
            if "UNANSWERABLE" in verdict:
                return IntentClassification(
                    category="UNANSWERABLE",
                    reason="Model classified query as exceeding schema boundary.",
                    clarification_prompt="This question cannot be answered using the available tables and columns in this catalog.",
                )
        except Exception as e:
            logger.debug("Model planner classification fallback: %s", e)

    # Default: answerable
    return IntentClassification(
        category="ANSWERABLE",
        reason="Query intent maps cleanly to available semantic model tables and metrics.",
    )
