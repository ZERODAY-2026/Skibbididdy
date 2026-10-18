"""Multi-turn conversational follow-up rewriter (W6 / F5).

Carries conversational state (last question, SQL plan, result summary)
and resolves contextual references ("that product", "same period", "only for SP",
"and for last year", "why") by rewriting follow-up turns into standalone questions.
"""

from __future__ import annotations

import re
from typing import Any
import logging

logger = logging.getLogger(__name__)


def rewrite_followup_question(
    question: str,
    conversation_history: list[dict[str, Any]],
    llm_client: Any | None = None,
) -> tuple[str, str | None]:
    """Rewrite contextual follow-up question into a standalone verifiable question.

    Returns:
        (rewritten_question, rationale)
    """
    if not conversation_history:
        return question, None

    # Extract the last user question and assistant answer
    last_user_q = ""
    last_assistant_a = ""
    for msg in reversed(conversation_history):
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "assistant" and not last_assistant_a:
            last_assistant_a = content
        elif role == "user" and not last_user_q:
            last_user_q = content
        if last_user_q and last_assistant_a:
            break

    if not last_user_q:
        return question, None

    q_lower = question.strip().lower()

    # Rule-based fast resolution for common follow-up patterns

    # 1. Month and Year substitution: e.g. "And what about December 2017?" or "How about in February?"
    month_match = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", q_lower)
    year_match = re.search(r"\b(2016|2017|2018|last year)\b", q_lower)

    if month_match and any(w in q_lower for w in ["and", "what about", "how about", "in", "for"]):
        target_month = month_match.group(1).capitalize()
        months_pat = r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b"
        if re.search(months_pat, last_user_q, re.IGNORECASE):
            rewritten = re.sub(months_pat, target_month, last_user_q, flags=re.IGNORECASE)
        else:
            rewritten = f"{last_user_q.rstrip('?')} in {target_month}?"
        if year_match:
            target_year = year_match.group(1)
            if target_year == "last year":
                target_year = "2016" if "2017" in last_user_q else "2017"
            if re.search(r"\b(2016|2017|2018)\b", rewritten):
                rewritten = re.sub(r"\b(2016|2017|2018)\b", target_year, rewritten)
            else:
                rewritten = f"{rewritten.rstrip('?')} {target_year}?"
        return rewritten, f"Substituted month '{target_month}' from prior context."

    # 1b. "and for last year" / "and in 2016" / "what about 2018"
    if year_match and any(w in q_lower for w in ["and", "what about", "how about", "for", "in"]):
        target_year = year_match.group(1)
        if target_year == "last year":
            target_year = "2016" if "2017" in last_user_q else "2017"
        if re.search(r"\b(2016|2017|2018)\b", last_user_q):
            rewritten = re.sub(r"\b(2016|2017|2018)\b", target_year, last_user_q)
            return rewritten, f"Substituted year '{target_year}' from previous context."
        else:
            rewritten = f"{last_user_q.rstrip('?')} in the year {target_year}?"
            return rewritten, f"Scoped query to year '{target_year}'."

    # 2. "only for SP" / "what about RJ?" / "filter by MG" / "based in SP"
    state_match = re.search(r"\b(sp|rj|mg|rs|pr|ba|sc|sao paulo|rio de janeiro)\b", q_lower)
    if state_match and any(w in q_lower for w in ["only", "what about", "and for", "how about", "in", "just", "based"]):
        state = state_match.group(1).upper()
        clean_state = "SP" if state in ("SP", "SAO PAULO") else ("RJ" if state in ("RJ", "RIO DE JANEIRO") else state)
        if "seller" in last_user_q.lower() or "sellers" in last_user_q.lower() or "seller" in q_lower or "sellers" in q_lower:
            rewritten = f"{last_user_q.rstrip('?')} for sellers located in state {clean_state}?"
        elif "state" in last_user_q.lower() or "states" in last_user_q.lower():
            rewritten = f"{last_user_q.rstrip('?')} filtered specifically for customer state '{clean_state}'?"
        else:
            rewritten = f"{last_user_q.rstrip('?')} for customers in state {clean_state}?"
        return rewritten, f"Filtered previous query for state {clean_state}."

    # 3. "that product" / "the top product" / "which one"
    if any(phrase in q_lower for phrase in ["that product", "the top product", "this product", "the first one"]):
        # Extract product name or ID from last assistant answer or last question
        prod_match = re.search(r"(\b[a-f0-9]{32}\b|\b[A-Z][a-zA-Z0-9_-]{3,}\b)", last_assistant_a)
        if prod_match:
            prod_name = prod_match.group(1)
            rewritten = re.sub(r"\b(that product|the top product|this product|the first one)\b", f"product '{prod_name}'", question, flags=re.IGNORECASE)
            return rewritten, f"Resolved entity reference to product '{prod_name}'."

    # 4. "why" / "why did it drop" / "what caused this"
    if q_lower.startswith("why") or "what caused" in q_lower or "reason for" in q_lower:
        rewritten = f"What customer complaints, outage reports, or review sentiment explain the trend for: {last_user_q.strip('?')}?"
        return rewritten, f"Expanded 'why' inquiry into qualitative customer review investigation."

    # 5. LLM-assisted rewriting for general multi-turn ambiguities if client available
    if llm_client and hasattr(llm_client, "config"):
        try:
            prompt = (
                f"You are a conversational query rewriter. "
                f"Previous user question: '{last_user_q}'. "
                f"Previous assistant answer summary: '{last_assistant_a[:200]}'. "
                f"Follow-up user question: '{question}'. "
                f"Rewrite the follow-up question into a complete, standalone question that can be answered independently. "
                f"Reply ONLY with the rewritten question."
            )
            res = llm_client.chat([{"role": "user", "content": prompt}], max_tokens=100, temperature=0.0)
            rewritten = res.get("content", "").strip().strip('"')
            if rewritten and len(rewritten) > 10:
                return rewritten, "Contextual resolution via model rewriter."
        except Exception as e:
            logger.debug("LLM rewrite fallback: %s", e)

    return question, None
