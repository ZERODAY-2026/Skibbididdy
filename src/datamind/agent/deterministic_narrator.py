"""Deterministic fallback narration renderer for DataMind (W5 / F9).

Generates strictly grounded, cited answer text directly from executed query rows
and retrieved document chunks when model narration fails verification.
"""

from __future__ import annotations

import decimal
from typing import Any
from datamind.agent.evidence import EvidenceStore


def render_deterministic_answer(
    question: str,
    evidence_store: EvidenceStore,
    currency_symbol: str = "R$",
) -> str:
    """Deterministically render verifiable answer with exact citations from evidence."""
    q_clean = question.strip().rstrip("?")
    q_lower = question.lower()

    # Determine currency / percentage formatting
    is_currency = any(k in q_lower for k in ["revenue", "freight", "aov", "value", "price", "sales", "paid"]) and "percentage" not in q_lower and "percent" not in q_lower
    is_percent = "percentage" in q_lower or "percent" in q_lower or "%" in q_lower
    sym = currency_symbol if is_currency else ""
    pct = "%" if is_percent else ""

    def fmt_val(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, (float, decimal.Decimal)):
            f_val = float(v)
            if sym:
                return f"{sym} {f_val:,.2f}"
            if pct:
                return f"{f_val:,.2f}%"
            return f"{f_val:,.2f}"
        if isinstance(v, int):
            if sym:
                return f"{sym} {v:,}"
            if pct:
                return f"{v}%"
            return f"{v:,}"
        return str(v)

    lines: list[str] = []

    if evidence_store.queries:
        last_q = evidence_store.queries[-1]
        q_label = last_q.query_label
        rows = last_q.result.rows if last_q.result else []

        if not rows:
            lines.append(f"Based on verified database records, no matching data was found for '{q_clean}'.")
        elif len(rows) == 1 and len(rows[0]) == 1:
            val_str = fmt_val(rows[0][0])
            r_id = last_q.row_ids[0] if last_q.row_ids else f"{q_label}:R1"
            lines.append(f"Based on verified data, the result for '{q_clean}' is {val_str} [[{r_id}]].")
        elif len(rows) == 1:
            # Single row with multiple columns (e.g. key-value)
            parts = []
            for col_idx, c in enumerate(rows[0]):
                col_name = last_q.result.columns[col_idx] if last_q.result and col_idx < len(last_q.result.columns) else f"Field {col_idx+1}"
                parts.append(f"**{col_name}**: {fmt_val(c)}")
            r_id = last_q.row_ids[0] if last_q.row_ids else f"{q_label}:R1"
            lines.append(f"Based on verified data, the result for '{q_clean}' is {' - '.join(parts)} [[{r_id}]].")
        else:
            # Multi-row tabular / ranking result
            lines.append(f"Based on verified data, the results for '{q_clean}' are:")
            for r_idx, r in enumerate(rows[:10]):
                r_id = last_q.row_ids[r_idx] if r_idx < len(last_q.row_ids) else f"{q_label}:R{r_idx+1}"
                cells = [fmt_val(c) for c in r]
                row_label = f"{r_idx+1}. {' | '.join(cells)}"
                lines.append(f"- {row_label} [[{r_id}]]")

    # If document chunks were retrieved (qualitative customer reviews or tickets)
    if evidence_store.chunks:
        lines.append("\nSupporting customer feedback excerpts:")
        for chunk in evidence_store.chunks[:3]:
            # Extract clean quote
            clean_text = " ".join(chunk.text.split())[:180]
            lines.append(f"- \"{clean_text}\" [[{chunk.chunk_id}]]")

    if not lines:
        return f"Based on verified database evidence, no data was returned for '{q_clean}'."

    return "\n".join(lines)
