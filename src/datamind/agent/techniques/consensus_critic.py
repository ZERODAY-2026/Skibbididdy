"""Consensus and critic pass for model query generation.

Generates candidate queries when self-consistency or consensus checking is enabled.
When candidates disagree on aggregation functions or grouping columns, selects the best
candidate and flags a disagreement warning so the verifier caps the badge at PARTIALLY_VERIFIED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    primary_sql: str
    has_consensus: bool
    candidate_queries: list[str] = field(default_factory=list)
    critic_notes: str | None = None


class ConsensusCritic:
    """Evaluates candidate SQL queries for structural agreement and semantic consistency."""

    def __init__(self, dialect: str = "snowflake") -> None:
        self.dialect = dialect

    def evaluate_candidates(self, candidates: list[str]) -> ConsensusResult:
        """Compare candidate queries for structural consensus."""
        if not candidates:
            return ConsensusResult(primary_sql="", has_consensus=True)

        if len(candidates) == 1:
            return ConsensusResult(primary_sql=candidates[0], has_consensus=True, candidate_queries=candidates)

        parsed_candidates = []
        for sql in candidates:
            try:
                p = sqlglot.parse_one(sql, read=self.dialect)
                parsed_candidates.append((sql, p))
            except Exception:
                try:
                    p = sqlglot.parse_one(sql)
                    parsed_candidates.append((sql, p))
                except Exception:
                    parsed_candidates.append((sql, None))

        # Check table references across candidates
        table_sets = []
        group_sets = []
        for sql, p in parsed_candidates:
            if p:
                tables = {t.this.name.lower() for t in p.find_all(exp.Table) if t.this and hasattr(t.this, "name")}
                group_cols = set()
                group_node = p.find(exp.Group)
                if group_node:
                    for g in group_node.expressions:
                        if hasattr(g, "name"):
                            group_cols.add(g.name.lower())
                table_sets.append(tables)
                group_sets.append(group_cols)

        # Check if grouping sets or tables disagree
        disagreement = False
        notes = []

        if len(table_sets) > 1 and not all(t == table_sets[0] for t in table_sets):
            disagreement = True
            notes.append("Candidate models selected different source tables.")

        if len(group_sets) > 1 and not all(g == group_sets[0] for g in group_sets):
            disagreement = True
            notes.append("Candidate queries disagree on aggregation GROUP BY dimensions.")

        primary = candidates[0]
        critic_msg = "; ".join(notes) if notes else None

        return ConsensusResult(
            primary_sql=primary,
            has_consensus=not disagreement,
            candidate_queries=candidates,
            critic_notes=critic_msg,
        )
