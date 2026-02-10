from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

SYSTEM_SQL = (
    "You are a text-to-SQL system. Return ONLY a single SQL query. "
    "No markdown. No explanation."
)

def build_prompt(question: str, schema: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_SQL},
        {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion:\n{question}\n\nSQL:"},
    ]

def build_refine_prompt(question: str, schema: str, best_sql: str, best_score: float, worst_sql: str, worst_score: float) -> List[Dict[str, str]]:
    # Light, cheap refinement prompt (no judge model needed).
    return [
        {"role": "system", "content": SYSTEM_SQL},
        {"role": "user", "content": (
            f"Schema:\n{schema}\n\nQuestion:\n{question}\n\n"
            f"Best attempt so far (score={best_score}):\n{best_sql}\n\n"
            f"Worst attempt so far (score={worst_score}):\n{worst_sql}\n\n"
            "Improve upon the best SQL while avoiding mistakes from the worst. Return ONLY SQL."
        )},
    ]

@dataclass
class Candidate:
    sql: str
    score: float

def select_best(cands: List[Candidate]) -> Candidate:
    return max(cands, key=lambda c: c.score)

def select_worst(cands: List[Candidate]) -> Candidate:
    return min(cands, key=lambda c: c.score)
