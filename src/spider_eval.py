from __future__ import annotations

import re
import sqlite3
from typing import Any, List, Tuple


_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def clean_sql(text: str) -> str:
    """
    Extract a single SQL statement from model output.

    Handles:
      - ```sql ... ```
      - leading 'SQL:' prefixes
      - extra commentary before/after
      - multiple statements -> keep the first
    """
    if text is None:
        return ""
    t = text.strip()

    # Prefer content inside ```sql ... ```
    m = _CODE_FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    # Remove leading "SQL:" or similar
    t = re.sub(r"^\s*(sql\s*:\s*)", "", t, flags=re.IGNORECASE).strip()

    # Keep the first non-empty statement (Spider is typically single-statement)
    parts = [p.strip() for p in t.split(";") if p.strip()]
    if not parts:
        return t.strip()
    return parts[0] + ";"


def _fetch_all(db_path: str, sql: str) -> Tuple[bool, List[Tuple[Any, ...]]]:
    con = None
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        out = [tuple(r) for r in rows]
        return True, out
    except Exception:
        return False, []
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass


def _bag(rows: List[Tuple[Any, ...]]) -> dict:
    """Multiset / bag representation of result rows."""
    d: dict = {}
    for r in rows:
        d[r] = d.get(r, 0) + 1
    return d


def execution_accuracy(db_path: str, pred_sql: str, gold_sql: str) -> float:
    """
    Binary reward in {0,1} based on exact match of result sets (order-insensitive).
    """
    pred_sql = clean_sql(pred_sql)
    gold_sql = clean_sql(gold_sql)

    ok_p, res_p = _fetch_all(db_path, pred_sql)
    ok_g, res_g = _fetch_all(db_path, gold_sql)
    if not ok_g:
        return 0.0
    if not ok_p:
        return 0.0
    return 1.0 if sorted(res_p) == sorted(res_g) else 0.0


def execution_f1(db_path: str, pred_sql: str, gold_sql: str) -> float:
    """
    Continuous reward in [0,1] based on overlap between result *multisets*.

    - If pred query fails to execute -> 0
    - If gold fails (shouldn't) -> 0
    - Otherwise: F1 = 2 * prec * rec / (prec + rec)

    Precision/recall computed from multiset intersection sizes.
    """
    pred_sql = clean_sql(pred_sql)
    gold_sql = clean_sql(gold_sql)

    ok_p, res_p = _fetch_all(db_path, pred_sql)
    ok_g, res_g = _fetch_all(db_path, gold_sql)

    if not ok_g:
        return 0.0
    if not ok_p:
        return 0.0

    bag_p = _bag(res_p)
    bag_g = _bag(res_g)

    inter = 0
    for k, vp in bag_p.items():
        vg = bag_g.get(k, 0)
        inter += min(vp, vg)

    total_p = len(res_p)
    total_g = len(res_g)

    # Empty result edge-cases
    if total_p == 0 and total_g == 0:
        return 1.0
    if total_p == 0 or total_g == 0:
        return 0.0

    prec = inter / total_p
    rec = inter / total_g
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))
