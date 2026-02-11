from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Any, List, Tuple


def _fetch_all(db_path: str, sql: str) -> Tuple[bool, List[Tuple[Any, ...]]]:
    con = None
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        # normalize rows as tuples (stable comparable objects)
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


def execution_accuracy(db_path: str, pred_sql: str, gold_sql: str) -> float:
    ok_p, res_p = _fetch_all(db_path, pred_sql)
    ok_g, res_g = _fetch_all(db_path, gold_sql)
    if not ok_g or not ok_p:
        return 0.0
    return 1.0 if sorted(res_p) == sorted(res_g) else 0.0


def execution_f1(db_path: str, pred_sql: str, gold_sql: str) -> float:
    """
    Continuous reward in [0,1]:
    - Execute predicted & gold SQL on the same sqlite DB.
    - Compute multiset F1 overlap between result rows.
    """
    ok_p, res_p = _fetch_all(db_path, pred_sql)
    ok_g, res_g = _fetch_all(db_path, gold_sql)

    if not ok_g:
        # should not happen in Spider; if it does, treat as 0
        return 0.0
    if not ok_p:
        return 0.0

    # Handle empty-gold edge case
    if len(res_g) == 0:
        return 1.0 if len(res_p) == 0 else 0.0

    cp = Counter(res_p)
    cg = Counter(res_g)
    inter = sum((cp & cg).values())
    if inter == 0:
        return 0.0

    prec = inter / max(1, sum(cp.values()))
    rec = inter / max(1, sum(cg.values()))
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))
