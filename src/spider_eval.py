from __future__ import annotations
import sqlite3
from typing import Any, List, Tuple

def _fetch_all(db_path: str, sql: str) -> Tuple[bool, List[Tuple[Any, ...]]]:
    try:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        # normalize rows as tuples
        out = [tuple(r) for r in rows]
        return True, out
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        return False, []

def execution_accuracy(db_path: str, pred_sql: str, gold_sql: str) -> float:
    ok_p, res_p = _fetch_all(db_path, pred_sql)
    ok_g, res_g = _fetch_all(db_path, gold_sql)
    if not ok_g:
        # should not happen in Spider; if it does, mark as missing
        return 0.0
    if not ok_p:
        return 0.0
    # order-insensitive set compare
    return 1.0 if sorted(res_p) == sorted(res_g) else 0.0
