# src/spider_eval.py
from __future__ import annotations

import re
import sqlite3
from collections import Counter
from typing import Any, List, Tuple


_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START_RE = re.compile(r"(?is)\b(with|select|insert|update|delete)\b")


def clean_sql(text: str) -> str:
    """
    Extract a single SQL statement from messy model output.
    Handles:
      - chat-template text containing 'system/user/assistant'
      - fenced code blocks ```sql ... ```
      - extra commentary before/after SQL
    """
    if text is None:
        return ""
    t = text.strip()

    # Prefer fenced SQL
    m = _CODE_FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    # Find first SQL keyword
    m2 = _SQL_START_RE.search(t)
    if not m2:
        # best-effort: strip backticks and return
        return t.strip().strip("`").strip()

    t2 = t[m2.start():].strip()

    # Keep only first statement
    semi = t2.find(";")
    if semi != -1:
        stmt = t2[: semi + 1].strip()
    else:
        stmt = t2.strip()
        if not stmt.endswith(";"):
            stmt += ";"

    return stmt.replace("```", "").strip()


def _fetch_all(db_path: str, sql: str, max_rows: int = 2000) -> Tuple[bool, List[Tuple[Any, ...]]]:
    """
    Execute SQL and return rows as tuples.
    We cap rows to avoid huge transfers.
    """
    con = None
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchmany(max_rows)
        # normalize rows to tuples
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


def _f1_from_multisets(a: List[Tuple[Any, ...]], b: List[Tuple[Any, ...]]) -> float:
    """
    Multiset F1 on row tuples.
    """
    if not a and not b:
        return 1.0
    if not a and b:
        return 0.0
    if a and not b:
        return 0.0

    ca = Counter(a)
    cb = Counter(b)
    inter = ca & cb
    tp = sum(inter.values())
    pa = sum(ca.values())
    pb = sum(cb.values())

    prec = tp / pa if pa > 0 else 0.0
    rec = tp / pb if pb > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def execution_f1(db_path: str, pred_sql: str, gold_sql: str) -> float:
    """
    Fractional execution score: F1 overlap between result sets.
    If pred SQL fails to execute -> 0.
    If gold SQL fails (shouldn’t happen) -> 0.
    """
    pred_sql = clean_sql(pred_sql)
    gold_sql = clean_sql(gold_sql)

    ok_g, res_g = _fetch_all(db_path, gold_sql)
    if not ok_g:
        return 0.0

    ok_p, res_p = _fetch_all(db_path, pred_sql)
    if not ok_p:
        return 0.0

    return float(_f1_from_multisets(res_p, res_g))


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|<=|>=|!=|==|[-+*/(),.;=<>]|\d+|\S")


def sql_token_f1(pred_sql: str, gold_sql: str) -> float:
    """
    Token-level F1 between predicted and gold SQL (bag-of-tokens).
    This is almost always fractional → smooths the reward distribution.
    """
    p = clean_sql(pred_sql).lower()
    g = clean_sql(gold_sql).lower()

    tp = _TOKEN_RE.findall(p)
    tg = _TOKEN_RE.findall(g)

    if not tp and not tg:
        return 1.0
    if not tp or not tg:
        return 0.0

    cp = Counter(tp)
    cg = Counter(tg)
    inter = cp & cg
    num = sum(inter.values())
    den_p = sum(cp.values())
    den_g = sum(cg.values())

    prec = num / den_p if den_p > 0 else 0.0
    rec = num / den_g if den_g > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def combined_reward(db_path: str, pred_sql: str, gold_sql: str, w_exec: float = 0.7) -> float:
    """
    Final continuous reward used by the experiment.
    """
    exec_f = execution_f1(db_path, pred_sql, gold_sql)
    tok_f = sql_token_f1(pred_sql, gold_sql)
    w_exec = float(w_exec)
    w_exec = min(max(w_exec, 0.0), 1.0)
    return float(w_exec * exec_f + (1.0 - w_exec) * tok_f)
