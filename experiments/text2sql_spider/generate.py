# experiments/text2sql_spider/generate.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cache import DiskCache
from src.spider_eval import clean_sql, combined_reward

# Optional: new reward (Spider official component-level F1 in [0,1])
try:
    from src.spider_eval import spider_component_f1 as _spider_component_f1
except Exception:
    _spider_component_f1 = None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def stable_hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def gold_sql(ex: Dict[str, Any]) -> str:
    for k in ["query", "sql", "gold", "gold_sql"]:
        if k in ex and isinstance(ex[k], str) and ex[k].strip():
            return ex[k]
    raise ValueError("Missing gold SQL in dataset example")


def db_path(ex: Dict[str, Any], data_root: str) -> str:
    db_id = ex.get("db_id")
    if not db_id:
        raise ValueError("Missing db_id")
    cand = os.path.join(data_root, "database", db_id, f"{db_id}.sqlite")
    if os.path.exists(cand):
        return cand
    # Some variants may store explicit paths
    for k in ["db_path", "database_path", "sqlite_path"]:
        p = ex.get(k)
        if isinstance(p, str) and os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not locate sqlite db for db_id={db_id} (looked for {cand})")


def build_schema_map(tables_json_path: str) -> Dict[str, str]:
    """
    Build a schema text per db_id using Spider tables.json.
    This is crucial; otherwise the model hallucinates table names (singers vs singer).
    """
    if not os.path.exists(tables_json_path):
        print(f"[warn] tables.json not found at {tables_json_path}; schema will be empty (quality will be poor).")
        return {}

    with open(tables_json_path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    out: Dict[str, str] = {}
    for t in tables:
        db_id = t["db_id"]
        table_names = t.get("table_names_original", t.get("table_names", []))
        col_names = t.get("column_names_original", t.get("column_names", []))
        col_types = t.get("column_types", [])
        pks = set(t.get("primary_keys", []))
        fks = t.get("foreign_keys", [])

        # columns: list of [table_id, column_name]
        cols_by_table: Dict[int, List[str]] = {}
        for idx, pair in enumerate(col_names):
            table_id, col = pair
            if table_id == -1:
                continue
            ty = col_types[idx] if idx < len(col_types) else ""
            pk = " PK" if idx in pks else ""
            cols_by_table.setdefault(table_id, []).append(f"{col} ({ty}){pk}".strip())

        lines: List[str] = []
        lines.append("Database schema:")
        for tid, tname in enumerate(table_names):
            cols = cols_by_table.get(tid, [])
            if cols:
                lines.append(f"- {tname}: " + ", ".join(cols))
            else:
                lines.append(f"- {tname}")

        if fks:
            lines.append("Foreign keys:")
            # fks is list of [col_idx_1, col_idx_2]
            for a, b in fks:
                # map column idx -> (table, col)
                try:
                    ta, ca = col_names[a]
                    tb, cb = col_names[b]
                    if ta != -1 and tb != -1:
                        lines.append(f"- {table_names[ta]}.{ca} -> {table_names[tb]}.{cb}")
                except Exception:
                    continue

        out[db_id] = "\n".join(lines).strip()

    return out


SYSTEM_SQL = "You are a text-to-SQL system. Return ONLY a single SQL query. No markdown. No explanation."


def build_prompt(question: str, schema: str) -> str:
    # Qwen chat template will wrap it; we pass messages in a compatible form
    # but here we build the user content string.
    return f"Schema:\n{schema}\n\nQuestion:\n{question}\n\nSQL:"


@dataclass
class LocalLLM:
    model_id: str
    max_new_tokens: int
    device: str = "cuda"

    def __post_init__(self) -> None:
        self.tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    @torch.no_grad()
    def chat(self, messages: List[Dict[str, str]], *, do_sample: bool, temperature: float, top_p: float) -> str:
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=bool(do_sample),
        )
        # Transformers will error if temperature==0 and do_sample=True
        if do_sample:
            gen_kwargs["temperature"] = max(float(temperature), 1e-6)
            gen_kwargs["top_p"] = float(top_p)

        out = self.model.generate(**inputs, **gen_kwargs)
        return self.tok.decode(out[0], skip_special_tokens=True)


def effective_decoding_params(cfg: Dict[str, Any], strat: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strategy-level overrides take precedence over llm defaults.
    Returns: do_sample, temperature, top_p.
    """
    llm_cfg = cfg.get("llm", {})

    # defaults
    temperature = float(llm_cfg.get("temperature", 0.7))
    top_p = float(llm_cfg.get("top_p", 0.95))

    # overrides (per-strategy)
    if "temperature" in strat:
        temperature = float(strat["temperature"])
    if "top_p" in strat:
        top_p = float(strat["top_p"])

    # do_sample logic:
    # - single uses llm.do_sample_single unless overridden in strategy
    # - bon/iad are sampled by design unless overridden explicitly
    if "do_sample" in strat:
        do_sample = bool(strat["do_sample"])
    else:
        if strat.get("type") == "single":
            do_sample = bool(llm_cfg.get("do_sample_single", False))
        else:
            do_sample = True

    return {"do_sample": do_sample, "temperature": temperature, "top_p": top_p}


def compute_reward(cfg: Dict[str, Any], db: str, pred_sql: str, gold_sql_str: str) -> float:
    ver = cfg.get("verifier", {})
    reward_type = ver.get("reward_type", "combined")  # default keeps old behavior

    if reward_type == "spider_component_f1":
        if _spider_component_f1 is None:
            raise KeyError(
                "Config requests verifier.reward_type=spider_component_f1, but src.spider_eval "
                "does not export spider_component_f1. Either (i) add spider_component_f1 to "
                "src/spider_eval.py or (ii) remove reward_type and use verifier.w_exec."
            )
        return float(_spider_component_f1(db, pred_sql, gold_sql_str))

    # legacy combined reward
    w_exec = float(ver.get("w_exec", 0.7))
    return float(combined_reward(db, pred_sql, gold_sql_str, w_exec=w_exec))


def run_one(
    llm: LocalLLM,
    cache: DiskCache,
    ex: Dict[str, Any],
    ex_idx: int,
    strat: Dict[str, Any],
    cfg: Dict[str, Any],
    data_root: str,
    schema_map: Dict[str, str],
) -> Dict[str, Any]:
    db_id = ex.get("db_id", "")
    q = ex.get("question", "")
    gold = gold_sql(ex)
    db = db_path(ex, data_root)

    schema = schema_map.get(db_id, "")
    schema_h = stable_hash_str(schema) if schema else "no_schema"

    # decoding params (strategy overrides supported)
    dec = effective_decoding_params(cfg, strat)
    do_sample = bool(dec["do_sample"])
    temperature = float(dec["temperature"])
    top_p = float(dec["top_p"])

    # reward version for caching / provenance
    ver = cfg.get("verifier", {})
    reward_type = ver.get("reward_type", "combined")
    reward_version = "spider_component_f1" if reward_type == "spider_component_f1" else "v2_execf1_plus_tokf1"

    key = {
        "run_name": cfg["run"]["name"],
        "model_id": cfg["llm"]["model_id"],
        "max_new_tokens": int(cfg["llm"]["max_new_tokens"]),
        "do_sample": bool(do_sample),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "strategy": strat,
        "db_id": db_id,
        "schema_hash": schema_h,
        "question": q,
        "gold_hash": stable_hash_str(gold),
        "reward_version": reward_version,
    }

    cached = cache.get_json(key)
    if cached is not None:
        return cached

    def do_messages(sql_user: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_SQL},
            {"role": "user", "content": sql_user},
        ]

    typ = strat["type"]

    candidates: List[Dict[str, Any]] = []

    if typ == "single":
        raw = llm.chat(
            do_messages(build_prompt(q, schema)),
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        pred = clean_sql(raw)
        score = compute_reward(cfg, db, pred, gold)
        candidates.append({"raw": raw, "pred": pred, "score": score})
        best = max(candidates, key=lambda c: c["score"])

    elif typ == "bon":
        n = int(strat["n"])
        for _ in range(n):
            raw = llm.chat(
                do_messages(build_prompt(q, schema)),
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            pred = clean_sql(raw)
            score = compute_reward(cfg, db, pred, gold)
            candidates.append({"raw": raw, "pred": pred, "score": score})
        best = max(candidates, key=lambda c: c["score"])

    elif typ == "iad":
        t = int(strat["t"])
        # initial sample
        raw0 = llm.chat(
            do_messages(build_prompt(q, schema)),
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        pred0 = clean_sql(raw0)
        s0 = compute_reward(cfg, db, pred0, gold)
        candidates.append({"raw": raw0, "pred": pred0, "score": s0})

        best = max(candidates, key=lambda c: c["score"])
        worst = min(candidates, key=lambda c: c["score"])

        for _ in range(1, t):
            refine_user = (
                f"Schema:\n{schema}\n\nQuestion:\n{q}\n\n"
                f"Best attempt so far:\n{best['pred']}\n\n"
                f"Worst attempt so far:\n{worst['pred']}\n\n"
                "Improve upon the best SQL while avoiding mistakes from the worst. Return ONLY SQL."
            )
            raw = llm.chat(
                do_messages(refine_user),
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            pred = clean_sql(raw)
            score = compute_reward(cfg, db, pred, gold)
            candidates.append({"raw": raw, "pred": pred, "score": score})
            best = max(candidates, key=lambda c: c["score"])
            worst = min(candidates, key=lambda c: c["score"])

    else:
        raise ValueError(f"Unknown strategy type: {typ}")

    out = {
        "example_index": ex_idx,
        "strategy_name": str(strat["name"]),
        "db_id": db_id,
        "question": q,
        "gold_sql": clean_sql(gold),
        "best_pred_sql": best["pred"],
        "score": float(best["score"]),
        "candidates": candidates,
        "schema_hash": schema_h,
        "reward_version": reward_version,
    }

    cache.set_json(key, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data"]["data_root"]
    split = cfg["data"]["split"]
    jsonl = os.path.join(data_root, "validation.jsonl" if split == "dev" else f"{split}.jsonl")
    examples = load_jsonl(jsonl)

    max_ex = cfg["data"].get("max_examples")
    if max_ex:
        examples = examples[: int(max_ex)]

    # schema map (requires official Spider tables.json)
    tables_json = os.path.join(data_root, "tables.json")
    schema_map = build_schema_map(tables_json)

    llm = LocalLLM(
        model_id=cfg["llm"]["model_id"],
        max_new_tokens=int(cfg["llm"]["max_new_tokens"]),
    )

    cache = DiskCache(cfg["run"]["cache_dir"])
    os.makedirs(cfg["run"]["cache_dir"], exist_ok=True)

    # run id directory
    safe_model = cfg["llm"]["model_id"].replace("/", "__")
    run_id = f"{cfg['run']['name']}_{safe_model}"
    out_dir = os.path.join(cfg["run"]["out_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)

    strategies = cfg["strategies"]

    rows: List[Dict[str, Any]] = []
    total = len(examples) * len(strategies)

    print(f"[info] examples={len(examples)} strategies={len(strategies)} total_jobs={total}")
    print(f"[info] writing materialized results to {out_dir}/materialized_results.jsonl")

    for i, ex in enumerate(tqdm(examples, desc="Generating (cached)")):
        if "question" not in ex or not ex["question"]:
            continue
        for strat in strategies:
            r = run_one(llm, cache, ex, i, strat, cfg, data_root, schema_map)
            rows.append(r)

    mat_path = os.path.join(out_dir, "materialized_results.jsonl")
    with open(mat_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] generated + cached. materialized at {mat_path}")
    print(f"[ok] cuda: {torch.cuda.is_available()} | model: {cfg['llm']['model_id']}")


if __name__ == "__main__":
    main()
