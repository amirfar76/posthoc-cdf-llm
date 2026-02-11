# experiments/text2sql_spider/generate.py
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cache import DiskCache
from src.strategies import build_prompt, build_refine_prompt
from src.spider_eval import execution_accuracy, execution_f1

def run_one_strategy(llm, cache, ex, strat, cfg, data_root):
    verifier_type = cfg.get("verifier", {}).get("type", "execution_accuracy")

    key = {
        "run_name": cfg["run"]["name"],
        "model_id": cfg["llm"]["model_id"],  # (or whatever your generate.py uses for local model)
        "strategy": strat,
        "verifier_type": verifier_type,
        "db_id": ex.get("db_id"),
        "question": ex.get("question"),
        # optionally include schema string if you want stricter caching
    }

    cached = cache.get_json(key)
    if cached is not None:
        return cached

    q = ex["question"]
    sch = schema_string(ex)
    db = db_path(ex, data_root)
    gold = gold_sql(ex)

    if verifier_type == "execution_accuracy":
        def score_sql(sql: str) -> float:
            return execution_accuracy(db, sql, gold)
    elif verifier_type == "execution_f1":
        def score_sql(sql: str) -> float:
            return execution_f1(db, sql, gold)
    else:
        raise ValueError(f"Unknown verifier.type: {verifier_type}")

    # ---- keep the rest of your strategy logic the same ----
    # single / bon / iad should call score_sql(...) to score candidates
    # and store out = {"sql": ..., "score": ..., "candidates": ...}

    # finally:
    cache.set_json(key, out)
    return out



def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def schema_string(ex: Dict[str, Any]) -> str:
    # Best effort: dataset variants differ
    for k in ["schema", "db_schema", "schema_str", "create_table_statement"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v

    if "table_names_original" in ex and "column_names_original" in ex:
        return f"Tables: {ex['table_names_original']}\nColumns: {ex['column_names_original']}"

    return "Schema unavailable (dataset variant)."


def db_path(ex: Dict[str, Any], data_root: str) -> str:
    db_id = ex.get("db_id")
    if not db_id:
        raise ValueError("Missing db_id")
    cand = os.path.join(data_root, "database", db_id, f"{db_id}.sqlite")
    if os.path.exists(cand):
        return cand

    for k in ["db_path", "database_path", "sqlite_path"]:
        v = ex.get(k)
        if isinstance(v, str) and os.path.exists(v):
            return v

    raise FileNotFoundError(f"Could not locate sqlite db for db_id={db_id}")


def gold_sql(ex: Dict[str, Any]) -> str:
    for k in ["query", "sql", "gold", "gold_sql"]:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError("Missing gold SQL in dataset example")


_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """
    Make outputs robust:
      - If the model returns ```sql ...```, extract inside.
      - Strip leading role text, keep first SQL-ish chunk.
      - If multiple statements, keep everything (Spider gold sometimes uses ;).
    """
    t = text.strip()

    m = _SQL_FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()

    # Remove obvious chat headers if they appear in decoded text
    # e.g., "assistant\n..."; keep last portion if that happens
    for prefix in ["assistant\n", "assistant:", "Assistant:", "ASSISTANT:"]:
        if t.startswith(prefix):
            t = t[len(prefix) :].strip()

    # If still contains markdown fences (rare), strip them
    t = t.replace("```sql", "").replace("```", "").strip()

    # Some models prepend "SQL:" or similar
    if t.lower().startswith("sql:"):
        t = t[4:].strip()

    return t


class LocalHFChatLLM:
    """
    Minimal local “chat” wrapper using Transformers.
    Assumes the tokenizer supports apply_chat_template (Qwen2.5 does).
    """

    def __init__(self, model_id: str, dtype: torch.dtype = torch.float16):
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return self.model.device

    @torch.inference_mode()
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> str:
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(text, return_tensors="pt").to(self.device)

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
        )
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        out = self.model.generate(**inputs, **gen_kwargs)
        decoded = self.tok.decode(out[0], skip_special_tokens=True)
        return decoded


def run_one_strategy(
    llm: LocalHFChatLLM,
    cache: DiskCache,
    ex: Dict[str, Any],
    strat: Dict[str, Any],
    cfg: Dict[str, Any],
    data_root: str,
) -> Dict[str, Any]:
    # Stable identifiers
    ex_id = ex.get("question_id", ex.get("id", None))
    q = ex["question"]
    sch = schema_string(ex)
    db = db_path(ex, data_root)
    gold = gold_sql(ex)

    # Caching key: include the local model + decoding params
    # so changing them doesn’t reuse stale outputs.
    llm_cfg = cfg.get("llm", {})
    model_id = llm_cfg.get("local_model_id", "Qwen/Qwen2.5-7B-Instruct")
    max_tokens = int(llm_cfg.get("max_tokens", 512))
    temperature = float(llm_cfg.get("temperature", 0.7))
    top_p = float(llm_cfg.get("top_p", 0.9))

    key = {
        "run_name": cfg["run"]["name"],
        "backend": "local_transformers",
        "local_model_id": model_id,
        "strategy": strat,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "example_id": ex_id,
        "db_id": ex.get("db_id"),
        "question": q,
    }

    cached = cache.get_json(key)
    if cached is not None:
        return cached

    def score_sql(pred_sql: str) -> float:
        return float(execution_accuracy(db, pred_sql, gold))

    typ = strat["type"]

    if typ == "single":
        messages = build_prompt(q, sch)
        raw = llm.chat(
            messages,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
        )
        sql = extract_sql(raw)
        s = score_sql(sql)
        out = {
            "example_id": ex_id,
            "db_id": ex.get("db_id"),
            "strategy": strat.get("name", "single"),
            "sql": sql,
            "score": s,
            "candidates": [{"sql": sql, "score": s}],
        }

    elif typ == "bon":
        n = int(strat["n"])
        cands: List[Dict[str, Any]] = []
        best_sql: Optional[str] = None
        best_score: float = -1.0

        # For BON we sample to get diversity
        for _ in range(n):
            raw = llm.chat(
                build_prompt(q, sch),
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            sql = extract_sql(raw)
            s = score_sql(sql)
            cands.append({"sql": sql, "score": s})
            if s > best_score:
                best_score = s
                best_sql = sql

        assert best_sql is not None
        out = {
            "example_id": ex_id,
            "db_id": ex.get("db_id"),
            "strategy": strat.get("name", f"bon_{n}"),
            "sql": best_sql,
            "score": best_score,
            "candidates": cands,
        }

    elif typ == "iad":
        t = int(strat["t"])
        cands: List[Dict[str, Any]] = []

        # initial (sample to avoid getting stuck in same bad answer)
        raw0 = llm.chat(
            build_prompt(q, sch),
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        sql0 = extract_sql(raw0)
        s0 = score_sql(sql0)
        cands.append({"sql": sql0, "score": s0})

        def best_and_worst(cands_: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            best = max(cands_, key=lambda d: float(d["score"]))
            worst = min(cands_, key=lambda d: float(d["score"]))
            return best, worst

        best, worst = best_and_worst(cands)

        # refine steps
        for _ in range(1, t):
            raw = llm.chat(
                build_refine_prompt(
                    q, sch, best["sql"], float(best["score"]), worst["sql"], float(worst["score"])
                ),
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            sql = extract_sql(raw)
            s = score_sql(sql)
            cands.append({"sql": sql, "score": s})
            best, worst = best_and_worst(cands)

        out = {
            "example_id": ex_id,
            "db_id": ex.get("db_id"),
            "strategy": strat.get("name", f"iad_{t}"),
            "sql": best["sql"],
            "score": float(best["score"]),
            "candidates": cands,
        }

    else:
        raise ValueError(f"Unknown strategy type: {typ}")

    cache.set_json(key, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- paths (keep exactly as in your project)
    data_root = "data/spider"
    split = cfg["data"]["split"]
    jsonl = os.path.join(
        data_root, "validation.jsonl" if split == "dev" else f"{split}.jsonl"
    )

    examples = load_jsonl(jsonl)
    max_ex = cfg["data"].get("max_examples")
    if max_ex:
        examples = examples[: int(max_ex)]

    # --- local model config
    llm_cfg = cfg.get("llm", {})
    model_id = llm_cfg.get("local_model_id", "Qwen/Qwen2.5-7B-Instruct")

    # NOTE: dtype float16 is usually correct on Runpod GPUs
    llm = LocalHFChatLLM(model_id=model_id, dtype=torch.float16)

    # --- cache
    cache_dir = cfg["run"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    cache = DiskCache(cache_dir)

    strategies = cfg["strategies"]

    # --- output folder
    run_id = f"{cfg['run']['name']}_{model_id.replace('/', '__')}"
    out_dir = os.path.join(cfg["run"]["out_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)

    # --- run
    results: List[Dict[str, Any]] = []
    for ex in tqdm(examples, desc="Generating (cached)"):
        if "question" not in ex:
            continue
        for strat in strategies:
            r = run_one_strategy(llm, cache, ex, strat, cfg, data_root)
            results.append(r)

    mat_path = os.path.join(out_dir, "materialized_results.jsonl")
    with open(mat_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] generated + cached. materialized at {mat_path}")
    print(f"[ok] cuda: {torch.cuda.is_available()} | model: {model_id}")


if __name__ == "__main__":
    main()
