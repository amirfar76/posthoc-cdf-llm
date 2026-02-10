from __future__ import annotations
import argparse, asyncio, json, os, random
from typing import Any, Dict, List
import yaml
from tqdm.asyncio import tqdm_asyncio

from src.cache import DiskCache
from src.llm_client import LLMConfig, OpenAICompatClient
from src.strategies import build_prompt, build_refine_prompt, Candidate, select_best, select_worst
from src.spider_eval import execution_accuracy

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def get_env(cfg: dict, key: str) -> str:
    return os.environ[cfg[key]]

def schema_string(ex: Dict[str, Any]) -> str:
    # Best effort: many Spider dumps include a preformatted schema string; if not, fall back
    for k in ["schema", "db_schema", "schema_str", "create_table_statement"]:
        if k in ex and isinstance(ex[k], str) and ex[k].strip():
            return ex[k]
    # fallback: at least table names / columns if present
    if "table_names_original" in ex and "column_names_original" in ex:
        return f"Tables: {ex['table_names_original']}\nColumns: {ex['column_names_original']}"
    return "Schema unavailable (dataset variant)."

def db_path(ex: Dict[str, Any], data_root: str) -> str:
    # Many Spider formats store db_id and have /database/<db_id>/<db_id>.sqlite
    db_id = ex.get("db_id")
    if not db_id:
        raise ValueError("Missing db_id")
    cand = os.path.join(data_root, "database", db_id, f"{db_id}.sqlite")
    if os.path.exists(cand):
        return cand
    # Some HF variants store path explicitly
    for k in ["db_path", "database_path", "sqlite_path"]:
        if k in ex and os.path.exists(ex[k]):
            return ex[k]
    raise FileNotFoundError(f"Could not locate sqlite db for db_id={db_id}")

def gold_sql(ex: Dict[str, Any]) -> str:
    for k in ["query", "sql", "gold", "gold_sql"]:
        if k in ex and isinstance(ex[k], str):
            return ex[k]
    raise ValueError("Missing gold SQL in dataset example")

async def run_one_strategy(
    client: OpenAICompatClient,
    cache: DiskCache,
    ex: Dict[str, Any],
    strat: Dict[str, Any],
    cfg: Dict[str, Any],
    data_root: str,
) -> Dict[str, Any]:
    key = {
        "run_name": cfg["run"]["name"],
        "model": cfg["llm"]["model_env"],
        "temp": cfg["llm"]["temperature"],
        "max_tokens": cfg["llm"]["max_tokens"],
        "strategy": strat,
        "example_id": ex.get("question_id", ex.get("id", None)),
        "db_id": ex.get("db_id"),
        "question": ex.get("question"),
    }
    cached = cache.get_json(key)
    if cached is not None:
        return cached

    q = ex["question"]
    sch = schema_string(ex)
    db = db_path(ex, data_root)
    gold = gold_sql(ex)

    async def score_sql(sql: str) -> float:
        return execution_accuracy(db, sql, gold)

    typ = strat["type"]

    if typ == "single":
        msg = build_prompt(q, sch)
        sql = await client.chat(msg)
        s = await asyncio.to_thread(score_sql, sql)
        out = {"sql": sql, "score": s, "candidates": [{"sql": sql, "score": s}]}

    elif typ == "bon":
        n = int(strat["n"])
        cands: List[Candidate] = []
        for _ in range(n):
            sql = await client.chat(build_prompt(q, sch))
            s = await asyncio.to_thread(score_sql, sql)
            cands.append(Candidate(sql=sql, score=s))
        best = select_best(cands)
        out = {"sql": best.sql, "score": best.score,
               "candidates": [{"sql": c.sql, "score": c.score} for c in cands]}

    elif typ == "iad":
        t = int(strat["t"])
        cands: List[Candidate] = []
        # initial
        sql0 = await client.chat(build_prompt(q, sch))
        s0 = await asyncio.to_thread(score_sql, sql0)
        cands.append(Candidate(sql=sql0, score=s0))
        best = select_best(cands)
        worst = select_worst(cands)
        # refine steps
        for _ in range(1, t):
            sql = await client.chat(build_refine_prompt(q, sch, best.sql, best.score, worst.sql, worst.score))
            s = await asyncio.to_thread(score_sql, sql)
            cands.append(Candidate(sql=sql, score=s))
            best = select_best(cands)
            worst = select_worst(cands)
        out = {"sql": best.sql, "score": best.score,
               "candidates": [{"sql": c.sql, "score": c.score} for c in cands]}
    else:
        raise ValueError(f"Unknown strategy type: {typ}")

    cache.set_json(key, out)
    return out

async def main_async(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = "data/spider"
    split = cfg["data"]["split"]
    jsonl = os.path.join(data_root, "validation.jsonl" if split == "dev" else f"{split}.jsonl")
    examples = load_jsonl(jsonl)

    max_ex = cfg["data"].get("max_examples")
    if max_ex:
        examples = examples[:int(max_ex)]

    base_url = os.environ.get(cfg["llm"]["base_url_env"], "https://api.openai.com/v1")
    api_key = os.environ[cfg["llm"]["api_key_env"]]
    model = os.environ[cfg["llm"]["model_env"]]
    llm_cfg = LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=float(cfg["llm"]["temperature"]),
        max_tokens=int(cfg["llm"]["max_tokens"]),
        timeout_s=int(cfg["llm"]["request_timeout_s"]),
    )
    client = OpenAICompatClient(llm_cfg)

    cache = DiskCache(cfg["run"]["cache_dir"])
    os.makedirs(cfg["run"]["cache_dir"], exist_ok=True)

    strategies = cfg["strategies"]

    sem = asyncio.Semaphore(int(cfg["llm"]["concurrency"]))

    async def work(ex: Dict[str, Any], strat: Dict[str, Any]) -> Dict[str, Any]:
        async with sem:
            return await run_one_strategy(client, cache, ex, strat, cfg, data_root)

    tasks = []
    for ex in examples:
        if "question" not in ex:
            continue
        for strat in strategies:
            tasks.append(work(ex, strat))

    results = await tqdm_asyncio.gather(*tasks)

    # write a materialized table for analysis (but cache remains the source of truth)
    run_id = f"{cfg['run']['name']}_{model}_temp{cfg['llm']['temperature']}"
    out_dir = os.path.join(cfg["run"]["out_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "materialized_results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    await client.aclose()
    print(f"[ok] generated + cached. materialized at {out_dir}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args.config))

if __name__ == "__main__":
    main()
