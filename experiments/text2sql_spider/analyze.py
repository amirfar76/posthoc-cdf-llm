from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np
import yaml
import matplotlib.pyplot as plt

from src.posthoc_bands import eps_sqrt_calibrator, cdf_band_from_samples


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _slug_model_name(name: str) -> str:
    # match how generate.py writes folder names (it used model.replace("/", "__"))
    return name.replace("/", "__")


def resolve_run_dir(cfg: Dict[str, Any]) -> str:
    """
    Prefer deterministic run_dir based on config.
    If missing or not found, fallback to auto-detecting newest matching run directory.
    """
    out_root = cfg["run"]["out_dir"]
    run_name = cfg["run"]["name"]

    llm = cfg.get("llm", {})
    model_name = llm.get("model_name")
    temp = llm.get("temperature", None)

    # 1) Deterministic (new local HF path): runs/<run_name>_<model_slug>
    if model_name:
        cand = os.path.join(out_root, f"{run_name}_{_slug_model_name(model_name)}")
        if os.path.isdir(cand):
            return cand

    # 2) Old OpenAI-like path (if user kept it): runs/<run_name>_<MODELENV>_tempX
    # Try to reconstruct if model_env exists
    model_env = llm.get("model_env")
    if model_env:
        env_val = os.environ.get(model_env)
        if env_val:
            suffix = f"{run_name}_{env_val}"
            if temp is not None:
                suffix += f"_temp{temp}"
            cand = os.path.join(out_root, suffix)
            if os.path.isdir(cand):
                return cand

    # 3) Fallback: find any directory starting with runs/<run_name>_*
    pattern = os.path.join(out_root, f"{run_name}_*")
    cands = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not cands:
        raise FileNotFoundError(
            f"[error] Could not find any run directories matching: {pattern}\n"
            f"Expected something like:\n"
            f"  {out_root}/{run_name}_<model>\n"
            f"or (older format)\n"
            f"  {out_root}/{run_name}_<model>_temp<temperature>\n"
        )

    # choose most recently modified directory (likely your latest run)
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]


def ecdf(samples: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    s = np.sort(samples)
    return np.searchsorted(s, xgrid, side="right") / max(len(s), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = resolve_run_dir(cfg)
    mat_path = os.path.join(run_dir, "materialized_results.jsonl")
    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"[error] materialized_results.jsonl not found at:\n  {mat_path}\n"
            f"Run directory resolved to:\n  {run_dir}\n"
            f"Contents:\n  {sorted(os.listdir(run_dir))}\n"
        )

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    rows = load_jsonl(mat_path)

    strategies = cfg["strategies"]
    K = len(strategies)
    strat_names = [s["name"] for s in strategies]

    if len(rows) % K != 0:
        raise ValueError(
            f"[error] materialized results length {len(rows)} is not divisible by K={K}.\n"
            f"This usually means generate.py changed output ordering or crashed mid-write.\n"
            f"Try re-running generate, or delete {mat_path} and regenerate."
        )

    n_ex = len(rows) // K
    scores = np.zeros((n_ex, K), dtype=float)
    for i in range(n_ex):
        for j in range(K):
            scores[i, j] = float(rows[i * K + j]["score"])

    rng = np.random.default_rng(int(cfg["run"].get("seed", 0)))

    M = int(cfg["selection"]["M"])
    delta = float(cfg["selection"]["delta"])
    R = int(cfg["selection"]["replications"])
    cal_frac = float(cfg["selection"]["cal_frac"])

    n_cal = int(np.floor(cal_frac * n_ex))
    n_hol = n_ex - n_cal
    if n_cal <= 1 or n_hol <= 1:
        raise ValueError(
            f"[error] Too few examples for split: n_ex={n_ex}, n_cal={n_cal}, n_hol={n_hol}.\n"
            f"Increase data.max_examples in configs/spider_run.yaml."
        )

    fcps: List[float] = []
    example_band_made = False

    make_example = bool(cfg.get("plots", {}).get("make_example_band_plot", True))
    example_rank = int(cfg.get("plots", {}).get("example_strategy_rank", 1)) - 1
    example_rank = max(0, min(example_rank, M - 1))

    for _ in range(R):
        idx = rng.permutation(n_ex)
        cal_idx = idx[:n_cal]
        hol_idx = idx[n_cal:]

        cal_scores = scores[cal_idx, :]
        hol_scores = scores[hol_idx, :]

        means = cal_scores.mean(axis=0)
        sel = np.argsort(-means)[:M]  # top M by mean reward
        S = len(sel)

        failures = 0
        eps = eps_sqrt_calibrator(n=n_cal, K=K, S=S, delta=delta)

        for rank_in_sel, j in enumerate(sel):
            samp_cal = cal_scores[:, j]
            x, L, U = cdf_band_from_samples(samp_cal, eps)

            F_hold = ecdf(hol_scores[:, j], x)
            ok = np.all((F_hold >= L) & (F_hold <= U))
            if not ok:
                failures += 1

            # Make one representative CDF-band plot once
            if make_example and (not example_band_made) and rank_in_sel == example_rank:
                Fhat = np.arange(1, len(x) + 1) / len(x)

                plt.figure()
                plt.step(x, Fhat, where="post", label="ECDF (cal)")
                plt.step(x, L, where="post", label="Lower band")
                plt.step(x, U, where="post", label="Upper band")
                plt.step(x, F_hold, where="post", label="ECDF (holdout)")
                plt.xlabel("reward")
                plt.ylabel("CDF")
                plt.title(f"Post-selection CDF band (strategy={strat_names[j]})")
                plt.legend()
                plt.savefig(
                    os.path.join(plot_dir, f"cdf_band_example_{strat_names[j]}.pdf"),
                    bbox_inches="tight",
                )
                plt.close()
                example_band_made = True

        fcps.append(failures / max(S, 1))

    fcps_np = np.asarray(fcps, dtype=float)
    fcr = float(fcps_np.mean())

    # Plot FCP histogram
    plt.figure()
    plt.hist(fcps_np, bins=30)
    plt.axvline(float(delta), linestyle="--")
    plt.xlabel("FCP")
    plt.ylabel("count")
    plt.title(f"FCP over replications (mean={fcr:.3f}, target δ={delta})")
    plt.savefig(os.path.join(plot_dir, "fcp_hist.pdf"), bbox_inches="tight")
    plt.close()

    # Save summary
    summary = {
        "FCR": fcr,
        "delta": delta,
        "R": R,
        "n_examples": int(n_ex),
        "n_cal": int(n_cal),
        "n_holdout": int(n_hol),
        "run_dir": run_dir,
        "K": K,
        "M": M,
        "selected_example_plot_rank": int(example_rank + 1),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] analyze done. FCR={fcr:.4f}")
    print(f"[ok] outputs in: {run_dir}")
    print(f"[ok] plots in:   {plot_dir}")


if __name__ == "__main__":
    main()
