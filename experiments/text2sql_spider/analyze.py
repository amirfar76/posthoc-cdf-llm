# experiments/text2sql_spider/analyze.py
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
            out.append(json.loads(line))
    return out


def newest_run_dir(runs_root: str, run_name_prefix: str) -> str:
    cands = sorted(glob.glob(os.path.join(runs_root, f"{run_name_prefix}_*")))
    cands = [d for d in cands if os.path.isdir(d)]
    if not cands:
        raise FileNotFoundError(f"[error] no run dirs under {runs_root} matching prefix {run_name_prefix}_*")
    # pick newest by mtime
    cands.sort(key=lambda d: os.path.getmtime(d))
    return cands[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    runs_root = cfg["run"]["out_dir"]
    run_name = cfg["run"]["name"]

    run_dir = newest_run_dir(runs_root, run_name)
    mat_path = os.path.join(run_dir, "materialized_results.jsonl")

    if not os.path.exists(mat_path):
        contents = os.listdir(run_dir) if os.path.isdir(run_dir) else []
        raise FileNotFoundError(
            "[error] materialized_results.jsonl not found at:\n"
            f"  {mat_path}\n"
            "Run directory resolved to:\n"
            f"  {run_dir}\n"
            f"Contents:\n  {contents}"
        )

    rows = load_jsonl(mat_path)

    # strategies from config
    strategies = cfg["strategies"]
    strat_names = [s["name"] for s in strategies]
    K = len(strat_names)

    # infer number of examples from rows
    ex_indices = sorted({int(r["example_index"]) for r in rows})
    n_ex = len(ex_indices)
    ex_to_pos = {ex_idx: pos for pos, ex_idx in enumerate(ex_indices)}
    strat_to_pos = {name: j for j, name in enumerate(strat_names)}

    scores = np.full((n_ex, K), np.nan, dtype=float)

    for r in rows:
        i = ex_to_pos[int(r["example_index"])]
        sname = str(r["strategy_name"])
        j = strat_to_pos.get(sname)
        if j is None:
            continue
        scores[i, j] = float(r["score"])

    # drop examples with any missing scores
    mask = ~np.isnan(scores).any(axis=1)
    scores = scores[mask, :]
    n_ex = scores.shape[0]

    rng = np.random.default_rng(int(cfg["run"]["seed"]))

    M = int(cfg["selection"]["M"])
    delta = float(cfg["selection"]["delta"])
    R = int(cfg["selection"]["replications"])
    cal_frac = float(cfg["selection"]["cal_frac"])
    n_cal = int(np.floor(cal_frac * n_ex))
    n_hol = n_ex - n_cal

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    def ecdf(samples: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
        s = np.sort(samples)
        return np.searchsorted(s, xgrid, side="right") / len(s)

    fcps: List[float] = []
    example_band_made = False

    for _ in range(R):
        idx = rng.permutation(n_ex)
        cal_idx = idx[:n_cal]
        hol_idx = idx[n_cal:]

        cal_scores = scores[cal_idx, :]
        hol_scores = scores[hol_idx, :]

        means = cal_scores.mean(axis=0)
        sel = np.argsort(-means)[:M]
        S = len(sel)

        failures = 0
        for j in sel:
            samp_cal = cal_scores[:, j]
            eps = eps_sqrt_calibrator(n=n_cal, K=K, S=S, delta=delta)
            x, L, U = cdf_band_from_samples(samp_cal, eps)

            F_hold = ecdf(hol_scores[:, j], x)
            ok = np.all((F_hold >= L) & (F_hold <= U))
            failures += (0 if ok else 1)

            if (not example_band_made) and cfg["plots"]["make_example_band_plot"]:
                # choose the best selected (rank=1 means best)
                rank = int(cfg["plots"]["example_strategy_rank"]) - 1
                rank = max(0, min(rank, S - 1))
                if j == sel[rank]:
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
                    plt.savefig(os.path.join(plot_dir, f"cdf_band_example_{strat_names[j]}.pdf"), bbox_inches="tight")
                    plt.close()
                    example_band_made = True

        fcp = failures / max(S, 1)
        fcps.append(float(fcp))

    fcps_arr = np.array(fcps, dtype=float)
    fcr = float(fcps_arr.mean())

    # FCP histogram
    plt.figure()
    plt.hist(fcps_arr, bins=30)
    plt.axvline(float(delta), linestyle="--")
    plt.xlabel("FCP")
    plt.ylabel("count")
    plt.title(f"FCP over replications (mean={fcr:.3f}, target δ={delta})")
    plt.savefig(os.path.join(plot_dir, "fcp_hist.pdf"), bbox_inches="tight")
    plt.close()

    # summary
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "FCR": fcr,
                "delta": delta,
                "R": R,
                "n_examples_used": int(n_ex),
                "n_cal": int(n_cal),
                "n_holdout": int(n_hol),
            },
            f,
            indent=2,
        )

    print(f"[ok] analyze done. FCR={fcr:.4f}")
    print(f"[ok] plots in: {plot_dir}")


if __name__ == "__main__":
    main()
