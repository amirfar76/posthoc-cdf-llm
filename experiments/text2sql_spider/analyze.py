from __future__ import annotations
import argparse, os, json, random
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from src.posthoc_bands import eps_sqrt_calibrator, cdf_band_from_samples

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = os.environ.get(cfg["llm"]["model_env"], "model")
    run_id = f"{cfg['run']['name']}_{model}_temp{cfg['llm']['temperature']}"
    out_dir = os.path.join(cfg["run"]["out_dir"], run_id)
    os.makedirs(out_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # We re-read examples directly from cache materialization.
    mat_path = os.path.join(out_dir, "materialized_results.jsonl")
    rows = load_jsonl(mat_path)

    # Build a table keyed by (example_index, strategy_name)
    # NOTE: materialized rows are just outputs; so we need to reconstruct mapping.
    # In v1, easiest is to rebuild by regenerating the same iteration order:
    # Instead, we store “strategy” and “example_id” inside the cached object if you want.
    # For now: we parse materialized_results only, assumes same ordering is stable.

    # For simplicity: store results as list; we’ll infer K strategies from config and slice.
    strategies = cfg["strategies"]
    K = len(strategies)

    # We also need a stable list of examples length:
    # materialized rows are in blocks of K per example.
    assert len(rows) % K == 0, "materialized results length not divisible by num strategies"
    n_ex = len(rows) // K

    # scores[ex_idx, strat_idx]
    scores = np.zeros((n_ex, K), dtype=float)
    for i in range(n_ex):
        for j in range(K):
            scores[i, j] = float(rows[i*K + j]["score"])

    strat_names = [s["name"] for s in strategies]
    rng = np.random.default_rng(int(cfg["run"]["seed"]))

    M = int(cfg["selection"]["M"])
    delta = float(cfg["selection"]["delta"])
    R = int(cfg["selection"]["replications"])
    cal_frac = float(cfg["selection"]["cal_frac"])
    n_cal = int(np.floor(cal_frac * n_ex))
    n_hol = n_ex - n_cal

    def ecdf(samples: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
        s = np.sort(samples)
        return np.searchsorted(s, xgrid, side="right") / len(s)

    fcps = []
    example_band_made = False

    for r in range(R):
        idx = rng.permutation(n_ex)
        cal_idx = idx[:n_cal]
        hol_idx = idx[n_cal:]

        cal_scores = scores[cal_idx, :]
        hol_scores = scores[hol_idx, :]

        means = cal_scores.mean(axis=0)
        sel = np.argsort(-means)[:M]  # top M
        S = len(sel)

        # Build bands on calibration for selected strategies; test failure on holdout ECDF
        failures = 0
        for j in sel:
            samp_cal = cal_scores[:, j]
            eps = eps_sqrt_calibrator(n=n_cal, K=K, S=S, delta=delta)
            x, L, U = cdf_band_from_samples(samp_cal, eps)

            # Proxy “truth”: holdout ECDF evaluated at same x points
            F_hold = ecdf(hol_scores[:, j], x)

            ok = np.all((F_hold >= L) & (F_hold <= U))
            failures += (0 if ok else 1)

            if (not example_band_made) and cfg["plots"]["make_example_band_plot"]:
                # plot one representative band (best selected)
                if j == sel[int(cfg["plots"]["example_strategy_rank"]) - 1]:
                    plt.figure()
                    Fhat = np.arange(1, len(x)+1) / len(x)
                    plt.step(x, Fhat, where="post")
                    plt.step(x, L, where="post")
                    plt.step(x, U, where="post")
                    plt.step(x, F_hold, where="post")
                    plt.xlabel("reward")
                    plt.ylabel("CDF")
                    plt.title(f"Post-selection CDF band (strategy={strat_names[j]})")
                    plt.savefig(os.path.join(plot_dir, f"cdf_band_example_{strat_names[j]}.pdf"), bbox_inches="tight")
                    plt.close()
                    example_band_made = True

        fcp = failures / max(S, 1)
        fcps.append(fcp)

    fcps = np.array(fcps, dtype=float)
    fcr = fcps.mean()

    # Plot FCP distribution
    plt.figure()
    plt.hist(fcps, bins=30)
    plt.axvline(float(delta), linestyle="--")
    plt.xlabel("FCP")
    plt.ylabel("count")
    plt.title(f"FCP over replications (mean={fcr:.3f}, target δ={delta})")
    plt.savefig(os.path.join(plot_dir, "fcp_hist.pdf"), bbox_inches="tight")
    plt.close()

    # Save summary
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"FCR": float(fcr), "delta": delta, "R": R, "n_examples": n_ex, "n_cal": n_cal}, f, indent=2)

    print(f"[ok] analyze done. FCR={fcr:.4f}, outputs in {out_dir}")

if __name__ == "__main__":
    main()
