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


def resolve_run_dir(run_name: str, out_root: str) -> str:
    """
    Pick the newest directory matching runs/<run_name>_* that contains materialized_results.jsonl.
    """
    pattern = os.path.join(out_root, f"{run_name}_*")
    cands = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

    for d in cands:
        if os.path.exists(os.path.join(d, "materialized_results.jsonl")):
            return d

    # fallback: maybe someone used exactly run_name without suffix
    direct = os.path.join(out_root, run_name)
    if os.path.exists(os.path.join(direct, "materialized_results.jsonl")):
        return direct

    raise FileNotFoundError(
        "[error] Could not find materialized_results.jsonl for run.\n"
        f"Searched pattern:\n  {pattern}\n"
        f"Candidates found:\n  {cands[:10]}\n"
    )


def ecdf(samples: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    s = np.sort(samples)
    return np.searchsorted(s, xgrid, side="right") / len(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_name = cfg["run"]["name"]
    out_root = cfg["run"]["out_dir"]

    run_dir = resolve_run_dir(run_name, out_root)
    mat_path = os.path.join(run_dir, "materialized_results.jsonl")
    rows = load_jsonl(mat_path)

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    strategies = cfg["strategies"]
    K = len(strategies)
    strat_names = [s["name"] for s in strategies]

    if len(rows) % K != 0:
        raise ValueError(
            f"[error] materialized_results rows ({len(rows)}) not divisible by K ({K}). "
            "This usually means the materialization order changed or some rows are missing."
        )

    n_ex = len(rows) // K
    scores = np.zeros((n_ex, K), dtype=float)
    for i in range(n_ex):
        for j in range(K):
            scores[i, j] = float(rows[i * K + j]["score"])

    seed = int(cfg["run"].get("seed", 0))
    rng = np.random.default_rng(seed)

    M = int(cfg["selection"]["M"])
    delta = float(cfg["selection"]["delta"])
    R = int(cfg["selection"]["replications"])
    cal_frac = float(cfg["selection"]["cal_frac"])
    n_cal = int(np.floor(cal_frac * n_ex))
    n_hol = n_ex - n_cal
    if n_cal <= 5 or n_hol <= 5:
        raise ValueError(f"[error] Too few examples for split: n_ex={n_ex}, n_cal={n_cal}, n_hol={n_hol}")

    fcps: List[float] = []
    example_band_made = False
    example_rank = int(cfg.get("plots", {}).get("example_strategy_rank", 1)) - 1
    example_rank = max(0, min(example_rank, M - 1))

    for r in range(R):
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

            if (not example_band_made) and cfg.get("plots", {}).get("make_example_band_plot", True):
                if j == sel[example_rank]:
                    plt.figure()
                    Fhat = np.arange(1, len(x) + 1) / len(x)
                    plt.step(x, Fhat, where="post", label="ECDF (cal)")
                    plt.step(x, L, where="post", label="Lower")
                    plt.step(x, U, where="post", label="Upper")
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

    plt.figure()
    plt.hist(fcps_arr, bins=30)
    plt.axvline(delta, linestyle="--")
    plt.xlabel("FCP")
    plt.ylabel("count")
    plt.title(f"FCP over replications (mean={fcr:.3f}, target δ={delta})")
    plt.savefig(os.path.join(plot_dir, "fcp_hist.pdf"), bbox_inches="tight")
    plt.close()

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "FCR": fcr,
                "delta": delta,
                "R": R,
                "n_examples": n_ex,
                "n_cal": n_cal,
                "n_hol": n_hol,
                "strategies": strat_names,
            },
            f,
            indent=2,
        )

    print(f"[ok] analyze done. FCR={fcr:.4f}")
    print(f"[ok] run_dir: {run_dir}")
    print(f"[ok] plots: {plot_dir}")


if __name__ == "__main__":
    main()
