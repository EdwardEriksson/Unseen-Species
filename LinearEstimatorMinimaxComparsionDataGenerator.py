#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_rmse_for_plot.py — Generate CSVs for your 2×3 plot using a single, uniform convention.

What we plot (for every estimator except 'their analysis'):
  outer_RMSE(H) = sqrt(Phi_min(H)) where Phi_min(H) ≡ A(H)^2 + B(H).
  With the softmax solver’s HARD checker:
    A = sup_x t e^{-x}/x * |1 - e^{-r x} - g_H(x)|
    B = sup_y t e^{-y}/y * ( 1 - e^{-r y} + g_{H^2}(y) )
    F = A^2 + B  =>  RMSE = sqrt(F)

Pipeline per estimator:
  1) Produce H (STAR via softmax solver, HO via closed-form formula, ZERO as all-zeros of length N).
  2) Score H with solver.evaluate_objective_hard(...) and take sqrt(A^2 + B).
  3) Write sqrt_total_* to CSV.

Their analysis (closed form):
  - You supplied an MSE f(r,t); we output sqrt_f = sqrt( f(r,t) ).

Outputs:
  - rmse_vs_t.csv : rows with mode="vs_t" (fixed r, varying t log-spaced)
  - rmse_vs_r.csv : rows with mode="vs_r" (fixed t, varying r linearly)

Columns (exactly as your plotting code reads):
  mode, t, r, sqrt_total_star, sqrt_total_HO, sqrt_f, sqrt_g
"""

import argparse
import csv
import math
import warnings
from typing import Iterable, Optional
import QSIP_solver_neo as solver
from cutting_plane_qsip import solve_qsip_cutting_plane

import numpy as np
from scipy.stats import binom

# -------------------- Utilities --------------------

def log_base_3(x: float) -> float:
    return float(np.log(x) / np.log(3.0))


def pad_or_truncate(H: np.ndarray, N: int) -> np.ndarray:
    """Ensure H is exactly length N (pad with zeros or truncate)."""
    H = np.asarray(H, dtype=float).ravel()
    if H.size < N:
        return np.pad(H, (0, N - H.size), mode="constant")
    if H.size > N:
        return H[:N].copy()
    return H


# -------------------- Their analysis: MSE -> sqrt --------------------

def their_mse(t: float, r: float) -> float:
    """
    Your closed-form MSE:
        rt = r*t
        log3 = log_3(1 + 2/r)
        L = ( 1/(r-1) + ((r+2)^2 (r+1)^2)/r^4 ) * ((r-1)/r^2)^log3 + 1/r
        f(r,t) = L * (rt)^(2 - log3)
    Defined for r > 1. Returns NaN otherwise.
    """
    if r <= 1.0:
        return float("nan")
    rt = r * t
    log3 = log_base_3(1.0 + 2.0 / r)
    try:
        L = ((1.0 / (r - 1.0)) + ((r + 2.0) ** 2 * (r + 1.0) ** 2) / (r ** 4)) \
            * ((r - 1.0) / (r ** 2)) ** log3 + (1.0 / r)
        mse = L * (rt ** (2.0 - log3))
        return float(mse) if mse >= 0 else float("nan")
    except Exception:
        return float("nan")


def their_rmse(t: float, r: float) -> float:
    mse = their_mse(t, r)
    return float(math.sqrt(mse)) if (np.isfinite(mse) and mse >= 0) else float("nan")


# -------------------- Reference H_O (length N) --------------------

def H_O_reference(r: float, t: int, N: int) -> np.ndarray:
    """
    H_i = -(-r)^i * P(L >= i), for i >= 1. (Index 0 stores H_1.)
    L ~ Binomial(k, q),  k = floor(0.5 * log_3(t*r^2/(r-1))),  q = 2/(r+2).
    Returns a length-N vector (padded with zeros if k < N). Requires r > 1.
    """
    if r <= 1.0:
        raise ValueError("r must be > 1 for H_O to be defined.")
    base = t * (r ** 2) / (r - 1.0)
    k_val = int(math.floor(0.5 * log_base_3(base))) if base > 0 else 0
    if k_val < 0:
        k_val = 0
    q = 2.0 / (r + 2.0)

    H = np.zeros(max(N, k_val), dtype=float)
    # Note: H[0] corresponds to H_1 in notation
    for i in range(1, k_val + 1):
        tail_prob = binom.sf(i - 1, k_val, q)  # P(L >= i)
        H[i - 1] = - ((-r) ** i) * tail_prob
    return pad_or_truncate(H, N)


# -------------------- Uniform scoring: outer RMSE via HARD checker --------------------

def outer_rmse_from_solver(H: np.ndarray, t: int, r: float, N: int) -> float:
    """
    Score ANY H using the softmax solver's hard checker (dense geometric grids + boundary).
    Returns sqrt(A^2 + B).
    """
    H = pad_or_truncate(np.asarray(H, dtype=float), N)
    hard = solver.score_objective_hard(H, t, r)
    F = hard["F"]
    return math.sqrt(F) if (np.isfinite(F) and F >= 0) else float("nan")


# -------------------- Progress tracker --------------------

class Progress:
    def __init__(self, total: int):
        self.total = int(total)
        self.done = 0

    def tick(self):
        self.done += 1
        print(f"[progress] {self.done} / {self.total} solver calls completed")


# -------------------- STAR H producer (SOFTMAX solver) --------------------

def opt_H_via_solver(t: int, r: float, N: int, K: int,
                     progress: Optional[Progress] = None) -> np.ndarray:
    """
    Produce H* using the new QSIP_solver_neo adaptive L-BFGS-B solver.
    We call solve_qsip_adaptive_scipy, which returns H of length K+1 with H[0]=0.
    We then truncate/pad to length N for consistency with the rest of the pipeline.
    """

    try:
        # neo solver returns array length K+1; we want length N
        H_full = solve_qsip_cutting_plane(
            t=float(t),
            r=float(r),
            K=int(N)
        )

        if progress is not None:
            progress.tick()

        # Keep exactly N entries, ignoring H[0] convention
        return pad_or_truncate(H_full, N)

    except Exception as e:
        warnings.warn(f"NEO solve failed (t={t}, r={r}): {e}")
        if progress is not None:
            progress.tick()
        return np.zeros(N, dtype=float) * float("nan")


# -------------------- Row builder --------------------

def make_row(mode: str, t: int, r: float, N: int, K: int, progress: Optional[Progress]) -> dict:
    # 1) Our estimator (STAR): solve for H*, then score H* (outer RMSE)
    H_star = opt_H_via_solver(t, r, N, K, progress=progress)
    sqrt_total_star = outer_rmse_from_solver(H_star, t, r, N)

    # 2) SGT binomial (our analysis): build H_O (H_1..H_N), score with neo convention
    if r > 1.0:
        try:
            H_O = H_O_reference(r, t, N)           # length N: H_1..H_N
            H_full = np.zeros(N + 1, dtype=float)  # length N+1, H_0 = 0
            H_full[1:] = H_O
            hard = solver.score_objective_hard(H_full, t, r)
            F_HO = hard["F"]
            sqrt_total_HO = math.sqrt(F_HO) if (np.isfinite(F_HO) and F_HO >= 0) else float("nan")
        except Exception:
            sqrt_total_HO = float("nan")
    else:
        sqrt_total_HO = float("nan")



    # 3) Null estimator: H ≡ 0, score it the SAME way
    sqrt_g = outer_rmse_from_solver(np.zeros(N, dtype=float), t, r, N)

    # 4) "Their analysis": closed-form MSE → sqrt
    sqrt_f = their_rmse(t, r)

    return {
        "mode": "vs_t" if mode == "vs_t" else "vs_r",
        "t": float(t),
        "r": float(r),
        "sqrt_total_star": sqrt_total_star,
        "sqrt_total_HO":   sqrt_total_HO,
        "sqrt_f":          sqrt_f,
        "sqrt_g":          sqrt_g,
    }


# -------------------- Dumps --------------------

def dump_vs_t(outfile: str,
              N: int,
              K: int,
              rs: Iterable[float],
              t_vals: np.ndarray,
              progress: Optional[Progress]) -> None:
    """
    Top row: fixed r ∈ rs, vary t over provided t_vals (already log-spaced & unique).
    """
    fieldnames = ["mode", "t", "r", "sqrt_total_star", "sqrt_total_HO", "sqrt_f", "sqrt_g"]

    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rs:
            for t in t_vals:
                row = make_row("vs_t", int(t), float(r), N, K, progress)
                print(row)
                w.writerow(row)
    print(f"[saved] {outfile}")


def dump_vs_r(outfile: str,
              N: int,
              K: int,
              ts: Iterable[int],
              r_vals: np.ndarray,
              progress: Optional[Progress]) -> None:
    """
    Bottom row: fixed t ∈ ts, vary r over provided r_vals (already linear grid).
    """
    fieldnames = ["mode", "t", "r", "sqrt_total_star", "sqrt_total_HO", "sqrt_f", "sqrt_g"]

    with open(outfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in ts:
            for r in r_vals:
                row = make_row("vs_r", int(t), float(r), N, K, progress)
                w.writerow(row)
    print(f"[saved] {outfile}")


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate CSVs for 2×3 plot with uniform outer-loss scoring (softmax solver).")
    p.add_argument("--N", type=int, default=8, help="Truncation level for H (default: 8)")
    p.add_argument("--K", type=int, default=None, help="Ignored (kept for compatibility)")

    p.add_argument("--t-min", type=int, default=10, help="Min t for vs_t (default: 10)")
    p.add_argument("--t-max", type=int, default=10000, help="Max t for vs_t (default: 10000)")
    p.add_argument("--n-t", type=int, default=12, help="#points in t grid (default: 12)")
    p.add_argument("--rs", type=float, nargs=3, default=(1.5, 3.0, 6.0),
                   help="Three r values for the vs_t row (default: 1.5 3.0 6.0)")

    p.add_argument("--r-min", type=float, default=1.01, help="Min r for vs_r (default: 1.01)")
    p.add_argument("--r-max", type=float, default=6.0, help="Max r for vs_r (default: 6.0)")
    p.add_argument("--n-r", type=int, default=8, help="#points in r grid (default: 8)")
    p.add_argument("--ts", type=int, nargs=3, default=(100, 1000, 10000),
                   help="Three t values for the vs_r row (default: 100 1000 10000)")

    p.add_argument("--out-vs-t", type=str, default="rmse_vs_t.csv", help="Output CSV for vs_t")
    p.add_argument("--out-vs-r", type=str, default="rmse_vs_r.csv", help="Output CSV for vs_r")
    return p.parse_args()


def main():
    args = parse_args()

    # Build the grids
    t_vals = np.unique(np.logspace(np.log10(args.t_min), np.log10(args.t_max), args.n_t).astype(int))
    r_vals = np.linspace(args.r_min, args.r_max, args.n_r)

    # Compute total number of solver calls (one per row produced)
    total_solves = len(tuple(args.rs)) * len(t_vals) + len(tuple(args.ts)) * len(r_vals)
    progress = Progress(total_solves)
    print(f"[plan] Total solver calls planned: {total_solves}")

    # Generate CSVs with progress reporting
    dump_vs_t(outfile=args.out_vs_t,
              N=args.N, K=args.K,
              rs=tuple(args.rs),
              t_vals=t_vals,
              progress=progress)

    dump_vs_r(outfile=args.out_vs_r,
              N=args.N, K=args.K,
              ts=tuple(args.ts),
              r_vals=r_vals,
              progress=progress)

    # Final sanity: ensure we ticked exactly total_solves times
    if progress.done != progress.total:
        warnings.warn(f"Progress counter mismatch: counted {progress.done} of planned {progress.total} solves.")


if __name__ == "__main__":
    main()
