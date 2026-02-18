#!/usr/bin/env python3
"""
qsip_ineq_sweep.py
==================
Compute a coarse truth grid for:
    H[2]^2 - 2 H[1]^2 > r^2 + 2r
where H = solve_qsip_adaptive_scipy(t, r).

Saves an .npz containing:
- r_values (1D)
- t_values (1D)
- truth   (2D float with {0,1} and NaN for failures), shape (len(t), len(r))
- failed  (2D bool), same shape
- meta    (dict-like object with a few strings)

Example:
    python qsip_ineq_sweep.py
    python qsip_ineq_sweep.py --out sweep.npz --rmin 1.5 --rmax 6 --nr 13 --tmin 1e2 --tmax 1e4 --nt 11
"""

import argparse
import time
import numpy as np

from QSIP_solver_neo import solve_qsip_adaptive_scipy

def check_ineq(H, r: float) -> bool:
    return (H[2] ** 2 - 2 * (H[1] ** 2)) > (r ** 2 + 2 * r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="qsip_ineq_sweep.npz")
    ap.add_argument("--rmin", type=float, default=2)
    ap.add_argument("--rmax", type=float, default=3)
    ap.add_argument("--nr", type=int, default=10)
    ap.add_argument("--tmin", type=float, default=1e2)
    ap.add_argument("--tmax", type=float, default=1e4)
    ap.add_argument("--nt", type=int, default=10)
    ap.add_argument("--tlog", action="store_true", help="Use log-spaced t grid (default).")
    ap.add_argument("--tlinear", action="store_true", help="Use linear-spaced t grid.")
    ap.add_argument("--seed", type=int, default=0, help="Reserved for future use; keeps CLI stable.")
    args = ap.parse_args()

    # Default: log-spaced t unless user explicitly asks linear.
    if args.tlinear:
        t_values = np.linspace(args.tmin, args.tmax, args.nt)
        t_grid_kind = "linear"
    else:
        t_values = np.logspace(np.log10(args.tmin), np.log10(args.tmax), args.nt)
        t_grid_kind = "log"

    r_values = np.linspace(args.rmin, args.rmax, args.nr)

    truth = np.full((t_values.size, r_values.size), np.nan, dtype=float)
    failed = np.zeros_like(truth, dtype=bool)

    t0 = time.time()
    total = t_values.size * r_values.size
    done = 0

    for i, t in enumerate(t_values):
        for j, r in enumerate(r_values):
            try:
                H = solve_qsip_adaptive_scipy(float(t), float(r))
                truth[i, j] = 1.0 if check_ineq(H, float(r)) else 0.0
                if truth[i,j] == 0:
                    print(H[1],H[2])
                    print(r)
            except Exception:
                failed[i, j] = True
                truth[i, j] = np.nan
            done += 1

        # light progress print (per t-row) so Spyder users see movement
        row_fail = int(failed[i, :].sum())
        print(f"[{i+1:>2}/{t_values.size}] t={t:,.6g}  row failures={row_fail}  done={done}/{total}")

    meta = {
        "ineq": "H[2]^2 - 2 H[1]^2 > r^2 + 2r",
        "solver": "solve_qsip_adaptive_scipy",
        "t_grid": t_grid_kind,
        "created_unix": str(time.time()),
        "elapsed_sec": str(time.time() - t0),
    }

    np.savez_compressed(
        args.out,
        r_values=r_values,
        t_values=t_values,
        truth=truth,
        failed=failed,
        meta=meta,
    )

    n_fail = int(np.sum(failed))
    print(f"Saved: {args.out}")
    print(f"Failures: {n_fail}/{total}")


if __name__ == "__main__":
    main()
