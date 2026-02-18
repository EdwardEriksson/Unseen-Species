#!/usr/bin/env python3
"""
plot_qsip_ineq_sweep.py
=======================
Plot a saved sweep (created by qsip_ineq_sweep.py).

Example:
    python plot_qsip_ineq_sweep.py --inp qsip_ineq_sweep.npz
    python plot_qsip_ineq_sweep.py --inp sweep.npz --out ineq_map.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def make_edges(vals, *, log_safe=False):
    """
    Bin edges from bin centers.

    If log_safe=True (recommended for log-scaled positive axes),
    uses geometric midpoints and multiplicative extrapolation so edges stay > 0.
    """
    vals = np.asarray(vals, dtype=float)

    if vals.size == 1:
        v = vals[0]
        return np.array([v * 0.9, v * 1.1]) if (log_safe and v > 0) else np.array([v - 0.5, v + 0.5])

    if log_safe:
        if np.any(vals <= 0):
            raise ValueError("log_safe=True requires all vals > 0.")

        mids = np.sqrt(vals[:-1] * vals[1:])  # geometric means
        edges = np.empty(vals.size + 1, dtype=float)
        edges[1:-1] = mids

        # multiplicative extrapolation
        edges[0] = vals[0]**2 / mids[0]
        edges[-1] = vals[-1]**2 / mids[-1]
        return edges

    # linear-safe (arithmetic) edges
    mids = (vals[:-1] + vals[1:]) / 2.0
    edges = np.empty(vals.size + 1, dtype=float)
    edges[1:-1] = mids
    edges[0] = vals[0] - (mids[0] - vals[0])
    edges[-1] = vals[-1] + (vals[-1] - mids[-1])
    return edges



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=str, default="qsip_ineq_sweep.npz")
    ap.add_argument("--out", type=str, default=None, help="If set, save figure to this path.")
    ap.add_argument("--fail_color", type=str, default="gray")
    ap.add_argument("--yscale", choices=["linear", "log", "auto"], default="auto")
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    data = np.load(args.inp, allow_pickle=True)
    r_values = data["r_values"].astype(float)
    t_values = data["t_values"].astype(float)
    truth = data["truth"].astype(float)
    failed = data["failed"].astype(bool)
    meta = data["meta"].item() if "meta" in data.files else {}

    r_edges = make_edges(r_values)
    t_edges = make_edges(t_values, log_safe=True)

    cmap = ListedColormap(["red", "green"])
    cmap.set_bad(color=args.fail_color)  # NaN = failure tiles
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=cmap.N)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=140)

    mesh = ax.pcolormesh(
        r_edges, t_edges, truth,
        cmap=cmap, norm=norm,
        shading="flat",
        linewidth=0.8, edgecolors=(1, 1, 1, 0.55)  # crisp tiles on coarse grids
    )

    ax.set_xlabel("r")
    ax.set_ylabel("t")

    if args.title is not None:
        ax.set_title(args.title)
    else:
        ax.set_title(meta.get("ineq", "Inequality truth map"))

    # y-scale handling
    if args.yscale == "log":
        ax.set_yscale("log")
    elif args.yscale == "linear":
        ax.set_yscale("linear")
    else:
        # auto: if t looks log-spaced-ish and positive, use log
        if np.all(t_values > 0) and t_values.size >= 3:
            ratios = t_values[1:] / t_values[:-1]
            if np.nanstd(ratios) < 1e-3 and np.nanmean(ratios) > 1.01:
                ax.set_yscale("log")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["False", "True"])

    n_fail = int(np.sum(failed))
    total = int(failed.size)
    if n_fail > 0:
        ax.text(
            0.99, 0.01, f"solver failures: {n_fail}/{total}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
        )

    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, bbox_inches="tight")
        print(f"Saved figure: {args.out}")

    plt.show()


if __name__ == "__main__":
    main()
