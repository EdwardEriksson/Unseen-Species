#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.ticker import LogFormatterMathtext

LABELS = [
    r"Our linear estimator, $H^{*}$",
    "SGT, our analysis",
    "SGT, their analysis",
    "Null-estimator",
]

def _scatter_masked(ax, x, y, *args, **kwargs):
    """Scatter while masking non-finite or non-positive y values (for log plots)."""
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    return ax.scatter(x[mask], y[mask], *args, **kwargs)

def read_vs_t(csv_path):
    by_r = defaultdict(lambda: {"t": [], "sqrt_star": [], "sqrt_HO": [], "sqrt_f": [], "sqrt_g": []})
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode") != "vs_t":
                continue
            r = float(row["r"])
            t = float(row["t"])
            by_r[r]["t"].append(t)
            by_r[r]["sqrt_star"].append(float(row["sqrt_total_star"]))
            by_r[r]["sqrt_HO"].append(float(row["sqrt_total_HO"]))
            by_r[r]["sqrt_f"].append(float(row["sqrt_f"]))
            by_r[r]["sqrt_g"].append(float(row["sqrt_g"]))
    # Sort by t
    for r, d in by_r.items():
        order = np.argsort(np.array(d["t"]))
        for k in d:
            d[k] = np.array(d[k])[order]
    return by_r

def read_vs_r(csv_path):
    by_t = defaultdict(lambda: {"r": [], "sqrt_star": [], "sqrt_HO": [], "sqrt_f": [], "sqrt_g": []})
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("mode") != "vs_r":
                continue
            t = float(row["t"])
            r = float(row["r"])
            by_t[t]["r"].append(r)
            by_t[t]["sqrt_star"].append(float(row["sqrt_total_star"]))
            by_t[t]["sqrt_HO"].append(float(row["sqrt_total_HO"]))
            by_t[t]["sqrt_f"].append(float(row["sqrt_f"]))
            by_t[t]["sqrt_g"].append(float(row["sqrt_g"]))
    # Sort by r
    for t, d in by_t.items():
        order = np.argsort(np.array(d["r"]))
        for k in d:
            d[k] = np.array(d[k])[order]
    return by_t

def plot_comparison_grid(vs_t_csv="rmse_vs_t.csv",
                         vs_r_csv="rmse_vs_r.csv",
                         rs=(1.5, 3.0, 6.0),
                         ts=(100.0, 1e3, 1e4)):
    by_r = read_vs_t(vs_t_csv)
    by_t = read_vs_r(vs_r_csv)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)

    # ---------------- Top row: fixed r, vary t (LOG–LOG SCALE) ----------------
    handles = None
    for j, r in enumerate(rs):
        ax = axes[0, j]
        d = by_r.get(r, None)
        if d is None:
            ax.set_visible(False)
            continue
        t = d["t"]

        h1 = _scatter_masked(ax, t, d["sqrt_star"], label=LABELS[0], marker="o",s=70, color="C0")
        h2 = _scatter_masked(ax, t, d["sqrt_HO"],   label=LABELS[1], marker="s",s=70, color="C1")
        h3 = _scatter_masked(ax, t, d["sqrt_f"],    label=LABELS[2], marker="^",s=70, color="C2")
        h4 = _scatter_masked(ax, t, d["sqrt_g"],    label=LABELS[3], marker="x",s=70, color="C3")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t", fontsize=20)
        ax.set_title(f"r = {r}",fontsize=20)

        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.tick_params(axis="y", which="both", labelleft=True,labelsize=14)
        ax.tick_params(axis="x", which="both",labelsize=15)

        if handles is None:
            handles = [h1, h2, h3, h4]

    axes[0, 0].set_ylabel("Worst case RMSE",fontsize=20)

    # ---------------- Bottom row: fixed t, vary r (LINEAR SCALES) ----------------
    for j, tval in enumerate(ts):
        ax = axes[1, j]
        d = by_t.get(tval, None)
        if d is None:
            ax.set_visible(False)
            continue
        r = d["r"]

        ax.scatter(r, d["sqrt_star"], label=LABELS[0], marker="o",s=70, color="C0")
        ax.scatter(r, d["sqrt_HO"],   label=LABELS[1], marker="s",s=70, color="C1")
        ax.scatter(r, d["sqrt_f"],    label=LABELS[2], marker="^",s=70, color="C2")
        ax.scatter(r, d["sqrt_g"],    label=LABELS[3], marker="x",s=70, color="C3")

        ax.set_xlabel("r",fontsize=20)
        t_label = int(tval) if float(tval).is_integer() else tval
        ax.set_title(f"t = {t_label}",fontsize=20)

        ax.tick_params(axis="y", which="both", labelleft=True,labelsize=13)
        ax.tick_params(axis="x", which="both",labelsize=13)

    axes[1, 0].set_ylabel("Worst case RMSE",fontsize=20)

    # ---------------- Title & legend ----------------
    fig.suptitle("Worst Case RMSE Comparison", y=0.94,fontsize=20)

    if handles:
        labels = [h.get_label() for h in handles]
        fig.legend(handles, labels, loc="lower center", ncol=2,fontsize=20,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.show()

if __name__ == "__main__":
    plot_comparison_grid(
        vs_t_csv="rmse_vs_t.csv",
        vs_r_csv="rmse_vs_r.csv",
        rs=(1.5, 3.0, 6.0),
        ts=(100.0, 1e3, 1e4)
    )
