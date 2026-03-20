#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a fixed 4×3 grid figure for unseen-species estimator comparison.
Each dataset is placed in a specific subplot slot (row-major).
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. FILES — must appear in this exact order (12 files = full grid)
# ============================================================

FILES = [
    # Row 1
    "./comparison_results\hamlet_TO1_K1_S1_run.json",          # 0 hamlet true
    "./comparison_results/butterflies_TO1_K1_S1_run.json",      # 1 butterflies true
    "./comparison_results\college_msg_TO1_K100_S1_run.json",    # 2 messages true

    # Row 2
    "./comparison_results\hamlet_TO0_K1_S100_run.json",        # 3 hamlet perm
    "./comparison_results/butterflies_TO0_K1_S100_run.json",    # 4 butterflies perm
    "./comparison_results\college_msg_TO0_K100_S100_run.json",  # 5 messages perm

    # Row 3
    "./comparison_results\mtg_N12_TO0_K1_S100_run.json",         # 6 mtg N=12
    "./comparison_results\mtg_N24_TO0_K1_S100_run.json",         # 7 mtg N=24
    "./comparison_results\mtg_N48_TO0_K1_S100_run.json",         # 8 mtg N=48

    # Row 4
    "./comparison_results/genes_TO0_K10_S100_run.json",        # 9 genes
    "./comparison_results/synthetic_uniform_M1000_TO0_K1_S100_run.json",   # 10 synthetic small
    "./comparison_results/synthetic_uniform_M10000_TO0_K1_S100_run.json",  # 11 synthetic large
]


# ============================================================
# 2. Load data
# ============================================================

def load_result(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

DATA = []
for path in FILES:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    DATA.append(load_result(path))


# ============================================================
# 3. Labels
# ============================================================

def dataset_title(label: str, path: str) -> str:
    t = (label + " " + path).lower()

    if "hamlet" in t:
        return "Hamlet (True Order)" if "to1" in t else "Hamlet (Permutation Average)"
    if "butter" in t:
        return "Butterflies (True Order)" if "to1" in t else "Butterflies (Permutation Average)"
    if "college" in t or "msg" in t:
        return "Messages (True Order)" if "to1" in t else "Messages (Permutation Average)"
    if "mtg" in t:
        n_match = re.search(r"_n(\d+)_", t)
        if n_match:
            return f"MTG (N={n_match.group(1)})"
        return "MTG"
    if "gene" in t:
        return "Genome Coverage"
    if "synthetic" in t:
        m_match = re.search(r"_M(\d+)_", path)
        m_val = int(m_match.group(1)) if m_match else None
        N = 2000  # your synthetic generator
        return f"Synthetic Uniform (N={N}, M={m_val})" if m_val is not None else f"Synthetic Uniform (N={N}, M=?)"

    return label


def dataset_x_label(label: str, path: str) -> str:
    t = (label + " " + path).lower()
    if "hamlet" in t: return "Fraction of words read (%)"
    if "butter" in t: return "Fraction of collecting days (%)"
    if "college" in t or "msg" in t: return "Fraction of messages seen (%)"
    if "gene" in t: return "Fraction of fragments sequenced (%)"
    if "synthetic" in t: return "Fraction of samples seen (%)"
    if "mtg" in t: return "Fraction of packs opened (%)"
    return "Fraction processed (%)"


# ============================================================
# 4. Plotting helper
# ============================================================

def plot_one(ax, data: dict) -> None:
    include = data["include"]
    mean = data["mean_abs_pct_err"]
    perc = np.array(data["perc_vals"], float)

    ms = 2
    lw = 1.0

    if include.get("QSIP", False):
        ax.plot(perc, mean["qsip"], "o-", ms=ms, lw=lw, label=r"$H^{*}$")

    if include.get("SGT", False):
        ax.plot(perc, mean["SGT"], "s-", ms=ms, lw=lw, label="SGT")

    if include.get("TRIV", False):
        ax.plot(perc, mean["triv"], "^--", ms=ms, lw=lw, alpha=0.8, label="Trivial", color="C3")

    if include.get("RATIO", False):
        ax.plot(perc, mean["ratio"], "x-", ms=ms, lw=lw, label=r"Ratio-$\alpha$")

    if include.get("FAVARO", False):
        ax.plot(perc, mean["fav"], "d-", ms=ms, lw=lw, label="MLE-$\\alpha$")

    if include.get("PADEGT_PLAIN", False):
        P = data.get("pade_params", {}).get("P")
        Q = data.get("pade_params", {}).get("Q")
        lbl = f"Padé–GT [{P},{Q}]" if P is not None and Q is not None else "Padé–GT"
        ax.plot(perc, mean["padegt_plain"], "*-", ms=ms, lw=lw, label=lbl)

    if include.get("PADEGT_BOOT", False):
        P = data.get("pade_params", {}).get("P")
        Q = data.get("pade_params", {}).get("Q")
        lbl = f"Padé–GT boot [{P},{Q}]" if P is not None and Q is not None else "Padé–GT boot"
        ax.plot(perc, mean["padegt_boot"], "o--", ms=ms, lw=lw, label=lbl)

    if include.get("CHEBYSHEV", False):
        ax.plot(perc, mean["cheby"], "o-", ms=ms, lw=lw, label="Chebyshev")

    ax.set_facecolor("#fafafa")


# ============================================================
# 5. Build fixed 4×3 grid and place plots
# ============================================================

# Axes are row-major indices:
# 0  1  2
# 3  4  5
# 6  7  8
# 9 10 11
rows, cols = 4, 3
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
axes = np.array(axes).reshape(-1)

# dataset index → target axis index
# (here it is identity, but kept explicit to avoid confusion later)
placement = {i: i for i in range(len(FILES))}

for data_idx, (d, path) in enumerate(zip(DATA, FILES)):
    ax_i = placement[data_idx]
    ax = axes[ax_i]

    plot_one(ax, d)

    ax.set_xlabel(dataset_x_label(d.get("dataset_label", ""), path), fontsize=16)
    ax.set_title(dataset_title(d.get("dataset_label", ""), path), pad=8, fontsize=16)

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)

    # y-labels: only first column to reduce clutter
    if ax_i in [0, 3, 6, 9]:
        # preserve your earlier convention: top row says "Absolute ...", others say "Mean absolute ..."
        if ax_i in [0]:
            ax.set_ylabel("Absolute percentage error (%)", fontsize=16)
        else:
            ax.set_ylabel("Mean absolute percentage error (%)", fontsize=14)


# ============================================================
# 6. Global legend
# ============================================================

# Pick legend entries from the first axis that has any labels
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    if len(h) > 0:
        handles, labels = h, l
        break

if handles:
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=16)

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.suptitle("Experimental Comparison of Estimators", fontsize=22, y=1.02)
plt.show()
