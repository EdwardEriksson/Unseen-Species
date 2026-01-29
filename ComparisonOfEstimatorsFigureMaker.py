#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a custom 4×3 grid figure for unseen-species estimator comparison.
Each dataset is placed in a specific subplot slot.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import re


# ============================================================
# 1. FILES — must appear in this exact order
# ============================================================

FILES = [
    # Row 1
    "./comparison_results/hamlet_TO1_K1_S1_run3.json",          # 0 hamlet true
    "./comparison_results/butterflies_TO1_K1_S1_run.json",      # 1 butterflies true
    "./comparison_results/college_msg_TO1_K100_S1_run.json",    # 2 messages true

    # Row 2
    "./comparison_results/hamlet_TO0_K1_S100_run3.json",        # 3 hamlet perm
    "./comparison_results/butterflies_TO0_K1_S100_run.json",    # 4 butterflies perm
    "./comparison_results/college_msg_TO0_K100_S100_run.json",  # 5 messages perm

    # Row 3
    "./comparison_results/mtg_N14_TO0_K1_S10_run.json",         # 6 mtg N14
    "./comparison_results/mtg_N28_TO0_K1_S10_run.json",         # 7 mtg N28
    "./comparison_results/genes_TO0_K100_S100_run.json",        # 8 genes

    # Row 4
    "./comparison_results/synthetic_uniform_M1000_TO0_K1_S100_run.json",   # 9 synthetic small
    "./comparison_results/synthetic_uniform_M10000_TO0_K1_S100_run.json",  # 10 synthetic large

    # Slot 11 unused
]


# ============================================================
# 2. Load data
# ============================================================

def load_result(path):
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

def dataset_title(label, path):
    t = (label + " " + path).lower()

    if "hamlet" in t:
        return "Hamlet (True Order)" if "to1" in t else "Hamlet (Permutation Average)"
    if "butter" in t:
        return "Butterflies (True Order)" if "to1" in t else "Butterflies (Permutation Average)"
    if "college" in t or "msg" in t:
        return "Messages (True Order)" if "to1" in t else "Messages (Permutation Average)"
    if "mtg" in t:
        if "n14" in t: return "MTG (N=14)"
        if "n28" in t: return "MTG (N=28)"
        return "MTG"
    if "gene" in t:
        return "Genome Coverage"
    if "synthetic" in t:
        # Extract M from filename: look for "_M####_"
        m_match = re.search(r"_M(\d+)_", path)
        m_val = int(m_match.group(1)) if m_match else None
    
        # N = 2000 (your synthetic generator)
        N = 2000
    
        if m_val is not None:
            return f"Synthetic Uniform (N={N}, M={m_val})"
        else:
            return f"Synthetic Uniform (N={N}, M=?)"



    return label


def dataset_x_label(label, path):
    t = (label + " " + path).lower()
    if "hamlet" in t: return "Fraction of words read (%)"
    if "butter" in t: return "Fraction of collecting days (%)"
    if "college" in t or "msg" in t: return "Fraction of messages seen (%)"
    if "gene" in t: return "Fraction of fragments sequenced (%)"
    if "synthetic" in t: return "Fraction of samples seen (%)"
    if "mtg" in t: return "Fraction of boosters (%)"
    return "Fraction processed (%)"


# ============================================================
# 4. Plotting helper
# ============================================================

def plot_one(ax, data):
    include = data["include"]
    mean = data["mean_abs_pct_err"]
    perc = np.array(data["perc_vals"], float)

    ms = 2
    lw = 1.0

    if include["QSIP"]:
        ax.plot(perc, mean["qsip"], "o-", ms=ms, lw=lw, label=r"$H^{*}$")

    if include["SGT"]:
        ax.plot(perc, mean["SGT"], "s-", ms=ms, lw=lw, label="SGT")

    if include["TRIV"]:
        ax.plot(perc, mean["triv"], "^--", ms=ms, lw=lw, alpha=0.8, label="Trivial")

    if include["RATIO"]:
        ax.plot(perc, mean["ratio"], "x-", ms=ms, lw=lw, label=r"Ratio-$\alpha$")

    if include["FAVARO"]:
        ax.plot(perc, mean["fav"], "d-", ms=ms, lw=lw, label="MLE-$\\alpha$")

    if include["PADEGT_PLAIN"]:
        P = data["pade_params"].get("P")
        Q = data["pade_params"].get("Q")
        lbl = f"Padé–GT [{P},{Q}]" if P is not None and Q is not None else "Padé–GT"
        ax.plot(perc, mean["padegt_plain"], "*-", ms=ms, lw=lw, label=lbl)

    if include["PADEGT_BOOT"]:
        P = data["pade_params"].get("P")
        Q = data["pade_params"].get("Q")
        lbl = f"Padé–GT boot [{P},{Q}]" if P is not None and Q is not None else "Padé–GT boot"
        ax.plot(perc, mean["padegt_boot"], "o--", ms=ms, lw=lw, label=lbl)

    if include["CHEBYSHEV"]:
        ax.plot(perc, mean["cheby"], "o-", ms=ms, lw=lw, label="Chebyshev")

    ax.set_facecolor("#fafafa")
    for i, ax in enumerate(axes):
        if i in [0, 1, 2]:  # top row
            ax.set_ylabel("Absolute percentage error (%)")
        elif i != 11:      # skip unused axis
            ax.set_ylabel("Mean absolute percentage error (%)")



# ============================================================
# 5. Build fixed 4×3 grid and place plots
# ============================================================

# 12 axes arranged row-major:
# 0  1  2
# 3  4  5
# 6  7  8
# 9 10 11
rows, cols = 4, 3
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
axes = np.array(axes).reshape(-1)

# dataset index → target axis index
placement = {
    0: 0,   # hamlet true
    1: 1,   # butterflies true
    2: 2,   # messages true
    3: 3,   # hamlet perm
    4: 4,   # butterflies perm
    5: 5,   # messages perm
    6: 6,   # mtg N14
    7: 7,   # mtg N28
    8: 8,   # genes
    9: 9,   # synthetic small
    10: 10, # synthetic large
    # 11 unused
}

# main plotting loop
for data_idx, (d, path) in enumerate(zip(DATA, FILES)):
    ax = axes[ placement[data_idx] ]

    plot_one(ax, d)

    ax.set_xlabel(dataset_x_label(d.get("dataset_label", ""), path))
    ax.set_title(dataset_title(d.get("dataset_label", ""), path), pad=8)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 100)

# remove bottom-right unused axis
fig.delaxes(axes[11])


# ============================================================
# 6. Global legend
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10)

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()
