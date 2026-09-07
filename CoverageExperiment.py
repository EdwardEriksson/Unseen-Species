#!/usr/bin/env python3
"""
coverage_study.py
=================
Coverage of the *naive* prediction interval for the linear estimator H*,

    Uhat  +-  z * sqrt(Vhat),     Vhat = Uhat + sum_s H(N_s)^2,

i.e. Theorem 26 with the bias term dropped.  Everything used to build the
interval is observable.  Reports, for a grid of (distribution, r, t):

    delta     = sum_s b_s / sqrt(V)          exact studentized bias
    cov_pred  = Phi(z - delta) - Phi(-z - delta)
    cov_mc    = Monte-Carlo coverage using the random Vhat
    relRMSE   = sqrt(MSE) / E[U]             is the problem even solvable?

Usage
-----
    python coverage_study.py fast      # seconds
    python coverage_study.py slow      # a few minutes
    python coverage_study.py fast --nsim 4000 --out mytable.csv

Requires cutting_plane_qsip.py on the path.  Solved H* vectors are cached
in ./Hcache/ so re-runs are cheap.

MODEL (Poissonised).  Species s has probability p_s; N_s ~ Poi(p_s t) in
the observed window, M_s ~ Poi(r p_s t) in the future window, independent.
U = #{s : N_s = 0, M_s >= 1}.  Uhat = sum_k H_k phi_k.  With y = p t and
g_H(y) = sum_k H_k y^k / k!,

    b(y) = e^{-y} ( g_H(y) - (1 - e^{-ry}) )        per-species bias
    m(y) = e^{-y} ( g_{H^2}(y) + 1 - e^{-ry} )      per-species 2nd moment
    v(y) = m(y) - b(y)^2                            per-species variance

and Bias = sum_s b(p_s t), V = sum_s v(p_s t) exactly, by independence.

SPEED.  Species with equal p are exchangeable, so instead of drawing 2S
Poissons per replicate we bin p into NBIN log-spaced groups (preserving
total mass exactly) and draw one multinomial per group over the K+3
categories (N=0,M=0), (N=0,M>=1), N=1..K, N>K.  This is exact for the
binned population, and the reported delta is computed for that same
binned population, so prediction and simulation always refer to the same
distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

try:
    from cutting_plane_qsip import solve_qsip_cutting_plane, _find_sup_vec
except ImportError:
    sys.exit("Could not import cutting_plane_qsip.py -- put it on the path.")

# ===================================================================
#  >>>>>  TOGGLE HERE when running from an editor (no CLI args)  <<<<<
# ===================================================================
MODE = "slow"        # "fast"  ~5 s cold / 0.4 s warm,  10 cells
                     # "slow"  ~3 min cold / 2.5 min warm, 225 cells
NSIM_OVERRIDE = None  # e.g. 1000 to shorten "slow"; None = preset default
# ===================================================================

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hcache")
Z95 = 1.959963984540054


# ===================================================================
#  Solver (disk-cached)
# ===================================================================

def solve_H(t: float, r: float, K: int) -> np.ndarray:
    os.makedirs(CACHE, exist_ok=True)
    tag = hashlib.md5(f"{t:.10g}|{r:.10g}|{K}".encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"H_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)
    H = solve_qsip_cutting_plane(t=float(t), r=float(r), K=int(K))
    np.save(path, H)
    return H


def sup_terms(H, t, r, n_grid=3000):
    """(F1, F2) -- the two terms of the functional, for reference."""
    F1, _, _ = _find_sup_vec(H, t, r, 1e-10, "f1", n_grid=n_grid)
    F2, _, _ = _find_sup_vec(H, t, r, 1e-10, "f2", n_grid=n_grid)
    return float(F1), float(F2)


# ===================================================================
#  Exact per-species quantities
# ===================================================================

def pois_pmf(y: np.ndarray, K: int) -> np.ndarray:
    """P[i,k] = e^{-y_i} y_i^k / k!, k = 0..K.  Log-domain, no overflow."""
    y = np.atleast_1d(np.asarray(y, float))
    k = np.arange(K + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logy = np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), -np.inf)
        P = np.exp(-y[:, None] + k[None, :] * logy[:, None]
                   - gammaln(k + 1)[None, :])
    zero = (y == 0)
    if np.any(zero):
        P[zero, :] = 0.0
        P[zero, 0] = 1.0
    return P


def exact_moments(p: np.ndarray, H: np.ndarray, t: float, r: float,
                  mult: np.ndarray | None = None):
    """
    Exact (Bias, AbsBias, Var, EU).  `p` holds distinct probabilities and
    `mult` their multiplicities (default: all 1), so this costs O(#bins),
    not O(#species).
    """
    K = len(H) - 1
    y = np.asarray(p, float) * t
    w = np.ones_like(y) if mult is None else np.asarray(mult, float)
    P = pois_pmf(y, K)
    EZ = np.exp(-y) * (-np.expm1(-r * y))
    b = P @ H - EZ
    m = P @ (H * H) + EZ
    v = m - b * b
    return (float(w @ b), float(w @ np.abs(b)),
            float(w @ v), float(w @ EZ))


# ===================================================================
#  Binning + multinomial Monte Carlo
# ===================================================================

def bin_population(p: np.ndarray, nbin: int):
    """Group species into log-spaced bins.  Total mass preserved exactly."""
    p = np.sort(np.asarray(p, float))
    p = p[p > 0]
    lo, hi = np.log(p[0]), np.log(p[-1])
    if hi - lo < 1e-12:
        return np.array([len(p)]), np.array([p.mean()])
    idx = np.clip(((np.log(p) - lo) / (hi - lo) * nbin).astype(int), 0, nbin - 1)
    n = np.bincount(idx, minlength=nbin)
    s = np.bincount(idx, weights=p, minlength=nbin)
    keep = n > 0
    return n[keep], s[keep] / n[keep]


def category_probs(pb: np.ndarray, t: float, r: float, K: int) -> np.ndarray:
    """Shape (nbin, K+3): (N=0,M=0), (N=0,M>=1), N=1..K, N>K."""
    y = pb * t
    P = pois_pmf(y, K)
    e = np.exp(-y)
    q = np.empty((len(y), K + 3))
    q[:, 0] = e * np.exp(-r * y)
    q[:, 1] = e * (-np.expm1(-r * y))
    q[:, 2:K + 2] = P[:, 1:K + 1]
    q[:, K + 2] = np.maximum(1.0 - q[:, :K + 2].sum(1), 0.0)
    return q / q.sum(1, keepdims=True)


def mc_coverage(n_bin, p_bin, H, t, r, nsim, rng, z=Z95):
    K = len(H) - 1
    q = category_probs(p_bin, t, r, K)
    w1 = np.concatenate([[0.0, 0.0], H[1:K + 1], [0.0]])
    w2 = w1 ** 2
    Uh = np.zeros(nsim)
    Q = np.zeros(nsim)
    U = np.zeros(nsim)
    for g in range(len(n_bin)):
        if n_bin[g] == 0:
            continue
        c = rng.multinomial(n_bin[g], q[g], size=nsim)
        Uh += c @ w1
        Q += c @ w2
        U += c[:, 1]
    Vh = Uh + Q
    ok = np.abs(Uh - U) <= z * np.sqrt(np.maximum(Vh, 0.0))
    return float(ok.mean())


# ===================================================================
#  Distribution families
# ===================================================================

def d_uniform(S):
    S = int(S)
    return np.full(S, 1.0 / S)


def d_zipf(s, S):
    j = np.arange(1, int(S) + 1, dtype=float)
    w = j ** (-float(s))
    return w / w.sum()


def d_geometric(theta, S=None):
    if S is None:
        S = int(np.ceil(np.log(1e-16) / np.log1p(-theta))) + 1
    j = np.arange(int(S), dtype=float)
    w = (1 - theta) ** j
    return w / w.sum()


def d_loguniform(S, decades, seed):
    rng = np.random.default_rng(seed)
    w = 10.0 ** rng.uniform(-decades, 0.0, int(S))
    return w / w.sum()


def d_gem(theta, S, seed):
    rng = np.random.default_rng(seed)
    V = rng.beta(1.0, theta, int(S))
    rem = np.concatenate([[0.0], np.cumsum(np.log1p(-V[:-1]))])
    w = V * np.exp(rem)
    return w / w.sum()


def d_py(alpha, theta, S, seed):
    rng = np.random.default_rng(seed)
    k = np.arange(int(S))
    V = rng.beta(1.0 - alpha, theta + (k + 1) * alpha)
    rem = np.concatenate([[0.0], np.cumsum(np.log1p(-V[:-1]))])
    w = V * np.exp(rem)
    return w / w.sum()


def build_families(t, cfg):
    """Populations scale with t so difficulty is comparable across t."""
    out = []
    for mult in cfg["unif_mult"]:
        S = max(2, int(mult * t))
        out.append((f"uniform y={1.0/mult:g}", d_uniform(S)))
    for s in cfg["zipf_s"]:
        for S in cfg["zipf_S"]:
            out.append((f"Zipf s={s} S=1e{int(round(np.log10(S)))}", d_zipf(s, S)))
    for th in cfg["geom_theta"]:
        out.append((f"geometric th={th:g}", d_geometric(th)))
    for dec in cfg["logunif_dec"]:
        out.append((f"log-unif {dec}dec", d_loguniform(cfg["misc_S"], dec, cfg["seed"])))
    for th in cfg["gem_theta"]:
        out.append((f"GEM th={th:g}", d_gem(th, cfg["misc_S"], cfg["seed"])))
    for a in cfg["py_alpha"]:
        out.append((f"PY a={a}", d_py(a, 10.0, cfg["misc_S"], cfg["seed"])))
    return out


# ===================================================================
#  Presets
# ===================================================================

FAST = dict(
    K=10, nsim=1000, nbin=50, seed=0,
    ts=[1000.0], rs=[2.0, 4.0],
    unif_mult=[2, 20],
    zipf_s=[1.05, 1.5], zipf_S=[10**4],
    geom_theta=[], logunif_dec=[3.0], gem_theta=[], py_alpha=[],
    misc_S=20000,
)

SLOW = dict(
    K=10, nsim=4000, nbin=120, seed=0,
    ts=[100.0, 1000.0, 10000.0], rs=[1.5, 2.0, 3.0, 4.0, 6.0],
    unif_mult=[2, 20],
    zipf_s=[1.05, 1.2, 1.5, 2.0], zipf_S=[10**4, 10**6],
    geom_theta=[1e-4], logunif_dec=[3.0, 5.0], gem_theta=[1e3], py_alpha=[0.9],
    misc_S=200000,
)


# ===================================================================
#  Main
# ===================================================================

def main():
    # argparse still works from a terminal and overrides MODE above
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default=MODE, choices=["fast", "slow"])
    ap.add_argument("--nsim", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = dict(FAST if args.mode == "fast" else SLOW)
    if NSIM_OVERRIDE:
        cfg["nsim"] = NSIM_OVERRIDE
    if args.nsim:
        cfg["nsim"] = args.nsim
    if args.K:
        cfg["K"] = args.K
    out_csv = args.out or f"coverage_{args.mode}.csv"

    K, nsim, nbin = cfg["K"], cfg["nsim"], cfg["nbin"]
    rng = np.random.default_rng(cfg["seed"] + 12345)
    t0 = time.time()

    rows = []
    for t in cfg["ts"]:
        fams = build_families(t, cfg)
        binned = [(nm, *bin_population(p, nbin)) for nm, p in fams]
        for r in cfg["rs"]:
            H = solve_H(t, r, K)
            F1, F2 = sup_terms(H, t, r)
            for nm, n_bin, p_bin in binned:
                B, A, V, EU = exact_moments(p_bin, H, t, r, mult=n_bin)
                delta = B / np.sqrt(V) if V > 0 else np.nan
                cov_pred = norm.cdf(Z95 - delta) - norm.cdf(-Z95 - delta)
                cov_mc = mc_coverage(n_bin, p_bin, H, t, r, nsim, rng)
                mse = V + B * B
                rows.append(dict(
                    t=t, r=r, dist=nm, S=int(n_bin.sum()), EU=EU,
                    kappa=(abs(B) / A if A > 0 else np.nan),
                    delta=delta, cov_pred=cov_pred, cov_mc=cov_mc,
                    se=np.sqrt(cov_mc * (1 - cov_mc) / nsim),
                    relRMSE=(np.sqrt(mse) / EU if EU > 0 else np.nan),
                    F1=F1, F2=F2))
                print(f"  t={t:>7.0f} r={r:>4.1f} {nm:<24s} "
                      f"delta={delta:+8.4f}  pred={cov_pred:.4f}  "
                      f"MC={cov_mc:.4f}  relRMSE={rows[-1]['relRMSE']:.3f}")

    # ---- output ----
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(out_csv, index=False)
        pd.set_option("display.width", 200)
        for t in cfg["ts"]:
            print(f"\n===== MC coverage, nominal 0.95, t = {t:.0f} "
                  f"(nsim={nsim}, SE ~ {np.sqrt(.05*.95/nsim):.4f}) =====")
            piv = df[df.t == t].pivot(index="dist", columns="r", values="cov_mc")
            piv = piv.reindex([d for d in dict.fromkeys(df.dist) if d in piv.index])
            print(piv.to_string(float_format=lambda x: f"{x:7.4f}"))
        worst = df.loc[df.cov_mc.idxmin()]
        print(f"\nworst cell: {worst.dist} r={worst.r} t={worst.t:.0f} "
              f"-> {worst.cov_mc:.4f} (delta={worst.delta:+.3f})")
        bad = df[df.cov_mc < 0.93]
        print(f"cells below 0.93: {len(bad)} / {len(df)}")
    else:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\nwrote {out_csv}   [{time.time() - t0:.1f} s, {len(rows)} cells]")


if __name__ == "__main__":
    main()