#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimators_neo.py — Increment (soon-to-be-discovered) estimators only.

Exports:
  - predict_new_SGT(phi, t, r)
  - predict_new_trivial(phi)
  - predict_new_ratio_multiplicative(phi, r)
  - predict_new_favaro(phi, r)

Notes:
  * These functions return the predicted INCREMENT beyond current sample size n.
  * They take phi (frequency-of-frequency vector) as input.
  * Internal helpers (binomial tails, root finding, Favaro's a) are kept private.
"""

from __future__ import annotations

import math
import sys
import numpy as np
from scipy.linalg import solve

__all__ = [
    "predict_new_SGT",
    "predict_new_trivial",
    "predict_new_ratio_multiplicative",
    "predict_new_favaro",
]

# --------------------------- SGT (binomial smoothing) --------------------------- #

def _binom_tail_probs(k: int, q: float):
    """
    Returns tail[l] = sum_{m=l}^k Binom(k, m) q^m (1-q)^{k-m} for l=0..k.
    Built via stable PMF recursion.
    """
    if k <= 0:
        return [1.0]
    pmf = [0.0] * (k + 1)
    pmf[0] = (1.0 - q) ** k
    for l in range(0, k):
        pmf[l + 1] = pmf[l] * (k - l) / (l + 1) * (q / (1.0 - q))
    tail = [0.0] * (k + 1)
    s = 0.0
    for l in range(k, -1, -1):
        s += pmf[l]
        tail[l] = s
    return tail


def predict_new_SGT(phi: np.ndarray, t: float, r: float) -> float:
    """
    SGT-style 'new species' term using binomial smoothing.
    """
    if r <= 1:
        return 0.0
    arg = t * (r * r) / (r - 1.0)
    if not math.isfinite(arg) or arg <= 1.0:
        k = 1
    else:
        k = int(math.floor(0.5 * (math.log(arg, 3.0))))
        if k < 1:
            k = 1
    q = 2.0 / (r + 2.0)
    tail = _binom_tail_probs(k, q)
    i_max = min(k, len(phi) - 1)
    total = 0.0
    rpow = r
    sign = 1.0  # (-1)^{i+1} r^i
    for i in range(1, i_max + 1):
        fi = phi[i]
        if fi:
            total += sign * rpow * fi * tail[i]
        sign *= -1.0
        rpow *= r
    return total


# ----------------------------- Trivial baseline ----------------------------- #

def predict_new_trivial(phi: np.ndarray) -> float:
    """
    Trivial increment baseline = 0.
    """
    return 0.0


# ------------------- Ratio multiplicative (increment form) ------------------- #

def predict_new_ratio_multiplicative(phi: np.ndarray, r: float) -> float:
    """
    Increment = total_ratio - S_n, where
      total_ratio = S_n * (1 + r)^{phi_1 / S_n}.
    Implemented without needing S_n from the caller.
    """
    Sn = float(sum(phi[1:]))
    if Sn <= 0:
        return float("nan")
    phi1 = float(phi[1]) if len(phi) > 1 else 0.0
    total = Sn * ((1.0 + r) ** (phi1 / Sn))
    return total - Sn


# ----------------------------- Favaro (increment) ----------------------------- #

def _brentq(f, a, b, tol=1e-10, maxiter=100):
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if not (fa * fb < 0):
        return float("nan")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb == 0.0:
            return b
        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a
        if abs(fa) < abs(fb):
            c, fc = b, fb
            b, fb = a, fa
            a, fa = c, fc
        m = 0.5 * (a - b)
        tol1 = 2.0 * sys.float_info.epsilon * abs(b) + 0.5 * tol
        if abs(m) <= tol1:
            return b
        if abs(e) >= tol1 and abs(fc) > abs(fb):
            s = fb / fc
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fc / fa
                r = fb / fa
                p = s * (2.0 * m * q * (q - r) - (b - c) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if (2.0 * p) < min(3.0 * m * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = e = m
        else:
            d = e = m
        c, fc = b, fb
        b += d if abs(d) > tol1 else (tol1 if m > 0 else -tol1)
        fb = f(b)
    return b


def _brentq_scan(f, a_lo, a_hi, steps=400, tol=1e-10, maxiter=100):
    f_lo, f_hi = f(a_lo), f(a_hi)
    if math.isfinite(f_lo) and math.isfinite(f_hi) and f_lo * f_hi < 0:
        return _brentq(f, a_lo, a_hi, tol=tol, maxiter=maxiter)
    prev_a, prev_f = a_lo, f_lo
    for i in range(1, steps + 1):
        a = a_lo + (a_hi - a_lo) * i / steps
        f_a = f(a)
        if math.isfinite(prev_f) and math.isfinite(f_a) and prev_f * f_a < 0:
            return _brentq(f, prev_a, a, tol=tol, maxiter=maxiter)
        prev_a, prev_f = a, f_a
    return float("nan")


def _a_favaro(phi: np.ndarray) -> float:
    """
    Solve Favaro's fixed-point for 'a' using φ and cumulative counts.
    Returns a in (0,1), or 0.0 if degenerate, or NaN if not solvable.
    """
    max_seen = len(phi) - 1
    if max_seen < 1:
        return float("nan")
    n = sum(j * phi[j] for j in range(1, len(phi)))
    if n <= 1:
        return float("nan")

    C = [0.0] * (max_seen + 1)
    tail = 0.0
    for ell in range(max_seen, 0, -1):
        tail += phi[ell]
        C[ell] = tail

    C1 = C[1]
    if C1 <= 1 or C1 >= n:
        return 0.0

    def varphi(a):
        s = 0.0
        kmax = min(n - 1, max_seen - 1)
        for k in range(1, kmax + 1):
            ck1 = C[k + 1]
            if ck1 > 0 and abs(k - a) > 1e-14:
                s += a / (k - a) * ck1
        return s - (C1 - 1)

    return _brentq_scan(varphi, 1e-9, 1 - 1e-9, 400)


def predict_new_favaro(phi: np.ndarray, r: float) -> float:
    """
    Increment = total_favaro - S_n, where
      total_favaro = S_n * (1 + r)^a, and a solves Favaro's equation.
    Implemented without needing S_n from the caller.
    """
    Sn = float(sum(phi[1:]))
    a = _a_favaro(phi)
    if not math.isfinite(a):
        return float("nan")
    total = Sn * ((1 + r) ** a)
    return total - Sn


def _padegt_core(phi, r, P=3, Q=4):
    m = min(len(phi) - 1, P + Q)
    if m <= 0:
        return 0.0

    # c_1,...,c_{P+Q}
    c = np.array([(-1)**(i+1) * phi[i] for i in range(1, m + 1)], float)
    if len(c) < P + Q:
        c = np.pad(c, (0, P + Q - len(c)))

    A = np.zeros((Q, Q))
    rhs = np.zeros(Q)
    for row in range(Q):
        k = P + 1 + row
        for col in range(Q):
            j = col + 1
            idx = k - j - 1
            if 0 <= idx < len(c):
                A[row, col] = c[idx]
        rhs[row] = -c[k - 1]

    if np.linalg.cond(A) > 1e12:
        return float("nan")
    b = solve(A, rhs)

    a = np.zeros(P + 1)
    for k in range(P + 1):
        s = 0.0
        for j in range(1, min(k, Q) + 1):
            idx = k - j - 1
            if idx >= 0:
                s += b[j - 1] * c[idx]
        if k == 0:
            a[k] = 0.0
        else:
            a[k] = c[k - 1] + s

    num = sum(a[i] * (r ** i) for i in range(P + 1))
    den = 1.0 + sum(b[j] * (r ** (j + 1)) for j in range(Q))
    return float(num / den)


def predict_new_padegt_plain(phi, r, P=3, Q=4):
    """Deterministic Padé–GT."""
    return _padegt_core(phi, r, P=P, Q=Q)


def predict_new_padegt_boot(phi, r, P=3, Q=4, B=200, rng=None):
    """
    Bootstrap-median Padé–GT:
    bootstrap species from φ, rebuild φ, run Padé–GT, take median.
    """
    if rng is None:
        rng = np.random.default_rng()

    # build species-size multiset from phi
    sizes = []
    for i in range(1, len(phi)):
        if phi[i] > 0:
            sizes.extend([i] * int(phi[i]))
    S_n = len(sizes)
    if S_n == 0:
        return 0.0

    ests = []
    for _ in range(B):
        samp = rng.choice(sizes, size=S_n, replace=True)
        max_s = int(np.max(samp))
        phi_b = np.zeros(max_s + 1, dtype=int)
        for s_ in samp:
            phi_b[s_] += 1
        est_b = _padegt_core(phi_b, r, P=P, Q=Q)
        if np.isfinite(est_b):
            ests.append(est_b)

    if not ests:
        return float("nan")
    return float(np.median(ests))



def predict_new_optimistic(phi, r):
    # predict "all singletons will show up r more times"
    if len(phi) < 2:
        return 0.0
    return float(phi[1] * r)

from numpy.polynomial import Chebyshev, Polynomial
from math import comb

def _chebyshev_interval_coeffs_increment(L, ell, r_hi):
    """
    Return a_0,...,a_L for
        P_L(x) = - T_L( (2x - r_hi - ell)/(r_hi - ell) )
                  / T_L( ( -r_hi - ell)/(r_hi - ell) )
    which is eq. (19) in Wu–Yang (2016).

    This guarantees P_L(0) = -1, so a_0 = -1.
    """
    if r_hi <= ell:
        raise ValueError("Need r_hi > ell for the Chebyshev scaling")

    T = Chebyshev.basis(L)

    # affine map
    alpha = 2.0 / (r_hi - ell)
    beta = -(r_hi + ell) / (r_hi - ell)

    # convert to ordinary polynomial
    T_poly = T.convert(kind=Polynomial)

    coeffs = np.zeros(L + 1, dtype=float)
    for k, c_k in enumerate(T_poly.coef):
        for j in range(k + 1):
            coeffs[j] += c_k * comb(k, j) * (alpha ** j) * (beta ** (k - j))

    denom = T(beta)
    coeffs = -coeffs / denom
    return coeffs


def predict_new_chebyshev_increment(phi, n, k_external, c0=0.45, c1=0.5):
    """
    Chebyshev estimator from Wu–Yang, returned as an *increment*:
        increment = sum_j g_L(j) * phi[j] - S_n

    Parameters
    ----------
    phi : array-like
        frequency-of-frequencies, phi[j] = #species seen exactly j times
    n : int or float
        current sample size
    k_external : int or float
        the 'k' that the paper takes as an input (NOT S_n)
    c0, c1 : floats
        constants from eq. (24). For the Hamlet experiment they used c0=0.45, c1=0.5.

    Returns
    -------
    float
        increment to add to S_n
    """
    k_external = float(k_external)
    if k_external <= 1.0:
        return 0.0

    # (24): L, ell, r
    L = int(math.floor(c0 * math.log(k_external)))
    L = max(L, 1)
    ell = 1.0 / k_external
    r_hi = (c1 * math.log(k_external)) / float(n)

    a = _chebyshev_interval_coeffs_increment(L, ell, r_hi)

    # S_n = #observed species
    Sn = float(sum(phi[1:]))

    total = 0.0
    max_j = len(phi) - 1
    for j in range(max_j + 1):
        if j == 0:
            g = 0.0
        elif j <= L:
            g = a[j] * math.factorial(j) / (float(n) ** j) + 1.0
        else:
            g = 1.0
        total += g * phi[j]

    # return increment
    return float(total - Sn)

