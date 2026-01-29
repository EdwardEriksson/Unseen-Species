#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qsip_solver.py — SciPy L-BFGS-B QSIP solver (optimized-H) with softmax-sup smoothing.

This module extracts only the QSIP solver and its minimal dependencies so it can
be imported and reused by other scripts.

Core API
--------
- solve_qsip_adaptive_scipy(t, r, K=10, ...)
    Returns the optimized H vector of length K+1 with H[0] = 0 and H[1:K] free.

Optional helpers
----------------
- score_objective(H, t, r, ...)
    Evaluate the smoothed objective at a given H (useful for diagnostics).
- make_log_grids(eps, p_init, q_init)
    Build initial log-spaced p- and q-grids in (eps, 1].

Notes
-----
We parameterize the optimization in terms of h_tail = H[1:], keeping H[0] fixed
to 0 and *not* part of the optimization variables (as discussed earlier).
"""

from __future__ import annotations

import math
from math import factorial
from typing import Tuple, Optional, Dict

import numpy as np
from scipy.optimize import minimize, minimize_scalar

__all__ = [
    "solve_qsip_adaptive_scipy",
    "score_objective",
    "make_log_grids",
    "logspace01",
    "precompute_factorials",
]

# ------------------------------ Grids & Facts ------------------------------ #

def logspace01(eps: float = 1e-12, n: int = 3001) -> np.ndarray:
    """
    Log-spaced grid on (eps, 1], dense near 0 (to resolve suprema there).
    """
    eps = float(eps)
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0,1).")
    return np.power(10.0, np.linspace(np.log10(eps), 0.0, int(n)))

def precompute_factorials(K: int) -> np.ndarray:
    """
    Returns [0!, 1!, ..., K!] as float64.
    """
    return np.array([factorial(i) for i in range(K + 1)], dtype=float)

def make_log_grids(eps: float = 1e-10, p_init: int = 161, q_init: int = 161) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience to build initial p- and q-grids.
    """
    return logspace01(eps=eps, n=p_init), logspace01(eps=eps, n=q_init)

# ------------------------------ Sup Terms ------------------------------ #

def _f1_vals_and_grads(h: np.ndarray,
                       t: float,
                       r: float,
                       p_grid: np.ndarray,
                       fact: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    First sup-term:
        f1(p; H) = e^{-pt}/p * ( 1 - e^{-rpt} + g_{H^2}(pt) )
    where g_{H^2}(x) = sum_{k>=0} (H_k^2) * x^k / k!.

    Returns:
        vals:  shape (P,)
        grads: shape (P, K+1) for gradient wrt H (including H0, which is zero here)
    """
    K = len(h) - 1
    k = np.arange(K + 1)
    vals = np.empty_like(p_grid, dtype=float)
    grads = np.empty((p_grid.size, K + 1), dtype=float)

    for i, p in enumerate(p_grid):
        y  = p * t
        ep = math.exp(-p * t)
        a  = 1.0 - math.exp(-r * y)                 # 1 - e^{-rpt}
        c  = (y ** k) / fact                         # [1, y, y^2/2!, ...]
        gH2 = np.sum((h ** 2) * c)

        vals[i]   = ep * (a + gH2) / p
        grads[i]  = ep * (2.0 * h * c) / p          # ∂/∂H_k of g_{H^2} = 2 H_k c_k

    return vals, grads

def _f2_vals_and_grads(h: np.ndarray,
                       t: float,
                       r: float,
                       q_grid: np.ndarray,
                       fact: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Second sup-term (squared residual):
        f2(q; H) = [ e^{-qt}/q * ( 1 - e^{-rqt} - g_H(qt) ) ]^2
                 = b * (a - <c, H>)^2
    where b = e^{-2qt}/q^2, a = 1 - e^{-rqt}, c_k = ( (qt)^k / k! ).

    Returns:
        vals:  shape (Q,)
        grads: shape (Q, K+1)
    """
    K = len(h) - 1
    k = np.arange(K + 1)
    vals = np.empty_like(q_grid, dtype=float)
    grads = np.empty((q_grid.size, K + 1), dtype=float)

    for i, q in enumerate(q_grid):
        y  = q * t
        b  = math.exp(-2.0 * q * t) / (q ** 2)      # (e^{-qt}/q)^2
        a  = 1.0 - math.exp(-r * y)
        c  = (y ** k) / fact
        ah = a - float(np.dot(c, h))                # (a - g_H)
        vals[i]  = b * (ah ** 2)
        grads[i] = -2.0 * b * ah * c

    return vals, grads

# ------------------------------ Softmax "sup" ------------------------------ #

def _softmax_max(vals: np.ndarray, tau: float) -> Tuple[float, np.ndarray]:
    """
    Smooth max via log-sum-exp with temperature tau>0.
    Returns (F, s) with F ≈ sup vals and s the softmax weights (sum to 1).
    """
    if tau <= 0:
        raise ValueError("tau must be positive.")
    m = float(np.max(vals))
    w = np.exp((vals - m) / tau)
    Z = float(np.sum(w))
    F = tau * (math.log(Z) + m / tau)
    s = w / Z
    return float(F), s

def _obj_grad_smoothed(h_tail: np.ndarray,
                       t: float,
                       r: float,
                       K: int,
                       p_grid: np.ndarray,
                       q_grid: np.ndarray,
                       fact: np.ndarray,
                       tau1: float,
                       tau2: float) -> Tuple[float, np.ndarray]:
    """
    Objective:
        F(H) = softmax_sup_p f1(p; H) + softmax_sup_q f2(q; H)
    Gradient returned is w.r.t. h_tail = H[1:].
    """
    # Build full H with H0 fixed to 0:
    h = np.zeros(K + 1, dtype=float)
    h[1:] = h_tail

    f1v, f1g = _f1_vals_and_grads(h, t, r, p_grid, fact)
    F1, s1   = _softmax_max(f1v, tau1)
    grad1    = (s1[:, None] * f1g).sum(axis=0)  # shape (K+1,)

    f2v, f2g = _f2_vals_and_grads(h, t, r, q_grid, fact)
    F2, s2   = _softmax_max(f2v, tau2)
    grad2    = (s2[:, None] * f2g).sum(axis=0)  # shape (K+1,)

    F         = F1 + F2
    grad_full = grad1 + grad2
    # drop derivative wrt H0
    return float(F), grad_full[1:].copy()

# ------------------------------ Public API ------------------------------ #


def _logspace_range(a: float, b: float, n: int) -> np.ndarray:
    """
    Like np.logspace but for an arbitrary positive interval [a,b].
    Requires 0 < a <= b.
    """
    a = float(a)
    b = float(b)
    if a <= 0:
        raise ValueError("a must be > 0 for log spacing.")
    return np.exp(np.linspace(np.log(a), np.log(b), int(n)))


def _sup_univariate_adaptive(
    f,
    a: float,
    b: float,
    *,
    n_coarse: int = 512,
    max_locals: int = 16,
    refine_points: int = 64,
) -> tuple[float, float]:
    """
    Approximate sup_{p in [a,b]} f(p) with an adaptive 1D procedure:

      1) coarse log-spaced grid on [a,b],
      2) detect up to max_locals coarse local maxima,
      3) on each candidate interval run a bounded scalar maximization,
      4) return (sup_val, p_argmax).

    All work is done directly in the original parameter (p or q); no variable changes.
    """
    a = float(a)
    b = float(b)
    if not (0.0 < a <= b):
        raise ValueError("Need 0 < a <= b for _sup_univariate_adaptive.")

    # Coarse log-spaced grid
    n_coarse = max(8, int(n_coarse))
    p_grid = _logspace_range(a, b, n_coarse)
    vals = np.array([f(float(p)) for p in p_grid], dtype=float)

    if not np.any(np.isfinite(vals)):
        return float("nan"), float("nan")

    # Candidate indices: endpoints + coarse local maxima
    candidates = [0, len(p_grid) - 1]
    for i in range(1, len(p_grid) - 1):
        if not np.isfinite(vals[i]):
            continue
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            candidates.append(i)

    candidates = list(sorted(set(candidates)))
    if len(candidates) > max_locals:
        candidates = sorted(candidates, key=lambda i: vals[i], reverse=True)[:max_locals]

    def f_safe(p_scalar: float) -> float:
        v = f(p_scalar)
        return float(v) if np.isfinite(v) else -float("inf")

    best_val = -float("inf")
    best_p = None

    for idx in candidates:
        left = p_grid[max(0, idx - 1)]
        right = p_grid[min(len(p_grid) - 1, idx + 1)]
        if right <= left:
            left, right = max(a, left), min(b, right)
        if right <= left:
            continue

        # Maximize f by minimizing -f
        res = minimize_scalar(
            lambda p: -f_safe(p),
            bounds=(left, right),
            method="bounded",
            options={"maxiter": 200, "xatol": (right - left) / refine_points},
        )

        if res.success:
            p_star = float(res.x)
            v_star = f_safe(p_star)
        else:
            p_star = float(p_grid[idx])
            v_star = f_safe(p_star)

        if v_star > best_val:
            best_val = v_star
            best_p = p_star

    # Check endpoints explicitly as well
    for p_end in (a, b):
        v_end = f_safe(p_end)
        if v_end > best_val:
            best_val = v_end
            best_p = p_end

    return best_val, float(best_p)





def score_objective_hard(H: np.ndarray,
                         t: float,
                         r: float,
                         *,
                         eps: float = 1e-10,
                         n_coarse_p: int = 1024,
                         n_coarse_q: int = 1024) -> Dict[str, float]:
    """
    Hard (non-smoothed) evaluation of the same objective that the solver
    approximates with softmax:

        F(H) = sup_p f1(p;H) + sup_q f2(q;H),

    where f1, f2 are EXACTLY those used by _f1_vals_and_grads and
    _f2_vals_and_grads (no change of variables).

    Supremum is taken over p,q ∈ [eps, 1].

    Parameters
    ----------
    H : 1D array-like
        If H[0] is nonzero we forcibly set it to 0, following the solver's convention.
    t : float
    r : float
    eps : float
        Lower bound for p and q (sup over [eps,1]).
    n_coarse_p, n_coarse_q : int
        Coarse grid sizes for p and q in the adaptive search.

    Returns
    -------
    dict with keys:
        F   : sup_p f1 + sup_q f2
        F1  : sup_p f1
        F2  : sup_q f2
        p_argmax_f1, q_argmax_f2 : argmax locations
    """
    H = np.asarray(H, dtype=float).ravel()
    if H.ndim != 1:
        raise ValueError("H must be a 1D array.")

    # Enforce H0 = 0 convention
    if H[0] != 0.0:
        H = H.copy()
        H[0] = 0.0

    K = len(H) - 1
    if K < 0:
        raise ValueError("H must have length at least 1 (for H0).")

    t = float(t)
    r = float(r)
    if t <= 0:
        raise ValueError("t must be positive.")
    if r <= 0:
        raise ValueError("r must be positive.")

    eps = float(eps)
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0,1).")

    fact = precompute_factorials(K)

    # Scalar wrappers around the existing vector f1/f2 implementations
    def f1_scalar(p: float) -> float:
        p_arr = np.array([float(p)], dtype=float)
        vals, _ = _f1_vals_and_grads(H, t, r, p_arr, fact)
        return float(vals[0])

    def f2_scalar(q: float) -> float:
        q_arr = np.array([float(q)], dtype=float)
        vals, _ = _f2_vals_and_grads(H, t, r, q_arr, fact)
        return float(vals[0])

    # Adaptive suprema directly in p and q
    F1, p_star = _sup_univariate_adaptive(
        f1_scalar,
        eps,
        1.0,
        n_coarse=n_coarse_p,
        max_locals=16,
        refine_points=64,
    )

    F2, q_star = _sup_univariate_adaptive(
        f2_scalar,
        eps,
        1.0,
        n_coarse=n_coarse_q,
        max_locals=16,
        refine_points=64,
    )

    F_total = float(F1 + F2)

    return {
        "F": F_total,
        "F1": float(F1),
        "F2": float(F2),
        "p_argmax_f1": float(p_star),
        "q_argmax_f2": float(q_star),
    }


def solve_qsip_adaptive_scipy(t: float,
                              r: float,
                              K: int = 10,
                              *,
                              p_init: int = 161,
                              q_init: int = 161,
                              eps: float = 1e-10,
                              tau1: float = 1e-4,
                              tau2: float = 1e-4,
                              outer_iters: int = 4,
                              x0: Optional[np.ndarray] = None,
                              maxiter_inner: int = 600,
                              refine_mode: str = "halve",
                              p0: float = 0.0) -> np.ndarray:
    """
    Solve for optimized H using SciPy L-BFGS-B with softmax-sup smoothing and
    simple grid refinement, but restrict both suprema to [p0, 1].

    Parameters
    ----------
    t : float
        Current sample size (used as in your scripts).
    r : float
        Extrapolation ratio ( (N-n)/n ). Typically r > 1 for prediction horizon.
    K : int, default 10
        Truncation; we solve for H_1..H_K. H_0 is fixed to 0 (excluded from vars).
    p_init, q_init : int
        Initial sizes of log-spaced p- and q-grids over (lower, 1], where
        lower = max(eps, p0).
    eps : float
        Lower bound to avoid division-by-zero; 0 < eps < 1.
    tau1, tau2 : float
        Softmax temperatures for the two sup terms.
    outer_iters : int
        Number of (solve → refine-grids) cycles.
    x0 : np.ndarray, optional
        Initial guess for H[1:], shape (K,). If None, zeros are used.
    maxiter_inner : int
        Max iterations for each inner L-BFGS-B call.
    refine_mode : {"halve", "none"}
        - "halve": after each inner solve, merge current grid with a halved copy
                   to densify near p0 (but never below p0).
        - "none":  keep grids fixed.
    p0 : float, default 0.0
        Minimum allowed p (and q) in the suprema. Must satisfy 0 <= p0 < 1.

    Returns
    -------
    H : np.ndarray
        Length K+1 vector with H[0] = 0 and optimized H[1:].
    """
    if r <= 0:
        raise ValueError("r must be positive.")
    if t <= 0:
        raise ValueError("t must be positive.")
    if not (0.0 <= p0 < 1.0):
        raise ValueError("p0 must be in [0,1).")

    # lower bound actually used for grids
    lower = max(float(eps), float(p0))

    fact   = precompute_factorials(K)
    p_grid = logspace01(eps=lower, n=p_init)
    q_grid = logspace01(eps=lower, n=q_init)

    if x0 is None:
        h_tail = np.zeros(K, dtype=float)
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (K,):
            raise ValueError(f"x0 must have shape {(K,)}, got {x0.shape}.")
        h_tail = x0.copy()

    for _ in range(int(outer_iters)):
        def fun(x):
            return _obj_grad_smoothed(x, t, r, K, p_grid, q_grid, fact, tau1, tau2)

        res = minimize(fun,
                       h_tail,
                       method="L-BFGS-B",
                       jac=True,
                       options={"maxiter": int(maxiter_inner)})

        h_tail = np.asarray(res.x, dtype=float)

        # refinement, but do NOT go below 'lower'
        if refine_mode == "halve":
            new_p = 0.5 * p_grid
            new_q = 0.5 * q_grid
            # keep only those >= lower
            new_p = new_p[new_p >= lower]
            new_q = new_q[new_q >= lower]
            if new_p.size:
                p_grid = np.unique(np.concatenate([p_grid, new_p]))
            if new_q.size:
                q_grid = np.unique(np.concatenate([q_grid, new_q]))
        elif refine_mode == "none":
            pass
        else:
            raise ValueError('refine_mode must be "halve" or "none".')

    H = np.zeros(K + 1, dtype=float)
    H[1:] = h_tail
    return H


# ------------------------------ Diagnostics ------------------------------ #

def score_objective(H: np.ndarray,
                    t: float,
                    r: float,
                    *,
                    p_grid: Optional[np.ndarray] = None,
                    q_grid: Optional[np.ndarray] = None,
                    eps: float = 1e-10,
                    p_init: int = 2001,
                    q_init: int = 2001,
                    tau1: float = 1e-6,
                    tau2: float = 1e-6) -> Dict[str, float]:
    """
    Evaluate the smoothed objective F(H) = softmax_sup_p f1 + softmax_sup_q f2.

    Smaller tau gives a tighter upper bound to the true sup (but may be less smooth).

    Returns a dict with:
        {
          "F": total objective,
          "F1": first term,
          "F2": second term
        }
    """
    H = np.asarray(H, dtype=float)
    if H.ndim != 1:
        raise ValueError("H must be a 1D array.")
    if H[0] != 0.0:
        # Enforce the convention H0=0 (the solver always returns H0=0)
        H = H.copy()
        H[0] = 0.0

    K    = len(H) - 1
    fact = precompute_factorials(K)

    if p_grid is None:
        p_grid = logspace01(eps=eps, n=p_init)
    if q_grid is None:
        q_grid = logspace01(eps=eps, n=q_init)

    f1v, _ = _f1_vals_and_grads(H, t, r, p_grid, fact)
    F1, _  = _softmax_max(f1v, tau1)
    f2v, _ = _f2_vals_and_grads(H, t, r, q_grid, fact)
    F2, _  = _softmax_max(f2v, tau2)

    return {"F": float(F1 + F2), "F1": float(F1), "F2": float(F2)}
