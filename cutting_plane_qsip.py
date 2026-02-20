#!/usr/bin/env python3
"""
cutting_plane_qsip.py
=====================
Self-contained, vectorised cutting-plane (exchange) solver for the QSIP
minimax problem.

Key features:
  - Vectorised f1 / f2 evaluation via broadcast NumPy (no Python loops).
  - Stable Poisson-weight computation via recurrence c[k] = c[k-1]*y/k,
    avoiding intermediate overflow from np.power(y, k) for large t*p.
  - Cutting-plane (exchange) outer loop with provable SIP gap certificate.
  - Tight L-BFGS-B tolerances (ftol, gtol) to prevent premature stops.
  - Adaptive sup-finder resolution scaled to the problem.
  - Post-solve verification at high resolution.

Usage
-----
    from cutting_plane_qsip import solve_qsip_cutting_plane
    H = solve_qsip_cutting_plane(t=50.0, r=2.0, K=10)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar


# ===================================================================
#  Utilities (self-contained — no external imports needed)
# ===================================================================

def logspace01(eps: float = 1e-10, n: int = 100) -> np.ndarray:
    """Return *n* points log-spaced in [eps, 1]."""
    return np.exp(np.linspace(np.log(eps), 0.0, n))


def _softmax_max(vals: np.ndarray, tau: float) -> Tuple[float, np.ndarray]:
    """
    Smooth approximation to max(vals) via the log-sum-exp trick.

    Returns
    -------
    value : float
        tau * log( sum( exp(vals / tau) ) )  (numerically stabilised).
    weights : ndarray, shape (len(vals),)
        Softmax weights  w_i = exp(vals_i / tau) / sum(exp(vals / tau)).
    """
    v = vals / tau
    v_max = np.max(v)
    ev = np.exp(v - v_max)
    s = np.sum(ev)
    value = float(tau * (v_max + np.log(s)))
    weights = ev / s
    return value, weights


# ===================================================================
#  Stable Poisson-weight matrix
# ===================================================================

def _poisson_weights(y: np.ndarray, K: int) -> np.ndarray:
    """
    Compute  c[i, k] = y[i]^k / k!  for k = 0..K  via the recurrence

        c[:, 0] = 1
        c[:, k] = c[:, k-1] * y / k

    This avoids computing y^k and k! independently, which can overflow
    for large y even though their ratio is representable.
    """
    P = y.shape[0]
    c = np.empty((P, K + 1), dtype=float)
    c[:, 0] = 1.0
    for k in range(1, K + 1):
        c[:, k] = c[:, k - 1] * (y / k)
    return c


# ===================================================================
#  Vectorised f1 / f2
# ===================================================================

def _f1_vec(h: np.ndarray, t: float, r: float,
            p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised first sup-term.

        f1(p; H) = e^{-pt}/p * (1 - e^{-rpt} + g_{H^2}(pt))

    Returns (vals, grads) with shapes (P,) and (P, K+1).
    """
    K = len(h) - 1
    y   = p * t
    ep  = np.exp(-y)
    a   = 1.0 - np.exp(-r * y)
    c   = _poisson_weights(y, K)

    gH2   = c @ (h * h)
    coeff = ep / p
    vals  = coeff * (a + gH2)
    grads = coeff[:, None] * (2.0 * h[None, :] * c)
    return vals, grads


def _f2_vec(h: np.ndarray, t: float, r: float,
            q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised second sup-term.

        f2(q; H) = (e^{-qt}/q)^2 * (1 - e^{-rqt} - g_H(qt))^2

    Returns (vals, grads) with shapes (Q,) and (Q, K+1).
    """
    K = len(h) - 1
    y  = q * t
    b  = np.exp(-2.0 * y) / (q * q)
    a  = 1.0 - np.exp(-r * y)
    c  = _poisson_weights(y, K)
    ah = a - c @ h

    vals  = b * ah * ah
    grads = (-2.0 * b * ah)[:, None] * c
    return vals, grads


# ===================================================================
#  Vectorised smoothed objective
# ===================================================================

def _obj_grad_vec(h_tail: np.ndarray,
                  t: float, r: float, K: int,
                  p_grid: np.ndarray, q_grid: np.ndarray,
                  tau1: float, tau2: float) -> Tuple[float, np.ndarray]:
    """Objective + gradient using vectorised f1/f2."""
    h = np.zeros(K + 1, dtype=float)
    h[1:] = h_tail

    f1v, f1g = _f1_vec(h, t, r, p_grid)
    F1, s1   = _softmax_max(f1v, tau1)
    grad1    = s1 @ f1g

    f2v, f2g = _f2_vec(h, t, r, q_grid)
    F2, s2   = _softmax_max(f2v, tau2)
    grad2    = s2 @ f2g

    return float(F1 + F2), (grad1 + grad2)[1:].copy()


# ===================================================================
#  Vectorised 1-D sup finder
# ===================================================================

def _find_sup_vec(h: np.ndarray, t: float, r: float,
                  lower: float,
                  term: str = "f1",
                  n_grid: int = 500,
                  extra_pts: Optional[np.ndarray] = None,
                  ) -> Tuple[float, float, float]:
    """
    Find  sup_{x in [lower, 1]} f(x; H)  via dense vectorised grid + local
    scalar refinement around every coarse local maximum.

    Parameters
    ----------
    extra_pts : optional array
        Additional points to include in the coarse grid (e.g. the current
        active set), guaranteeing the returned sup >= max over those points.

    Returns
    -------
    best_val, best_x, active_max
    """
    grid = np.exp(np.linspace(np.log(lower), 0.0, n_grid))
    _eval = _f1_vec if term == "f1" else _f2_vec

    active_max = -np.inf
    if extra_pts is not None and len(extra_pts) > 0:
        avals, _ = _eval(h, t, r, extra_pts)
        active_max = float(np.max(avals))
        grid = np.unique(np.concatenate([grid, extra_pts]))

    vals, _ = _eval(h, t, r, grid)

    cands = {0, len(grid) - 1}
    for i in range(1, len(grid) - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            cands.add(i)

    def _f_scalar(x):
        v, _ = _eval(h, t, r, np.array([x]))
        return float(v[0])

    best_val, best_x = -np.inf, lower
    for idx in cands:
        lo = grid[max(0, idx - 1)]
        hi = grid[min(len(grid) - 1, idx + 1)]
        if hi <= lo:
            v = float(vals[idx])
            if v > best_val:
                best_val, best_x = v, float(grid[idx])
            continue
        res = minimize_scalar(lambda x: -_f_scalar(x),
                              bounds=(lo, hi), method="bounded",
                              options={"maxiter": 200})
        v = _f_scalar(float(res.x))
        if v > best_val:
            best_val, best_x = v, float(res.x)

    for x_end in (lower, 1.0):
        v = _f_scalar(x_end)
        if v > best_val:
            best_val, best_x = v, x_end

    return best_val, best_x, active_max


# ===================================================================
#  Active-set helpers
# ===================================================================

def _neighbourhood_log(center: float, lower: float, upper: float = 1.0,
                       n: int = 6, half_decades: float = 0.15) -> np.ndarray:
    """*n* points in a log-space neighbourhood of *center*."""
    lc = np.log10(max(center, lower))
    lo = max(np.log10(lower), lc - half_decades)
    hi = min(0.0, lc + half_decades)
    if lo >= hi:
        return np.array([max(center, lower)])
    return np.power(10.0, np.linspace(lo, hi, n))


def _prune_active_set(pts: np.ndarray, scores: np.ndarray,
                      max_size: int) -> np.ndarray:
    """Prune to *max_size*: top-half by score + spread for coverage."""
    if len(pts) <= max_size:
        return pts

    n_top = max_size // 2
    n_spread = max_size - n_top

    top_idx = set(np.argsort(scores)[-n_top:])
    remaining = np.array([i for i in range(len(pts)) if i not in top_idx])
    if len(remaining) > 0 and n_spread > 0:
        step = max(1, len(remaining) // n_spread)
        spread_idx = remaining[::step][:n_spread]
        keep = np.array(sorted(top_idx | set(spread_idx)))
    else:
        keep = np.array(sorted(top_idx))

    return np.sort(pts[keep])


# ===================================================================
#  Main solver
# ===================================================================

def solve_qsip_cutting_plane(
    t: float,
    r: float,
    K: int = 10,
    *,
    eps: float = 1e-10,
    n_init: int = 120,
    tau: float = 1e-5,
    tau_final: float = 1e-7,
    tol_gap: float = 1e-10,
    max_outer: int = 20,
    min_outer: int = 3,
    maxiter_inner: int = 500,
    max_active: int = 250,
    x0: Optional[np.ndarray] = None,
    p0: float = 0.0,
) -> np.ndarray:
    """
    Vectorised cutting-plane solver for the QSIP minimax.

    Parameters
    ----------
    t : float
        Time parameter (must be > 0).
    r : float
        Rate parameter (must be > 0).
    K : int
        Polynomial order (number of h-coefficients is K+1, with h[0]=0).
    eps : float
        Small positive lower bound on the search domain.
    n_init : int
        Initial active-set size (log-spaced).
    tau, tau_final : float
        Softmax temperature schedule.
    tol_gap : float
        Relative SIP gap tolerance for convergence.
    max_outer / min_outer : int
        Max / minimum cutting-plane iterations.
    max_active : int
        Hard cap on active-set size.
    x0 : array-like, optional
        Initial guess for h[1:K+1].  Zeros if not supplied.
    p0 : float
        Lower bound on the search domain (default 0).

    Returns
    -------
    H : ndarray, shape (K+1,)
        Optimal coefficients with H[0] = 0.
    """
    if r <= 0:
        raise ValueError("r must be positive.")
    if t <= 0:
        raise ValueError("t must be positive.")
    if not (0.0 <= p0 < 1.0):
        raise ValueError("p0 must be in [0,1).")

    lower = max(float(eps), float(p0))

    n_decades = -np.log10(lower)
    n_sup_grid = int(np.clip(80 * n_decades, 500, 3000))

    S_p = logspace01(eps=lower, n=n_init)
    S_q = logspace01(eps=lower, n=n_init)

    h_tail = np.zeros(K, dtype=float) if x0 is None \
             else np.asarray(x0, dtype=float).copy()

    true_obj = 0.0

    for outer in range(max_outer):
        progress = outer / max(1, max_outer - 1)
        cur_tau = tau * (tau_final / tau) ** progress

        _sp, _sq = S_p.copy(), S_q.copy()

        def fun(x, _tau=cur_tau, _sp=_sp, _sq=_sq):
            return _obj_grad_vec(x, t, r, K, _sp, _sq, _tau, _tau)

        res = minimize(fun, h_tail, method="L-BFGS-B", jac=True,
                       options={"maxiter": maxiter_inner,
                                "ftol": 1e-30, "gtol": 1e-12})
        h_tail = np.asarray(res.x, dtype=float)

        H = np.zeros(K + 1, dtype=float)
        H[1:] = h_tail

        f1_sup, p_star, f1_active_max = \
            _find_sup_vec(H, t, r, lower, "f1",
                          n_grid=n_sup_grid, extra_pts=S_p)
        f2_sup, q_star, f2_active_max = \
            _find_sup_vec(H, t, r, lower, "f2",
                          n_grid=n_sup_grid, extra_pts=S_q)

        true_obj   = f1_sup + f2_sup
        active_obj = f1_active_max + f2_active_max

        gap = max(0.0, true_obj - active_obj)

        if outer >= min_outer:
            if gap < tol_gap * max(1.0, abs(true_obj)):
                break

        new_ps = _neighbourhood_log(p_star, lower)
        new_qs = _neighbourhood_log(q_star, lower)
        S_p = np.unique(np.concatenate([S_p, new_ps]))
        S_q = np.unique(np.concatenate([S_q, new_qs]))

        if len(S_p) > max_active:
            sc, _ = _f1_vec(H, t, r, S_p)
            S_p = _prune_active_set(S_p, sc, max_active)
        if len(S_q) > max_active:
            sc, _ = _f2_vec(H, t, r, S_q)
            S_q = _prune_active_set(S_q, sc, max_active)

    # Final polish
    def fun_final(x):
        return _obj_grad_vec(x, t, r, K, S_p, S_q, tau_final, tau_final)

    res = minimize(fun_final, h_tail, method="L-BFGS-B", jac=True,
                   options={"maxiter": maxiter_inner,
                            "ftol": 1e-30, "gtol": 1e-12})
    h_tail = np.asarray(res.x, dtype=float)
    H = np.zeros(K + 1, dtype=float)
    H[1:] = h_tail

    # Post-solve verification at high resolution
    n_verify = max(n_sup_grid, 2000)
    f1v, p1v, _ = _find_sup_vec(H, t, r, lower, "f1", n_grid=n_verify)
    f2v, q2v, _ = _find_sup_vec(H, t, r, lower, "f2", n_grid=n_verify)
    verify_obj = f1v + f2v

    if verify_obj > true_obj * (1.0 + 1e-6) + 1e-12:
        for pt in _neighbourhood_log(p1v, lower):
            S_p = np.unique(np.append(S_p, pt))
        for pt in _neighbourhood_log(q2v, lower):
            S_q = np.unique(np.append(S_q, pt))

        def fun_fix(x):
            return _obj_grad_vec(x, t, r, K, S_p, S_q, tau_final, tau_final)

        res = minimize(fun_fix, h_tail, method="L-BFGS-B", jac=True,
                       options={"maxiter": maxiter_inner,
                                "ftol": 1e-30, "gtol": 1e-12})
        h_tail = np.asarray(res.x, dtype=float)
        H = np.zeros(K + 1, dtype=float)
        H[1:] = h_tail

    return H


# ===================================================================
#  Smoke test
# ===================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=8, linewidth=120)

    print("Running smoke test ...")
    H = solve_qsip_cutting_plane(t=50.0, r=2.0, K=10)
    print(f"H = {H}")

    lower = 1e-10
    f1v, _, _ = _find_sup_vec(H, 50.0, 2.0, lower, "f1", n_grid=2000)
    f2v, _, _ = _find_sup_vec(H, 50.0, 2.0, lower, "f2", n_grid=2000)
    print(f"F = {f1v + f2v:.10e}  (F1 = {f1v:.6e}, F2 = {f2v:.6e})")
    print("Done.")