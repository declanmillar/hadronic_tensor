"""From measured two-current correlators to the hadronic tensor W^{mu nu}(q).

Chain (PLAN.md sec. 2, items 7-11):
  1. pointwise vacuum subtraction:      C -> C - <Omega|J J|Omega>
  2. pointwise disconnected subtraction: C -> C - <J^mu(x,t)><J^nu(0)>
  3. time completion via hermiticity + translation invariance:
       G^{mu nu}(x, -t) = [C^{nu mu}(-x, t)]*   (exact for |P>, O(sigma_p^2)
       for a wavepacket centered on the insertion)
  4. Gaussian-windowed double Fourier transform:
       W(q0, q1) = sum_{t,x} dt dx e^{i(q0 t - q1 x)}
                   e^{-t^2/(2 st^2) - x^2/(2 sx^2)} G(x, t)
     with window widths scanned as a systematic.

Positions x are in spatial-site units relative to the insertion point,
minimal-image folded on the ring.  W is the plain-ordered-product transform
(the DIS-relevant object), NOT Re of the T-ordered transform.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrelatorGrid:
    """Vacuum-/disconnected-subtracted correlator on a (t >= 0, x) grid."""

    times: np.ndarray  # uniform, starting at 0
    x: np.ndarray      # spatial units rel. to insertion, minimal-image folded
    c: np.ndarray      # (nt, nx) complex

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.c = np.asarray(self.c, dtype=complex)
        assert self.c.shape == (len(self.times), len(self.x))
        assert np.isclose(self.times[0], 0.0)


def ring_fold(x: np.ndarray, nx_sites: int) -> np.ndarray:
    """Minimal-image position on a ring of nx_sites spatial sites."""
    x = np.asarray(x, dtype=float)
    return (x + nx_sites / 2) % nx_sites - nx_sites / 2


def subtract(c_raw, c_vac=None, probe_1pt=None, insert_1pt=None):
    """Pointwise vacuum and disconnected subtractions (both matter: the
    interacting vacuum carries a staggered <J^0(v)> profile)."""
    c = np.asarray(c_raw, dtype=complex).copy()
    if c_vac is not None:
        c = c - np.asarray(c_vac, dtype=complex)
    if probe_1pt is not None and insert_1pt is not None:
        c = c - np.asarray(probe_1pt, dtype=complex) * complex(insert_1pt)
    return c


def complete_time(grid: CorrelatorGrid, grid_transposed: CorrelatorGrid | None = None):
    """Extend to t < 0 using G^{mu nu}(x, -t) = [C^{nu mu}(-x, t)]*.

    grid_transposed holds C^{nu mu} on the same (t, x) grid; for mu = nu pass
    None and the grid itself is used.  Requires the x grid to be mirror
    symmetric (x and -x both present after ring folding).
    Returns (t_full, c_full) with t from -T to +T.
    """
    gt = grid_transposed if grid_transposed is not None else grid
    # mirror map on x indices
    mirror = np.empty(len(grid.x), dtype=int)
    for i, xv in enumerate(grid.x):
        j = np.where(np.isclose(gt.x, -xv))[0]
        if len(j) == 0:  # ring edge: -x folds back onto +x (the L/2 point)
            j = np.where(np.isclose(np.abs(gt.x), np.abs(xv)))[0]
        if len(j) == 0:
            raise ValueError(f"x grid not mirror symmetric at x={xv}")
        mirror[i] = j[0]
    neg = np.conj(gt.c[1:, mirror])  # t = dt..T, x -> -x, conjugated
    t_full = np.concatenate([-grid.times[1:][::-1], grid.times])
    c_full = np.concatenate([neg[::-1], grid.c], axis=0)
    return t_full, c_full


def windowed_ft(t_full, x, c_full, q0s, q1s, sigma_t, sigma_x):
    """Gaussian-windowed double FT; returns W[(q0, q1)] complex.

    The window must WIDEN toward the all-data limit (L, T -> inf before
    sigma -> inf); in practice scan sigma_t, sigma_x by ~2x for systematics.
    """
    t_full = np.asarray(t_full, dtype=float)
    x = np.asarray(x, dtype=float)
    q0s, q1s = np.atleast_1d(q0s).astype(float), np.atleast_1d(q1s).astype(float)
    dt = np.median(np.diff(t_full))
    dx = np.median(np.diff(np.sort(x)))
    wt = np.exp(-t_full**2 / (2 * sigma_t**2))
    wx = np.exp(-x**2 / (2 * sigma_x**2))
    weighted = (wt[:, None] * wx[None, :]) * c_full
    # W[i,j] = sum_t sum_x e^{i q0 t} e^{-i q1 x} weighted
    et = np.exp(1j * np.outer(q0s, t_full))          # (nq0, nt)
    ex = np.exp(-1j * np.outer(x, q1s))              # (nx, nq1)
    return dt * dx * (et @ weighted @ ex)


def window_scan(t_full, x, c_full, q0s, q1s, sigma_t, sigma_x,
                factors=(0.75, 1.0, 1.5)):
    """W at several window widths; central value + spread as the windowing
    systematic."""
    ws = np.stack([windowed_ft(t_full, x, c_full, q0s, q1s, f * sigma_t,
                               f * sigma_x) for f in factors])
    return ws[len(factors) // 2], np.ptp(ws.real, axis=0)


def onesided_ft(times, x, G, q0, q1, sigma_t, sigma_x, dt, dx):
    """One-sided (t >= 0) Gaussian-windowed transform, t = 0 half-weighted:
    W1 = 2 Re sum_{t,x} dt dx e^{i(q0 t - q1 x)} w(t) w(x) G(x, t).

    This is the production transform for wavepacket states (a controlled
    hermiticity completion; see main_v4 sec. HT and the 2026-07 referee
    record).  Moved here from scripts_helpers_ridge.py unchanged."""
    times = np.asarray(times, float)
    wt = np.exp(-times**2 / (2 * sigma_t**2)).copy()
    wt[0] *= 0.5
    wx = np.exp(-np.asarray(x, float)**2 / (2 * sigma_x**2))
    et = np.exp(1j * np.outer(np.atleast_1d(q0), times))
    ex = np.exp(-1j * np.outer(np.asarray(x, float), np.atleast_1d(q1)))
    return 2 * np.real(dt * dx * (et @ ((wt[:, None] * wx[None, :]) * G) @ ex))
