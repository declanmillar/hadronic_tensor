"""Corrected quantitative panel: 101-qubit W^{00} ridge vs the parameter-free
two-band model, identical one-sided narrow-window transforms on both sides
(sigma_x = 0.83 kills |x| > 2, where the model and data correlators are both
volume-converged -- no volume extension, no normalization).

Also: integral Ward/continuity check on the raw production grids,
  C00(x,t) - C00(x,0) = - int_0^t dt' [C10(x+1/4,t') - C10(x-1/4,t')]
via Simpson integration (error ~ (w*dt)^4/180 ~ 2%), which avoids the ~30%
sinc bias of naive time differentiation.

  PYTHONPATH=. .venv/bin/python scripts/final_ridge_overlay.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_simpson

from htensor import Z2Lattice, exact, spectroscopy, analysis
from htensor import currents as cur
from scripts_helpers_ridge import two_band_space, model_correlator, onesided_ft

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
PALE = ["#80B9D9", "#EAAF80"]              # 50%-white versions of OI[0], OI[1]
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()
plt.rcParams.update({"axes.labelsize": 10})

M0, G2, ETA = 0.7, 1.1, 1.3
SIG_T, SIG_X = 8.0 / 3.0, 0.83
Q1S = np.array([2 * np.pi / 5, -2 * np.pi / 5, 4 * np.pi / 5, -4 * np.pi / 5])
Q0 = np.arange(-0.6, 1.601, 0.04)

# Plot-only fast path: transformed model/data curves cached to npz; pass
# --recompute to re-derive the two-band model from scratch (~10 min).
import sys as _sys
_CACHE = "data/ridge_overlay_cache.npz"
FAST = _os.path.exists(_CACHE) and "--recompute" not in _sys.argv
if FAST:
    _c = np.load(_CACHE)
    model_W = {0.0: _c["model_rest"], 2 * np.pi / 5: _c["model_boost"]}
    data_W = {0.0: _c["data_rest"], 2 * np.pi / 5: _c["data_boost"]}
    print(f"[fast path] loaded {_CACHE}; --recompute to re-derive")

# ---------------- ns=10 two-band model correlators
if not FAST:
    lat10 = Z2Lattice(10, pbc=True)
    sts, Ek, vac10, e0 = two_band_space(lat10, M0, G2, ETA)
    band1 = spectroscopy.meson_band(lat10, M0, G2, ETA, n_states=14,
                                    matrix_free=True)
    TIMES = np.arange(0.0, 8.01, 0.5)
    x10 = analysis.ring_fold((np.arange(lat10.ns) - 4) / 2, lat10.nx)
    model_W = {}
    for k0 in (0.0, 2 * np.pi / 5):
        mix = spectroscopy.optimize_interpolator(lat10, band1, k0=k0,
                                                 sigma_x=0.75, x0=2)
        wp, _ = spectroscopy.meson_wavepacket(lat10, band1, k0=k0,
                                              sigma_x=0.75, x0=2,
                                              mix=mix["mix"])
        f = np.array([np.vdot(s, wp) for s in sts])
        Gm, _ = model_correlator(lat10, sts, Ek, f, 4, TIMES, M0, G2, ETA)
        model_W[k0] = onesided_ft(TIMES, x10, Gm, Q0, Q1S, SIG_T, SIG_X,
                                  0.5, 0.5)
        print(f"model k0={k0:.2f} ready", flush=True)

    # ------------ production data, same one-sided transform
    data_W = {}
    ward = {}
    for k0, path in ((0.0, "data/w_meson_ns50_k0.00_v3.npz"),
                     (2 * np.pi / 5, "data/w_meson_ns50_k1.26_v3.npz")):
        d = np.load(path)
        ns, vc = int(d["ns"]), int(d["center"])
        lat = Z2Lattice(ns, pbc=True)
        times = d["times"]

        def conn(corr, one, ins):
            return analysis.subtract(corr, None, one, complex(ins))

        G = conn(d["corr_wp"], d["one_pt_wp"], d["insert_1pt_wp"]) \
            - conn(d["corr_vac"], d["one_pt_vac"], d["insert_1pt_vac"])
        x = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
        data_W[k0] = onesided_ft(times, x, G[:, :ns], Q0, Q1S, SIG_T, SIG_X,
                                 0.5, 0.5)

        # integral Ward check on raw wp grids
        c00 = d["corr_wp"][:, :ns]
        c10 = d["corr_wp"][:, ns:]
        div = c10 - np.roll(c10, 1, axis=1)
        lhs = c00 - c00[0][None, :]
        rhs = -cumulative_simpson(div, x=times, axis=0, initial=0.0)
        ward[k0] = float(np.abs(lhs - rhs).max() / np.abs(lhs).max())
        print(f"data k0={k0:.2f}: integral Ward residual = {ward[k0]:.3f}",
              flush=True)

    np.savez(_CACHE, model_rest=model_W[0.0],
             model_boost=model_W[2 * np.pi / 5], data_rest=data_W[0.0],
             data_boost=data_W[2 * np.pi / 5], Q0=Q0, Q1S=Q1S,
             ward_rest=ward[0.0], ward_boost=ward[2 * np.pi / 5])
    print(f"cached transformed curves -> {_CACHE}")

# ---------------- figure: single axes, all four curves + models.
# hue = sign of q1; thick/saturated = boosted packet, thin/pale = rest
# (the rest pair is nearly degenerate -- the null -- while the boost
# splits the +-q1 peaks; one axes makes that contrast direct).
fig, ax = plt.subplots(figsize=(3.8, 2.65), constrained_layout=True)
from matplotlib.lines import Line2D
for k0, cols, lw in ((0.0, PALE, 2.2), (2 * np.pi / 5, OI, 2.2)):
    for i, q in enumerate([2 * np.pi / 5, -2 * np.pi / 5]):
        j = np.argmin(np.abs(Q1S - q))
        ax.plot(Q0, data_W[k0][:, j], color=cols[i], lw=lw)
        # model: dark dashes riding on each colored data curve -- visible
        # against pale and saturated alike, so both layers clearly exist
        ax.plot(Q0, model_W[k0][:, j], color="0.15", lw=0.9,
                ls=(0, (4, 3)), zorder=6)
ax.axhline(0, color="0.4", lw=0.7)
ax.set_ylim(None, 0.62)                      # headroom so the legend floats free
ax.set_xlabel(r"$q^0$")
ax.set_ylabel(r"$W^{00}$ (one-sided, $\sigma_x = 0.83$)")
handles = [
    Line2D([], [], color=OI[0], lw=2.2, label=r"boosted, $q^1{=}{+}2\pi/5$"),
    Line2D([], [], color=OI[1], lw=2.2, label=r"boosted, $q^1{=}{-}2\pi/5$"),
    Line2D([], [], color=PALE[0], lw=2.2, label=r"rest, $q^1{=}{+}2\pi/5$"),
    Line2D([], [], color=PALE[1], lw=2.2, label=r"rest, $q^1{=}{-}2\pi/5$"),
    Line2D([], [], color="0.15", lw=0.9, ls=(0, (4, 3)),
           label="two-band model"),
]
ax.legend(handles=handles, fontsize=6.5, loc="upper right",
          framealpha=0.9)
fig.savefig("data/w_ridge_final.pdf", dpi=200)
print("wrote data/w_ridge_final.pdf")

for k0 in (0.0, 2 * np.pi / 5):
    for q in Q1S:
        j = np.argmin(np.abs(Q1S - q))
        r = data_W[k0][:, j] - model_W[k0][:, j]
        print(f"k0={k0:.2f} q1={q:+.2f}: rms(data-model)/max = "
              f"{np.sqrt(np.mean(r**2)) / np.abs(data_W[k0][:, j]).max():.3f}")
