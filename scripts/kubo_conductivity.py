"""Optical conductivity of the Z2 gauge vacuum from the measured charge
response, via the exact bond-current Ward identity:

    W^{11}(w, q) = (w/q)^2 W^{00}(w, q)   =>   Re sigma(w) = w W^{00}/q^2 |_{q->0}

Uses the 101-qubit vacuum-polarization data (task 5 / broaden-1).  Also
checks the equal-time sum rule  int dq0/2pi W^{00}(q0, q1) = S(q1)  against
the directly measured t=0 correlator (quantifies window bias).

  PYTHONPATH=. .venv/bin/python scripts/kubo_conductivity.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice, analysis

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()

d = np.load("data/w00_vac_ns50.npz")
ns, vc = int(d["ns"]), int(d["vc"])
lat = Z2Lattice(ns, pbc=True)
times = d["times"]

c_conn = analysis.subtract(d["corr"], None, d["probe_1pt"], complex(d["insert_1pt"]))
x = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
grid = analysis.CorrelatorGrid(times, x, c_conn)
t_full, c_full = analysis.complete_time(grid)

q0 = np.arange(0.0, 5.501, 0.02)
ks = np.array([1, 2, 3])
q1 = 2 * np.pi * ks / lat.nx
sigma_t, sigma_x = times[-1] / 3.0, lat.nx / 6.0
W, spread = analysis.window_scan(t_full, x, c_full, q0, q1, sigma_t, sigma_x)

# ---- sum-rule check: int dq0/2pi W vs direct t=0 structure factor
q0_full = np.arange(-8.0, 8.001, 0.02)
Wf = analysis.windowed_ft(t_full, x, c_full, q0_full, q1, sigma_t, sigma_x)
lhs = np.trapezoid(Wf.real, q0_full, axis=0) / (2 * np.pi)
dx = 0.5
rhs = np.array([np.sum(dx * np.exp(-1j * q * x) *
                       np.exp(-x**2 / (2 * sigma_x**2)) * c_conn[0]).real
                for q in q1])
print("sum rule  int dq0/2pi W  vs  windowed S(q1):")
for j, q in enumerate(q1):
    print(f"  q1={q:.3f}:  {lhs[j]:+.5f}  vs  {rhs[j]:+.5f}"
          f"   (ratio {lhs[j]/rhs[j]:.4f})")

# ---- conductivity
fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
for j, (k, q) in enumerate(zip(ks, q1)):
    # exact lattice Ward factor: the point-split current gives the lattice
    # momentum qhat = 2 sin(q1/4), not the continuum q1 (they differ by ~2,
    # a ~4x factor in sigma, persisting as q->0 since qhat -> q1/2)
    qhat = 2 * np.sin(q / 4)
    sig = q0 * W.real[:, j] / qhat**2
    band = q0 * spread[:, j] / 2 / qhat**2
    ax.fill_between(q0, sig - band, sig + band, color=OI[j], alpha=0.2, lw=0)
    ax.plot(q0, sig, color=OI[j], lw=1.8,
            label=rf"$q^1 = 2\pi\cdot{k}/25 = {q:.2f}$")
ax.axvline(2.7451, color="0.35", lw=0.9, ls="--")
ax.text(2.70, ax.get_ylim()[1] * 0.95, r"$M$", color="0.25",
        fontsize=10, ha="right")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"$\mathrm{Re}\,\sigma(\omega) = \omega\, W^{00}/"
              r"(2\sin(q^1/4))^2$")
ax.set_title(r"optical conductivity, $N_s=50$ vacuum (Ward-converted)",
             fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
fig.savefig("data/sigma_vac_ns50.pdf", dpi=200)
np.savez("data/sigma_vac_ns50.npz", q0=q0, q1=q1,
         sigma=(q0[:, None] * W.real / q1[None, :]**2), spread=spread)
print("wrote data/sigma_vac_ns50.pdf")
