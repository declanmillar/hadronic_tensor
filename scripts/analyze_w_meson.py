"""Assemble W^{00}(q0,q1) from a meson-wavepacket production run, check the
lattice continuity equation on the raw grids, and produce the figure.

  PYTHONPATH=. .venv/bin/python scripts/analyze_w_meson.py data/w_meson_ns50_k0.00_v3.npz
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice, analysis

OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()

path = sys.argv[1]
d = np.load(path)
ns, vc, k0 = int(d["ns"]), int(d["center"]), float(d["k0"])
lat = Z2Lattice(ns, pbc=True)
times = d["times"]
nJ0 = ns  # probes: J0 at all sites, then J1 at all bonds

# connected-minus-connected: [C_wp - <J><J>_wp] - [C_vac - <J><J>_vac]
def connected(corr, one_pt, ins):
    return analysis.subtract(corr, None, one_pt, complex(ins))

g_wp = connected(d["corr_wp"], d["one_pt_wp"], d["insert_1pt_wp"])
g_vac = connected(d["corr_vac"], d["one_pt_vac"], d["insert_1pt_vac"])
G = g_wp - g_vac

# ---------------- continuity check on raw (unsubtracted) wp grids
c00 = d["corr_wp"][:, :nJ0]
c10 = d["corr_wp"][:, nJ0:]
dt = times[1] - times[0]
lhs = (c00[2:] - c00[:-2]) / (2 * dt)          # d/dt C^{00}(v, t)
div = c10 - np.roll(c10, 1, axis=1)            # J1_{v+1/2} - J1_{v-1/2}
rhs = -div[1:-1]
num = np.abs(lhs - rhs).max()
den = np.abs(lhs).max()
print(f"continuity check (raw data): max|dC00/dt + div C10| = {num:.4f}, "
      f"max|dC00/dt| = {den:.4f}, ratio = {num/den:.3f} "
      f"(expect ~O(dt^2 + Trotter) ~ few %)")

# ---------------- W^{00} (one-sided transform: the hermiticity completion is
# invalid for wavepacket states -- see notes/adversarial_review_2026-07-06.txt
# finding A-F1 and the ridge-model validation record)
from scripts_helpers_ridge import onesided_ft

x0 = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
q0 = np.arange(-1.0, 6.001, 0.04)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)  # +-q1: boost asymmetry
q1 = 2 * np.pi * ks / lat.nx
Ws = [onesided_ft(times, x0, G[:, :nJ0], q0, q1,
                  f * times[-1] / 3, f * lat.nx / 6, 0.5, 0.5)
      for f in (0.75, 1.0, 1.5)]
W = Ws[1].astype(complex)
spread = np.ptp(np.stack(Ws), axis=0)
print(f"W^00: max|Im|/max|Re| = {np.abs(W.imag).max()/np.abs(W.real).max():.4f}")
np.savez(path.replace(".npz", "_W.npz"), q0=q0, q1=q1, W=W, spread=spread, k0=k0)

# ---------------- figure
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.8),
                               constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.05, 1]})
pm = axL.pcolormesh(q0, q1, W.real.T, cmap="Blues", shading="nearest",
                    vmin=0, vmax=np.percentile(W.real, 99.5), rasterized=True)
axL.set_xlabel(r"$q^0$")
axL.set_ylabel(r"$q^1$")
axL.set_title(rf"$W^{{00}}(q^0,q^1)$ meson packet, $k_0={k0:.2f}$",
              fontsize=10.5)
axL.grid(False)
fig.colorbar(pm, ax=axL, pad=0.02)

for i, j in enumerate([lat.nx // 2 + 2, lat.nx // 2 + 5,
                       lat.nx // 2 + 8, lat.nx // 2 + 12]):
    axR.fill_between(q0, W.real[:, j] - spread[:, j] / 2,
                     W.real[:, j] + spread[:, j] / 2,
                     color=OI[i], alpha=0.22, lw=0)
    axR.plot(q0, W.real[:, j], color=OI[i], lw=1.8,
             label=rf"$q^1={q1[j]:.2f}$")
axR.axhline(0, color="0.4", lw=0.7)
axR.set_xlabel(r"$q^0$")
axR.set_ylabel(r"$W^{00}(q^0,q^1)$")
axR.legend(fontsize=8.5, framealpha=0.9)
axR.set_title("cuts; band = window systematic", fontsize=10.5)
out = path.replace(".npz", "_W.pdf")
fig.savefig(out, dpi=200)
print("wrote", out)
