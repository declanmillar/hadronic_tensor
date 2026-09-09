"""Figure: vacuum-polarization W^{00}(q0, q1) at ns=50 (101 qubits).

Left: heatmap of W over (q1, q0) with the small-volume ED meson dispersion
overlaid.  Right: W(q0) cuts at selected q1 with window-scan systematic bands.

  PYTHONPATH=. .venv/bin/python scripts/plot_w00_vacuum.py [data/w00_vac_ns50.npz]
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice, analysis, spectroscopy

# Okabe-Ito, fixed assignment order (legacy paper style)
OI = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#000000"]
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()

path = sys.argv[1] if len(sys.argv) > 1 else "data/w00_vac_ns50.npz"
d = np.load(path)
ns, vc = int(d["ns"]), int(d["vc"])
lat = Z2Lattice(ns, pbc=True)
times = d["times"]

# ---- analysis chain: connected correlator -> time completion -> windowed FT
c_conn = analysis.subtract(d["corr"], None, d["probe_1pt"], complex(d["insert_1pt"]))
x = analysis.ring_fold((np.arange(ns) - vc) / 2, lat.nx)
grid = analysis.CorrelatorGrid(times, x, c_conn)
t_full, c_full = analysis.complete_time(grid)

q0 = np.arange(-0.5, 5.501, 0.04)
ks = np.arange(0, lat.nx // 2 + 1)
q1 = 2 * np.pi * ks / lat.nx
sigma_t, sigma_x = times[-1] / 3.0, lat.nx / 6.0
W, spread = analysis.window_scan(t_full, x, c_full, q0, q1, sigma_t, sigma_x)
Wr = W.real

# ---- small-volume ED dispersion for overlay (volume-converged by ns=8)
band = spectroscopy.meson_band(Z2Lattice(8, pbc=True), float(d["m0"]),
                               float(d["g2"]), float(d["eta"]))
k_ed = np.abs(band["k"])
e_ed = band["energy"]

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(9.2, 3.8), constrained_layout=True,
    gridspec_kw={"width_ratios": [1.05, 1]})

# ---- left: heatmap (sequential single hue), dispersion overlay
pm = axL.pcolormesh(q0, q1, Wr.T, cmap="Blues", shading="nearest",
                    vmin=0.0, vmax=np.percentile(Wr, 99.5), rasterized=True)
axL.plot(e_ed, k_ed, "o", ms=7, mfc="none", mew=1.8, color=OI[1],
         label=r"ED meson $E(k)$ ($N_s=8$)")
axL.set_xlabel(r"$q^0$")
axL.set_ylabel(r"$q^1$")
axL.set_title(r"$W^{00}_{\rm vac}(q^0,q^1)$  (MPS, 101 qubits)",
              fontsize=11)
axL.legend(loc="upper left", framealpha=0.9, fontsize=9)
axL.grid(False)
fig.colorbar(pm, ax=axL, pad=0.02, label=r"$W^{00}$")

# ---- right: log-scale cuts; the q1=0 channel is an EXACT null (charge
# conservation, J0(q1=0)=Q, Q|Omega>=0), so its content is a data-driven
# artifact floor (window leakage + ringing) against which real peaks stand.
FLOOR = 1e-5
null = np.abs(Wr[:, 0])
axR.fill_between(q0, FLOOR, np.clip(null, FLOOR, None), color="0.6", alpha=0.35,
                 lw=0, label=r"$q^1{=}0$ null floor")
phys_ks = [len(ks) // 3, 2 * len(ks) // 3, len(ks) - 1]
for i, j in enumerate(phys_ks):
    aug = np.sqrt((spread[:, j] / 2) ** 2 + null ** 2)  # window scan (+) null floor
    axR.fill_between(q0, np.clip(Wr[:, j] - aug, FLOOR, None),
                     np.clip(Wr[:, j] + aug, FLOOR, None), color=OI[i], alpha=0.20, lw=0)
    axR.plot(q0, np.clip(Wr[:, j], FLOOR, None), color=OI[i], lw=1.6,
             label=rf"$q^1={q1[j]:.2f}$")
axR.set_yscale("log")
axR.set_ylim(FLOOR * 0.7, 0.4)
axR.set_xlabel(r"$q^0$")
axR.set_ylabel(r"$W^{00}_{\rm vac}(q^0, q^1)$")
axR.set_title(r"cuts (log); band = window scan $\oplus$ null floor", fontsize=10)
axR.legend(fontsize=8.5, loc="upper left")

out = path.replace(".npz", ".pdf")
fig.savefig(out, dpi=200)
np.savez(path.replace(".npz", "_W.npz"), q0=q0, q1=q1, W=W, spread=spread,
         sigma_t=sigma_t, sigma_x=sigma_x)
print("wrote", out)
