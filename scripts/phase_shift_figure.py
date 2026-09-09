"""delta(p) collapse figure across seven volumes (replaces the phase-shift
prose table).  PYTHONPATH=. python scripts/phase_shift_figure.py"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI, MARKERS

d = np.load("data/phase_shifts_6vol.npz")
rows = d["rows"]                     # ns, E2, p, n, delta
M = float(d["M"])
cmap = {ns: (OI[i], MARKERS[i]) for i, ns in enumerate(range(8, 21, 2))}
fig, ax = plt.subplots(figsize=(3.3, 2.7), constrained_layout=True)
for ns in sorted(set(int(r[0]) for r in rows)):
    m = rows[:, 0] == ns
    c, mk = cmap[ns]
    ax.plot(rows[m, 2], rows[m, 4], mk, color=c, ms=6,
            label=f"$N_x={ns//2}$")
# 1D low-energy theorem: delta(0) = pi/2 generically; odd expansion
# delta = pi/2 - a p + r p^3 fit to all points (the 1D effective range)
X = np.vstack([rows[:, 2], rows[:, 2] ** 3]).T
coef, *_ = np.linalg.lstsq(X, np.pi / 2 - rows[:, 4], rcond=None)
resid = (np.pi / 2 - X @ coef) - rows[:, 4]
sig_a = float(np.sqrt(np.mean(resid ** 2)
                      * np.linalg.inv(X.T @ X)[0, 0]))
pf = np.linspace(0, rows[:, 2].max() * 1.03, 200)
fit = np.pi / 2 - coef[0] * pf - coef[1] * pf ** 3
cov = np.mean(resid ** 2) * np.linalg.inv(X.T @ X)
J = np.vstack([pf, pf ** 3])
band = np.sqrt(np.einsum("ip,ij,jp->p", J, cov, J))
ax.fill_between(pf, fit - band, fit + band, color="0.35", alpha=0.18,
                lw=0, zorder=0)
ax.plot(pf, fit, "-", color="0.35", lw=1.1, zorder=1)
print(f"ERE fit: a = {coef[0]:.3f} +- {sig_a:.3f}, "
      f"r = {-coef[1]:.4f}, rms {np.sqrt(np.mean(resid**2)):.3f}")
# delta = 0 transparency point: per-volume crossings in E, drawn via
# the same (p, E) relation (fig:cgkel conventions)
cE, cP = [], []
for ns in sorted(set(rows[:, 0].astype(int))):
    sv = rows[rows[:, 0] == ns]
    sv = sv[np.argsort(sv[:, 2])]
    for i in range(len(sv) - 1):
        if sv[i, 4] * sv[i + 1, 4] < 0:
            t0 = -sv[i, 4] / (sv[i + 1, 4] - sv[i, 4])
            cP.append(sv[i, 2] + t0 * (sv[i + 1, 2] - sv[i, 2]))
            cE.append(sv[i, 1] + t0 * (sv[i + 1, 1] - sv[i, 1]))
srt = rows[np.argsort(rows[:, 1])]
E2p0 = lambda e: np.interp(e, srt[:, 1], srt[:, 2])
Ec, Ece = float(np.mean(cE)), float(np.std(cE))
ax.axvspan(E2p0(Ec - Ece), E2p0(Ec + Ece), color="#0072B2", alpha=0.20,
           lw=0)
ax.axvline(E2p0(Ec), color="#0072B2", lw=0.9)
ax.text(E2p0(Ec) + 0.09, 0.30, rf"$E_{{\delta=0}} = {Ec:.3f}({round(Ece*1000)})$",
        fontsize=7, color="#0072B2", ha="left")
print(f"transparency: E0 = {Ec:.3f} +- {Ece:.3f}, p0 = {E2p0(Ec):.3f}")
ax.axhline(0, color="0.7", lw=0.6)
ax.axhline(np.pi / 2, color="0.7", lw=0.6, ls=":")
ax.text(0.05, np.pi / 2 + 0.05, r"$\pi/2$", fontsize=9, color="0.4")
ax.set_ylim(-0.75, 1.80)
srtE = rows[np.argsort(rows[:, 2])]
p2E = lambda x: np.interp(x, srtE[:, 2], srtE[:, 1])
E2pf = lambda x: np.interp(x, np.sort(srtE[:, 1]),
                           srtE[np.argsort(srtE[:, 1]), 2])
sx = ax.secondary_xaxis("top", functions=(p2E, E2pf))
sx.set_xlabel(r"$E$ (two-meson energy)", fontsize=8)
sx.set_xticks([5.5, 5.6, 5.7, 5.8, 5.9])
ax.set_xlabel("relative momentum $p$")
ax.set_ylabel(r"elastic phase shift $\delta(p)$")
ax.legend(fontsize=6.0, ncol=4, loc="lower left",
          bbox_to_anchor=(0.01, 0.015), framealpha=0.9,
          columnspacing=0.55, handletextpad=0.25,
          labelspacing=0.45, borderpad=0.3, handlelength=1.1)
fig.savefig("data/phase_shift_collapse.pdf", dpi=200)
print("wrote data/phase_shift_collapse.pdf  (%d points, %d volumes)"
      % (len(rows), len(set(rows[:, 0]))))
