"""Structure-factor hardware figure with two-parameter noise self-consistency.

(a) S(q^1) on 101 qubits: exact curve, ibm_kingston measurement (mitigated,
    bootstrap error bars), and the noise-model simulation at nominal gate
    error g*=1 -- which reproduces the device S(q) (corr 0.99) with NO tuning
    to it.
(b) The two error sectors separate: the site-only S(q) fixes the gate scale
    g*, while the link-sensitive Gauss witness fixes the link-readout scale
    L*.  Witness vs L at g* (S(q) is flat in L), pinned to the measured G_hw.

  PYTHONPATH=. .venv/bin/python scripts/hw_sq_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K0TAG = "k1.26"
hw = np.load(f"data/hwsq_HARDWARE_{K0TAG}.npz")
tp = np.load(f"data/hwsq_2param_{K0TAG}.npz")
q, Si = hw["q"], hw["S_ideal"]
Smit, Serr = hw["S_mit"], hw["S_err"]
Ghw, Gerr = float(hw["G"].mean()), float(hw["Gerr"])

gstar, Lstar = float(tp["gstar"]), float(tp["Lstar"])
S_sim, Smit_sim = tp["S_at_gstar"], tp["Smit_at_gstar"]
gates, Ls = tp["gates"], tp["Ls"]
gi = int(np.argmin(np.abs(gates - gstar)))
wl = tp["wls"][gi]                     # witness vs L at g*
corr_sim = float(np.corrcoef(S_sim, Si)[0, 1])
corr_hw = float(np.corrcoef(hw["S_raw"], Si)[0, 1])

# shot-bootstrap band on the sim S(q) at g* (site-only estimator, L-free)
bits = np.load(f"data/2p_bits_g{gi}_{K0TAG}.npz")["bits"].astype(np.int8)
lat_ns, nx = 50, 25
sites = np.arange(0, 2 * lat_ns, 2)
xv = (np.arange(lat_ns) // 2).astype(float)
A = np.vstack([Si, np.ones_like(Si)]).T
rng = np.random.default_rng(0)
N = len(bits)
Sb = []
for _ in range(200):
    idx = rng.integers(0, N, N)
    zv = 1 - 2 * bits[idx][:, sites]
    rho = zv @ np.exp(1j * np.outer(q, xv)).T
    S = ((np.abs(rho) ** 2).mean(0) - np.abs(rho.mean(0)) ** 2).real / nx
    (f, c), *_ = np.linalg.lstsq(A, S, rcond=None)
    Sb.append((S - c) / f)
Ssim_err = np.std(Sb, 0)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1]},
                             constrained_layout=True)

# --- panel (a): S(q) ---
a1.plot(q, Si, "k-", lw=1.4, zorder=2, label="exact (ED)")
a1.fill_between(q, Smit_sim - Ssim_err, Smit_sim + Ssim_err,
                color="C0", alpha=0.22, lw=0, zorder=1)
a1.plot(q, Smit_sim, "--", color="C0", lw=1.1, zorder=3,
        label=rf"model ($g^\ast\!=\!1,L^\ast\!=\!{Lstar:.1f}$)")
a1.errorbar(q, Smit, yerr=Serr, fmt="o", color="C3", ms=4.5, lw=0,
            elinewidth=1.1, capsize=2, zorder=4, label="ibm\\_kingston")
a1.set_xlabel(r"$q^1$")
a1.set_ylabel(r"$S(q^1)$")
a1.set_ylim(-0.05, 1.05)
a1.legend(fontsize=7.5, loc="upper left", handlelength=1.6)
a1.set_title("(a) structure factor, 101 qubits", fontsize=9)
rec_hw = int(np.sum(np.abs((Smit - Si) / Si) < 0.15))
a1.text(0.97, 0.05,
        rf"corr $=%.2f$ (dev), $%.2f$ (model)" % (corr_hw, corr_sim),
        transform=a1.transAxes, ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6))

# --- panel (b): the two sectors separate ---
a2.plot(Ls, wl, "-", color="C0", lw=1.3, zorder=3,
        label=r"witness $\langle G\rangle(L)$, $g{=}1$")
a2.axhspan(Ghw - Gerr, Ghw + Gerr, color="C3", alpha=0.18, lw=0)
a2.axhline(Ghw, color="C3", lw=1.0, ls=":")
a2.axvline(Lstar, color="0.5", lw=0.9, ls="--")
a2.plot([Lstar], [Ghw], "*", color="C3", ms=13, zorder=5,
        label="ibm\\_kingston")
# S(q) contrast is flat in L (site-only) -- draw as a reference band
a2.axhline(0.754, color="C2", lw=0.8, ls="-.", alpha=0.6)
a2.text(11.6, 0.775, r"$L{=}1$ (uniform)", color="C2", fontsize=6.5,
        ha="right", va="bottom")
a2.set_xlabel(r"link-readout scale $L$")
a2.set_ylabel(r"Gauss witness $\langle G\rangle$")
a2.set_ylim(0, 0.85)
a2.set_xlim(Ls.min(), Ls.max())
a2.annotate(rf"$L^\ast={Lstar:.1f}$", xy=(Lstar, 0.05),
            xytext=(Lstar + 0.8, 0.12), fontsize=8, color="0.3")
a2.legend(fontsize=7, loc="upper right", handlelength=1.4)
a2.set_title("(b) site/link error sectors separate", fontsize=9)

fig.savefig("data/hwsq_figure.pdf", dpi=200)
med_hw = np.median(np.abs((Smit - Si) / Si)[Si > 0.10])
med_sim = np.median(np.abs((Smit_sim - Si) / Si)[Si > 0.10])
print("wrote data/hwsq_figure.pdf")
print(f"g*={gstar:.1f}, L*={Lstar:.2f}; S(q) corr sim {corr_sim:.3f} vs "
      f"hardware {corr_hw:.3f}")
print(f"median |dS|/S (q>=0.75): model {100*med_sim:.1f}%  hardware "
      f"{100*med_hw:.1f}%; hardware recovered {rec_hw}/12")
