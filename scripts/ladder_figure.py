"""Synthesis-error ladder at 101 qubits: ridge deviation AND <H> error vs
rotation tolerance, stochastic (seed-averaged) vs deterministic rounding,
spanning the exact baseline to the Clifford endpoint.  Two panels, matched
to the ns=8 pilot (fig:synth).

  PYTHONPATH=. python scripts/ladder_figure.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
import numpy as np

from htensor import Z2Lattice, analysis
from scripts_helpers_ridge import onesided_ft

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6})
DET, STOCH = "#0072B2", "#D55E00"          # Okabe-Ito blue / orange

lat = Z2Lattice(50, pbc=True)
x = analysis.ring_fold((np.arange(50) - 24) / 2, lat.nx)
Q0 = np.arange(-0.6, 1.601, 0.04)


def ridge(f, W0, peak):
    d = np.load(f)
    G = d["corr"] - d["one_pt"] * complex(d["insert_1pt"])
    W = onesided_ft(d["times"], x, G, Q0, np.array([np.pi / 2]),
                    8 / 3, lat.nx / 6, 0.5, 0.5)[:, 0]
    return np.sqrt(np.mean(np.abs(W - W0) ** 2)) / peak


def herr(f, H0):
    return abs(complex(np.load(f)["H"]).real - H0)


d0 = np.load("data/synthladder_exact.npz")
G0 = d0["corr"] - d0["one_pt"] * complex(d0["insert_1pt"])
W0 = onesided_ft(d0["times"], x, G0, Q0, np.array([np.pi / 2]),
                 8 / 3, lat.nx / 6, 0.5, 0.5)[:, 0]
peak = np.abs(W0).max()
H0 = complex(d0["H"]).real

denoms = [128, 64, 32, 16]
delta = np.pi / np.array(denoms)


def collect(metric):
    sm, ss, rn = [], [], []
    for d in denoms:
        e = [metric(f) for f in glob.glob(f"data/synthladder_stoc_pi{d}*.npz")]
        sm.append(np.mean(e)); ss.append(np.std(e) if len(e) > 1 else 0)
        rf = f"data/synthladder_roun_pi{d}_s0.npz"
        rn.append(metric(rf) if os.path.exists(rf) else np.nan)
    return np.array(sm), np.array(ss), np.array(rn)


r_sm, r_ss, r_rn = collect(lambda f: ridge(f, W0, peak))
h_sm, h_ss, h_rn = collect(lambda f: herr(f, H0))
cliff_r = ridge("data/synthladder_clifford.npz", W0, peak)
cliff_h = herr("data/synthladder_clifford.npz", H0)


def style(ax, ylab):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"synthesis grid spacing $\delta$ (rad)")
    ax.set_ylabel(ylab)
    ax.set_xlim(0.021, 0.24)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    secax = ax.secondary_xaxis("top", functions=(
        lambda z: 3 * np.log2(2 / np.clip(z, 1e-9, None)),
        lambda t: 2 / 2 ** (t / 3)))
    secax.set_xlabel(r"$\sim T$ gates per rotation (Ross--Selinger)")
    secax.set_xticks([10, 12, 14, 16, 18])
    secax.xaxis.set_major_formatter(FuncFormatter(lambda t, _: f"{t:.0f}"))
    secax.xaxis.set_minor_locator(NullLocator())


fig, (aL, aR) = plt.subplots(1, 2, figsize=(6.6, 2.9), constrained_layout=True)
for ax, sm, ss, rn, cl, ylab in (
        (aL, r_sm, r_ss, r_rn, cliff_r, r"ridge RMS error / peak"),
        (aR, h_sm, h_ss, h_rn, cliff_h, r"$|\Delta\langle H\rangle|$ error")):
    ax.plot(delta, rn, "o-", color=DET, label="deterministic")
    ax.errorbar(delta, sm, yerr=ss, fmt="s-", color=STOCH, capsize=3,
                label="stochastic (seeds)")
    ax.axhline(cl, color="0.5", ls=":", lw=1)
    ax.text(delta[-1], cl * 1.04, "Clifford", fontsize=7, color="0.4", ha="right")
    style(ax, ylab)
aL.set_ylim(0.009, 0.8)
aL.legend(fontsize=7.5, loc="upper left")
fig.suptitle("Synthesis tolerance, 101 qubits", fontsize=9.5)
fig.savefig("data/ladder_figure.pdf", dpi=200)
print("ridge  stoch:", np.round(r_sm, 4), " det:", np.round(r_rn, 4))
print("|dH|   stoch:", np.round(h_sm, 4), " det:", np.round(h_rn, 4))
print(f"Clifford: ridge {cliff_r:.3f}, |dH| {cliff_h:.3f}; H0={H0:.4f}")
print("wrote data/ladder_figure.pdf")
