"""Hardware-certificate figure from the tier-1 IBM Heron runs (task: real-
hardware plot).  Gauss-law witness map across the 50 sites for both devices,
plus the global certificates (<H>, parity, charge) against ideal.

  PYTHONPATH=. .venv/bin/python scripts/hw_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()
import numpy as np

DEV = [("data/hw/job_tier1_d9610mqf47jc73a51pr0.npz", dev("marrakesh"), "#0072B2"),
       ("data/hw/job_tier1_d96f12l2su3c739gsg0g.npz", dev("fez"), "#D55E00")]
H_TRUTH = -49.369            # MPS-exact packet energy (routed-chain MPS 2026-09-03; the unrouted pipeline gave -49.354, truncation-limited)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [2.3, 1]},
                             constrained_layout=True)
for path, name, col in DEV:
    d = np.load(path)
    g = np.array([float(d[f"G{n}"]) for n in range(50)])
    ge = np.array([float(d[f"G{n}_err"]) for n in range(50)])   # EstimatorV2 stds
    a1.errorbar(range(50), g, yerr=ge, fmt="o-", color=col, ms=3, lw=0.8,
                elinewidth=0.6, capsize=1, label=name)
    for n in np.flatnonzero(g < 0):       # below-axis sites -> arrows
        a1.annotate("", xy=(n, 0.006), xytext=(n, 0.08),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0),
                    zorder=5)
    a1.axhline(g.mean(), color=col, lw=1.0, alpha=0.7)
    a1.axhspan(g.mean() - g.std(), g.mean() + g.std(), color=col,
               alpha=0.08, lw=0)
# ibm_kingston: BOOSTED-packet witnesses (k1.26; ibm_hardware.K0TAG feeds
# build_prep -> wp10reg_params_k1.26_L3.npz) from the forward-campaign t0.0
# pubs of passes 1 and 2 pooled, 2 x 70k shots (prep + correlator gadget,
# 1064 2q -- depth-comparable to the other two): sites in Z, links in X give
# the Gauss stabilizers per shot.  marrakesh/fez in DEV are the rest-packet
# runs; this row is the only boosted one, as in the v3 dagger note.
hk = np.load("data/hw/losch_harvest_pooled.npz")
gk, Nk = hk["G_t00"], 140_000
gke = np.sqrt(np.clip(1 - gk ** 2, 0, 1) / Nk)
a1.errorbar(range(50), gk, yerr=gke, fmt="s-", color="#009E73", ms=3, lw=0.8,
            elinewidth=0.6, capsize=1,
            label=dev("kingston"))
a1.axhline(gk.mean(), color="#009E73", lw=1.0, alpha=0.7)
a1.axhspan(gk.mean() - gk.std(), gk.mean() + gk.std(), color="#009E73",
           alpha=0.08, lw=0)
a1.axhline(0.0, color="0.7", lw=0.6)
a1.set_xlabel("site $n$")
a1.set_ylabel(r"Gauss witness $\langle G_n\rangle$")
a1.set_ylim(0, 0.67)
a1.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
a1.set_title("(a) gauge-law stabilizers, 101 qubits", fontsize=9)

# global certificates, normalized to ideal
labels = [r"$\langle G\rangle$", r"$\langle H\rangle$", r"$|P_f|$"]
x = np.arange(len(labels))
w = 0.26
for i, (path, name, col) in enumerate(DEV):
    d = np.load(path)
    gn = np.array([float(d[f"G{n}"]) for n in range(50)])
    gne = np.array([float(d[f"G{n}_err"]) for n in range(50)])
    vals = [gn.mean(), float(d["H"]) / H_TRUTH, abs(float(d["Pf"]))]
    verr = [np.sqrt(np.sum(gne ** 2)) / 50, float(d["H_err"]) / abs(H_TRUTH),
            float(d["Pf_err"])]
    a2.bar(x + (i - 1) * w, vals, w, color=col, label=name,
           yerr=verr, capsize=2, error_kw=dict(elinewidth=0.8))
# kingston: witness mean only -- the single-basis bit job measures neither
# <H> (needs hopping bases) nor a comparable |Pf| (50-qubit parity is
# readout-dominated without the tier-1 jobs' TREX mitigation)
a2.bar(0 + w, gk.mean(), w, color="#009E73",
       yerr=np.sqrt(np.sum(gke ** 2)) / 50, capsize=2,
       error_kw=dict(elinewidth=0.8))
a2.axhline(1.0, color="0.4", lw=0.8, ls="--")
a2.set_xticks(x)
a2.set_xticklabels(labels, fontsize=8)
a2.set_ylabel("measured / ideal")
a2.set_ylim(0, 1.1)
a2.set_title("(b) global certificates", fontsize=9)
fig.savefig("data/hw_certificates.pdf", dpi=200)
print("wrote data/hw_certificates.pdf")
for path, name, col in DEV:
    d = np.load(path)
    g = np.array([float(d[f"G{n}"]) for n in range(50)])
    print(f"{name}: Gauss mean {g.mean():.3f}, <H>/truth "
          f"{float(d['H'])/H_TRUTH:.2f}, Q {float(d['Q']):.2f}")
