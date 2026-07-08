"""Hardware-certificate figure from the tier-1 IBM Heron runs (task: real-
hardware plot).  Gauss-law witness map across the 50 sites for both devices,
plus the global certificates (<H>, parity, charge) against ideal.

  PYTHONPATH=. .venv/bin/python scripts/hw_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEV = [("data/hw/job_tier1_d9610mqf47jc73a51pr0.npz", "ibm\\_marrakesh", "C0"),
       ("data/hw/job_tier1_d96f12l2su3c739gsg0g.npz", "ibm\\_fez", "C1")]
H_TRUTH = -49.354            # MPS-exact packet energy

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [2.3, 1]},
                             constrained_layout=True)
for path, name, col in DEV:
    d = np.load(path)
    g = np.array([float(d[f"G{n}"]) for n in range(50)])
    ge = np.array([float(d[f"G{n}_err"]) for n in range(50)])   # EstimatorV2 stds
    a1.errorbar(range(50), g, yerr=ge, fmt="o-", color=col, ms=3, lw=0.8,
                elinewidth=0.6, capsize=1,
                label=f"{name} (mean {g.mean():.2f})")
# ibm_kingston: witnesses reconstructed from the boosted-packet S(q^1) run's
# own bitstrings (sites in Z, links in X natively give the Gauss stabilizers);
# each G_n is a mean of +-1 outcomes -> binomial shot error sqrt((1-G^2)/N).
hwk = np.load("data/hwsq_HARDWARE_k1.26.npz")
gk, Nk = hwk["G"], int(hwk["nshot"])
gke = np.sqrt(np.clip(1 - gk ** 2, 0, 1) / Nk)
a1.errorbar(range(50), gk, yerr=gke, fmt="s-", color="C2", ms=3, lw=0.8,
            elinewidth=0.6, capsize=1,
            label=f"ibm\\_kingston$^\\dagger$ (mean {gk.mean():.2f})")
a1.axhline(1.0, color="0.4", lw=0.8, ls="--")
a1.axhline(0.0, color="0.7", lw=0.6)
a1.set_xlabel("site $n$")
a1.set_ylabel(r"Gauss witness $\langle G_n\rangle$")
a1.set_ylim(-0.3, 1.1)
a1.legend(fontsize=7.5, loc="upper right")
a1.set_title("(a) gauge-law stabilizers, 101 qubits", fontsize=9)

# global certificates, normalized to ideal
labels = [r"$\langle G\rangle$", r"$\langle H\rangle$", r"$|P_f|$"]
x = np.arange(len(labels))
w = 0.35
for i, (path, name, col) in enumerate(DEV):
    d = np.load(path)
    gn = np.array([float(d[f"G{n}"]) for n in range(50)])
    gne = np.array([float(d[f"G{n}_err"]) for n in range(50)])
    vals = [gn.mean(), float(d["H"]) / H_TRUTH, abs(float(d["Pf"]))]
    verr = [np.sqrt(np.sum(gne ** 2)) / 50, float(d["H_err"]) / abs(H_TRUTH),
            float(d["Pf_err"])]
    a2.bar(x + (i - 0.5) * w, vals, w, color=col, label=name,
           yerr=verr, capsize=2, error_kw=dict(elinewidth=0.8))
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
