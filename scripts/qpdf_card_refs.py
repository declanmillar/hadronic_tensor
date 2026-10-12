"""Classical references for the prep-only quasi-PDF width scan (campaign preset
`qpdf-scan`): equal-time Wilson-line bilinears on every width card, the
connected same-sublattice h(m), the quasi-distribution and <x> in the
MEASURED convention of the paper (scripts/quasipdf_analysis.py: A(z) =
(C_R - i C_I)/2 connected and vacuum-subtracted, even z = 2m, taste phase
(-1)^m, Gaussian window sigma_m = 5, FT against P = k0 per spatial site,
normalize-then-moment on x in [-0.5, 1.5], h(0) = connected <J0(centre)>),
and the sigma_k^2 extrapolation of <x> (the plane-wave intercept).

  PYTHONPATH=. .venv/bin/python scripts/qpdf_card_refs.py
  -> qpdf_card_refs.npz and qpdf_width_scan.pdf in a new directory under
     data/mps/runs/qpdf/<UTC timestamp>/ (or --out-dir).
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_common import build_prep, stored_prep, log  # noqa: E402
from paper_style import OI  # noqa: E402
from htensor import Z2Lattice, backends, quasipdf  # noqa: E402
from htensor import currents as cur  # noqa: E402
import argparse  # noqa: E402
from run_outputs import check_output_prefix, output_directory  # noqa: E402

CARDS = [  # tag, couplings, vac file, params file
    ("prod", (0.7, 1.1, 1.3), "data/vac_prod.npz", "data/wp10reg_params_k1.26_L3.npz"),
    ("prod", (0.7, 1.1, 1.3), "data/vac_prod.npz", "data/wp_prod_k+1.26_s1.00_L3_v2.npz"),
    ("prod", (0.7, 1.1, 1.3), "data/vac_prod.npz", "data/wp_prod_k+1.26_s1.50_L4_v2.npz"),
    ("relA", (0.4, 1.4, 2.3), "data/vac_relA.npz", "data/wp_relA_k+1.26_s0.75_L3.npz"),
    ("relA", (0.4, 1.4, 2.3), "data/vac_relA.npz", "data/wp_relA_k+1.26_s1.00_L3_v2.npz"),
    ("relA", (0.4, 1.4, 2.3), "data/vac_relA.npz", "data/wp_relA_k+1.26_s1.50_L4_v2.npz"),
]
NS, CENTER = 50, 24
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("mmax", nargs="?", type=int, default=None)
p.add_argument("sig_m", nargs="?", type=float, default=5.0)
p.add_argument("--out-dir", help="output directory (default: a new dated MPS run)")
args = p.parse_args()
MMAX = args.mmax if args.mmax is not None else 5
SIG_M = args.sig_m
MS = np.arange(1, MMAX + 1)     # spatial separations m -> z = +-2m (seam-free for m <= 10 around site 24)
XS = np.linspace(-0.5, 1.5, 401)
TAG = f"_m{MMAX}_s{SIG_M:g}" if args.mmax is not None else ""
out_dir = output_directory("qpdf", args.out_dir, kind="mps")
for name in ("qpdf_card_refs", "qpdf_width_scan"):
    check_output_prefix(out_dir / (name + TAG))
print(f"output directory: {out_dir}", flush=True)


def with_anc(op, ap="I"):
    return SparsePauliOp([ap + l for l in op.paulis.to_labels()], op.coeffs)


def expectations(lat, mps, perm, ops, threads=2):
    n_tot = lat.n_qubits + 1
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    for k, op in ops.items():
        qc.save_expectation_value(backends.permute_pauli(with_anc(op), perm, n_tot), list(range(n_tot)), label=k)
    sim = AerSimulator(method="matrix_product_state", matrix_product_state_truncation_threshold=1e-10,
                       max_parallel_threads=threads)
    d = sim.run(qc).result().data()
    return {k: float(np.real(d[k])) for k in ops}


def make_args(tag, coup, vac, params, state):
    p = argparse.ArgumentParser()
    from prep_common import add_state_args, resolve
    add_state_args(p)
    argv = ["--ns", str(NS), "--center", str(CENTER), "--m0", str(coup[0]), "--g2", str(coup[1]),
            "--eta", str(coup[2]), "--vac", vac, "--tag", tag, "--threads", "2"]
    argv += ["--params", params] if state == "packet" else ["--state", "vac"]
    return resolve(p.parse_args(argv))


lat = Z2Lattice(NS, pbc=True)
ops = {}
for m in MS:
    for z in (2 * m, -2 * m):
        o_r, o_i = quasipdf.wilson_bilinear(lat, CENTER, z)
        ops[f"R_{z:+d}"], ops[f"I_{z:+d}"] = o_r, o_i
ops["J0c"] = cur.charge_density(lat, CENTER)

vac_cache = {}
rows = []
for tag, coup, vac, params in CARDS:
    if tag not in vac_cache:
        a = make_args(tag, coup, vac, None, "vac")
        l_, prep, info = build_prep(a)
        mps, perm, H = stored_prep(a, l_, prep, lat.site_qubit(CENTER), info)
        vac_cache[tag] = expectations(lat, mps, perm, ops)
        log(f"{tag} vacuum <H> = {H:.6f}")
    a = make_args(tag, coup, vac, params, "packet")
    l_, prep, info = build_prep(a)
    mps, perm, H = stored_prep(a, l_, prep, lat.site_qubit(CENTER), info)
    e = expectations(lat, mps, perm, ops)
    v = vac_cache[tag]
    k0, sig = info["k0"], info["sigma"]
    # connected same-sublattice h(m), m = -5..5 (z = 2m), h(0) = connected <J0(centre)>
    ms = np.concatenate([-MS[::-1], [0], MS])
    h = np.array([(0.5 * ((e[f"R_{2*m:+d}"] - v[f"R_{2*m:+d}"]) - 1j * (e[f"I_{2*m:+d}"] - v[f"I_{2*m:+d}"]))
                   if m != 0 else (e["J0c"] - v["J0c"]) + 0j) for m in ms])
    h = h * (-1.0) ** ms
    w = np.exp(-ms ** 2 / (2 * SIG_M ** 2))
    qt = np.array([(k0 / (2 * np.pi)) * np.sum(np.exp(1j * x * k0 * ms) * h * w) for x in XS]).real
    norm = np.trapezoid(qt, XS)
    xmean = np.trapezoid(XS * qt, XS) / norm
    sig_k = 1.0 / (2.0 * sig)
    rows.append(dict(tag=tag, sigma_x=sig, sigma_k=sig_k, k0=k0, H=H, h=h, ms=ms, qt=qt, x=xmean,
                     norm=norm, params=params))
    log(f"{tag} sigma_x={sig:.2f}: <H>={H:.5f}  h(0)={h[len(MS)].real:+.4f}  <x>={xmean:.4f}  (norm {norm:.4f})")

# ---- sigma_k^2 extrapolation per coupling
fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
fits = {}
for i, tag in enumerate(("prod", "relA")):
    r = [q for q in rows if q["tag"] == tag]
    s2 = np.array([q["sigma_k"] ** 2 for q in r]); xm = np.array([q["x"] for q in r])
    c = np.polyfit(s2, xm, 1)
    fits[tag] = c
    ax.plot(s2, xm, "o", color=OI[i], label=f"{tag}: intercept {c[1]:.3f}")
    ss = np.linspace(0, s2.max() * 1.05, 50)
    ax.plot(ss, np.polyval(c, ss), "-", color=OI[i], lw=1.2)
    print(f"{tag}: <x> = {c[1]:.4f} + {c[0]:.4f} sigma_k^2  (points {np.round(xm, 4).tolist()} at sigma_k^2 {np.round(s2, 4).tolist()})")
ax.set_xlabel(r"$\sigma_k^2$")
ax.set_ylabel(r"$\langle x\rangle$ (measured convention)")
ax.set_title("quasi-PDF width scan, classical references (Ns=50 preps)", fontsize=10)
ax.legend(fontsize=8.5)
fig.savefig(out_dir / f"qpdf_width_scan{TAG}.pdf", dpi=200)
np.savez(out_dir / f"qpdf_card_refs{TAG}.npz", xs=XS, ms=rows[0]["ms"],
         **{f"{q['tag']}_s{q['sigma_x']:.2f}_{k}": np.asarray(q[k]) for q in rows for k in ("h", "qt", "x", "norm", "H", "k0", "sigma_k")},
         **{f"fit_{t}": c for t, c in fits.items()})
print(f"wrote {out_dir / f'qpdf_card_refs{TAG}.npz'} {out_dir / f'qpdf_width_scan{TAG}.pdf'}")
