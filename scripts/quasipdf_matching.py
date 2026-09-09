"""Quasi-PDF: exact vs 101-qubit, and boost evolution (task 9).

The equal-time Wilson-line quasi-PDF q~(x;P) of the meson is computed two
ways at the same physical boost P = 2pi/5:
  - exactly on the band-1 momentum eigenstate |P> (gauge-fixed ED) at
    nx = 5 (ns=10) and nx = 10 (ns=20), and
  - as measured on the 101-qubit (nx=25) boosted wavepacket
    (data/quasipdf_x.npz).
CONVENTION (2026-09-03, two fixes):
 (a) CONNECTED -- the vacuum expectation of the same operator is subtracted,
     as the measured packet analysis does.  This is the volume-independent
     choice; the raw moment runs 0.34 -> 0.43 between nx = 5 and 10 because
     the vacuum condensate dominates A(m=1).
 (b) NORMALIZATION -- A(z) = <O_R>/2 for z != 0 and A(0) = <n> (no factor of
     two), matching scripts/quasipdf_analysis.py.  This file previously used
     <O_R> against <n>, double-weighting every z != 0 term.
With both fixes the eigenstate moments are 0.3256 (nx=5) and 0.3249 (nx=10)
for the production couplings -- volume-independent to 0.2% -- and 0.268 /
0.264 at relA.  The measured 101-qubit packet value (0.389 at sigma_x = 0.75)
is biased high by its momentum spread, and the width scan extrapolates to
0.332 (production) and 0.282 (relA) as sigma_k^2 -> 0, i.e. onto the
eigenstate values to 2% and 7%.
CORRECTION 2026-09-03: before the band-dict fix in
scattering.gauge_fixed_system (the highest level in the single-meson window
used to overwrite the lowest at each momentum) this script evaluated the
"exact" curves on a ~5.0-energy state and reported <x> = 0.61 / 0.85 and a
boost evolution of 0.63-0.70; those numbers were wrong and must not be
quoted.  Boost evolution at nx=6 now: <x> = 0.36, 0.42, 0.19 at P = 1.05,
2.09, 3.14.  Quantitative light-cone (LaMET) matching is noted to require
boosts beyond M/P ~ 2 reached here.

  PYTHONPATH=. .venv/bin/python scripts/quasipdf_matching.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice
from htensor import scattering as sc
from htensor.quasipdf import wilson_bilinear
from htensor.currents import charge_density

M0, G2, ETA = 0.7, 1.1, 1.3
XS = np.linspace(-0.5, 1.5, 201)


def exact_quasipdf(ns, P, connected=True):
    """Eigenstate quasi-distribution at momentum P.

    connected=True subtracts the vacuum expectation of the SAME operator, so
    the observable matches the measured packet analysis (which uses the
    connected, vacuum-subtracted bilinear).  This matters: the connected
    moment is volume-independent (0.2508 at nx=5 vs 0.2499 at nx=10) whereas
    the raw one is not (0.337 vs 0.426), because the vacuum condensate
    dominates h(m=1) and cancels only in the difference.
    """
    lat = Z2Lattice(ns, pbc=True)
    nx = lat.nx
    gf = sc.gauge_fixed_system(lat, M0, G2, ETA, n_band=50)
    _, st = gf["band"][round(P, 4)]
    st = st / np.linalg.norm(st)
    vac = gf["vac"] / np.linalg.norm(gf["vac"])
    zmax = nx // 2
    h = np.empty(zmax + 1, dtype=complex)
    for z in range(zmax + 1):
        if z == 0:
            acc = sum(charge_density(lat, 2 * x0) for x0 in range(nx)) / nx
        else:
            acc = sum(wilson_bilinear(lat, 2 * x0, 2 * z)[0]
                      for x0 in range(nx - z)) / (nx - z)
        O = gf["basis"].matrix(acc, sub=gf["sel"])
        # A(z) = <chi^dag(v+z) W chi(v)>: the Pauli operator O_R equals
        # chi^dag_a W chi_b + h.c., so Re A = <O_R>/2 for z != 0, while at
        # z = 0 the bilinear IS the number operator (no factor of two).
        # This matches scripts/quasipdf_analysis.py, the measured convention.
        fac = 1.0 if z == 0 else 0.5
        h[z] = fac * complex(st.conj() @ (O @ st))
        if connected:
            h[z] -= fac * complex(vac.conj() @ (O @ vac))
    zs = np.arange(-zmax, zmax + 1)
    hz = np.array([h[abs(z)] if z >= 0 else np.conj(h[abs(z)]) for z in zs])
    hz = hz * (-1.0) ** np.abs(zs)
    qt = np.array([np.sum(np.exp(1j * x * P * zs) * hz) for x in XS]).real
    return qt, gf["M"]


P0 = 2 * np.pi / 5
print(f"volume independence at P = 2pi/5 = {P0:.4f}:")
curves = {}
for ns in (10, 20):
    for conn in (True, False):
        qt, M = exact_quasipdf(ns, P0, connected=conn)
        qt = qt / np.trapezoid(qt, XS)                # normalize shape
        mx = np.trapezoid(XS * qt, XS)
        tag = "connected (matches the measurement)" if conn else "raw       (vacuum not subtracted)"
        print(f"  ns={ns} (nx={ns//2}) {tag}: <x> = {mx:.4f}")
        if conn:
            curves[ns] = qt

d = np.load("data/quasipdf_x.npz")
q101 = np.interp(XS, d["xs"], d["qt"].real)
q101 = q101 / np.trapezoid(q101, XS)
mx101 = np.trapezoid(XS * q101, XS)
print(f"  101q (nx=25): <x> = {mx101:.4f}")
# cross-volume shape deviation
for ns in (10, 20):
    dev = np.sqrt(np.mean((curves[ns] - q101) ** 2)) / np.abs(q101).max()
    print(f"  RMS(exact ns={ns} - 101q)/peak = {dev:.3f}")

# boost evolution at ns=12
print("\nboost evolution (ns=12):")
lat = Z2Lattice(12, pbc=True)
gf = sc.gauge_fixed_system(lat, M0, G2, ETA, n_band=50)
vac12 = gf["vac"] / np.linalg.norm(gf["vac"])
evo = {}
for P in sorted(k for k in gf["band"] if k > 1e-6):
    _, st = gf["band"][round(P, 4)]; st = st / np.linalg.norm(st)
    zmax = lat.nx // 2
    h = np.empty(zmax + 1, dtype=complex)
    for z in range(zmax + 1):
        acc = (sum(charge_density(lat, 2 * x0) for x0 in range(lat.nx)) / lat.nx
               if z == 0 else
               sum(wilson_bilinear(lat, 2 * x0, 2 * z)[0]
                   for x0 in range(lat.nx - z)) / (lat.nx - z))
        O = gf["basis"].matrix(acc, sub=gf["sel"])          # connected, as above
        fac = 1.0 if z == 0 else 0.5                        # A(z) = <O_R>/2, A(0) = <n>
        h[z] = fac * (complex(st.conj() @ (O @ st)) - complex(vac12.conj() @ (O @ vac12)))
    zs = np.arange(-zmax, zmax + 1)
    hz = np.array([h[abs(z)] if z >= 0 else np.conj(h[abs(z)]) for z in zs])
    hz = hz * (-1.0) ** np.abs(zs)
    qt = np.array([np.sum(np.exp(1j * x * P * zs) * hz) for x in XS]).real
    qt = qt / np.trapezoid(qt, XS)
    evo[P] = qt
    print(f"  P={P:.3f}: <x> = {np.trapezoid(XS*qt, XS):.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
a1.plot(XS, curves[10], "C0--", label="exact $n_x=5$")
a1.plot(XS, curves[20], "C1-.", label="exact $n_x=10$")
a1.plot(XS, q101, "k-", lw=2, label="101 qubits ($n_x=25$)")
a1.set_title(r"volume independence, $P=2\pi/5$", fontsize=10)
a1.set_xlabel("$x$"); a1.set_ylabel(r"$\tilde q(x)$ (normalized)"); a1.legend(fontsize=8)
for P, qt in evo.items():
    a2.plot(XS, qt, label=rf"$P={P:.2f}$")
a2.set_title("boost evolution ($n_x=6$)", fontsize=10)
a2.set_xlabel("$x$"); a2.legend(fontsize=8)
fig.savefig("data/quasipdf_matching.pdf", dpi=200)
np.savez("data/quasipdf_matching.npz", xs=XS, exact10=curves[10],
         exact20=curves[20], q101=q101, mx101=mx101)
print("\nwrote data/quasipdf_matching.pdf")
