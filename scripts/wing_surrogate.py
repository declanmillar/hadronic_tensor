"""Wing-anchor target from a small-lattice vacuum: the staggered Trotter
breathing, computed where it is cheap.

The wing anchor subtracts, per parity, the mean of (b_measured - b_ideal) over
spectator sites, so it needs the IDEAL one-point value <J0(v,t)> of the
dt = 0.5 hardware circuit at every campaign time.  At the relA couplings those
grids cost weeks (the MPS cost doubles every dt ~ 0.5 in t), and they stop at
t = 3.

Three measurements (2026-09-07, production point, Ns = 50) say we do not need
them:

  1. Wing sites are NOT static: they oscillate by +-0.042 in a pure staggered
     pattern with period 2 in t -- far above the 0.007-0.010 statistical error,
     so the t = 0 row alone is not a valid target.
  2. The oscillation is a dt = 0.5 TROTTER ARTEFACT: the dt_target = 0.1 MPS
     correlator grids do not contain it, and substituting them mis-anchors by
     up to 0.04.
  3. It is VACUUM physics: the packet's wing values track the pure-vacuum grid
     at a constant ratio (0.0220/0.0274, 0.0417/0.0520, 0.0197/0.0249, ... =
     0.80 at every time sampled).  The packet only rescales it.

So compute the breathing on a small ring, where a statevector is exact and
cheap, and use it as the wing target.  The amplitude is a bulk vacuum property;
`validate` checks that claim against the Ns = 50 grids at the production point,
where both exist, before it is trusted at relA.

  # validate the surrogate where the truth is known
  PYTHONPATH=. .venv/bin/python scripts/wing_surrogate.py validate
  # build the target used by the analysis
  PYTHONPATH=. .venv/bin/python scripts/wing_surrogate.py build --tag relA \
      --vac data/vac_relA.npz --m0 0.4 --g2 1.4 --eta 2.3
"""

import argparse
import sys

import numpy as np
from qiskit.quantum_info import Statevector

from htensor import Z2Lattice, stateprep, trotter
from htensor import currents as cur

DT = 0.5
TIMES = [round(0.5 * i, 2) for i in range(0, 17)]      # 0 .. 8


def breathing(m0, g2, eta, thetas, link_ref="+", ns=8, times=TIMES, dt=DT):
    """Staggered vacuum breathing under the dt-step hardware Trotterization.

    -> dict t -> (even_shift, odd_shift), the per-parity mean of
    <J0(v,t)> - <J0(v,0)> on a periodic ring of `ns` staggered sites.
    Exact (statevector); ns = 8 is 17 qubits, ns = 12 is 25.
    """
    lat = Z2Lattice(ns, pbc=True)
    prep = stateprep.vacuum_ansatz(lat, thetas, link_ref=link_ref)
    ops = [cur.charge_density(lat, v) for v in range(ns)]
    ev = np.arange(ns) % 2 == 0
    out, base = {}, None
    for t in times:
        qc = prep.copy()
        if t > 0:
            qc.compose(trotter.trotter_circuit(lat, m0, g2, eta, t, int(round(t / dt))), inplace=True)
        sv = Statevector(qc)
        b = np.array([sv.expectation_value(o).real for o in ops])
        if base is None:
            base = b.copy()
        d = b - base
        out[t] = (float(d[ev].mean()), float(d[~ev].mean()))
    return out


def _ns50_wing(path, center=24, wing_min=12):
    """Per-parity wing shift from an Ns=50 ideal grid: the quantity the
    surrogate must reproduce."""
    d = np.load(path, allow_pickle=True)
    ns = len(d["probes"])
    t = np.asarray(d["times"], float)
    b = np.array([[d[f"B_{v}"][i] for v in range(ns)] for i in range(len(t))]).real
    wing = np.abs(np.arange(ns) - center) > wing_min
    ev = np.arange(ns) % 2 == 0
    out = {}
    for i, tt in enumerate(t):
        dv = b[i] - b[0]
        out[round(float(tt), 2)] = (float(dv[wing & ev].mean()), float(dv[wing & ~ev].mean()))
    return out


def validate(args):
    """Compare the surrogate with the Ns=50 production grids, where both exist."""
    z = np.load(args.vac, allow_pickle=True)
    th, lr = z["thetas"], (str(z["link_ref"]) if "link_ref" in z.files else "+")
    ref_pk = _ns50_wing(args.packet_grid)
    # the vacuum card also carries the insertion gadget, so its sites near the
    # centre are mixed between the two ancilla branches: use the SAME wing
    # definition as the packet card rather than averaging over everything
    ref_vc = _ns50_wing(args.vacuum_grid)
    common = sorted(set(ref_pk) & set(ref_vc))
    print(f"surrogate validation at ({args.m0}, {args.g2}, {args.eta}), link_ref {lr}")
    for ns in args.ns:
        sur = breathing(args.m0, args.g2, args.eta, th, lr, ns=ns, times=common)
        e_vc = max(abs(sur[t][0] - ref_vc[t][0]) for t in common)
        # the packet rescales the vacuum breathing at wing sites; fit that one ratio
        num = sum(sur[t][0] * ref_pk[t][0] for t in common)
        den = sum(sur[t][0] ** 2 for t in common)
        r = num / den if den else float("nan")
        e_pk = max(abs(r * sur[t][0] - ref_pk[t][0]) for t in common)
        print(f"  ns={ns:>3}: vs Ns=50 vacuum grid  max|diff| = {e_vc:.5f}"
              f"   |   vs packet wings (x{r:.3f})  max|diff| = {e_pk:.5f}")
    print(f"\ngate: <= {args.tol} (about 25% of the 0.007-0.010 per-site statistical error)")
    print("columns: the vacuum comparison is what the vac-w00 card needs; the packet comparison is")
    print("what relA-core needs, and its ratio is the packet's rescaling of the same vacuum mode.")


def build(args):
    z = np.load(args.vac, allow_pickle=True)
    th, lr = z["thetas"], (str(z["link_ref"]) if "link_ref" in z.files else "+")
    sur = breathing(args.m0, args.g2, args.eta, th, lr, ns=args.ns[0], times=TIMES)
    t = np.array(sorted(sur))
    even = np.array([sur[x][0] for x in t])
    odd = np.array([sur[x][1] for x in t])
    out = args.out or f"data/wing_surrogate_{args.tag}.npz"
    np.savez(out, times=t, even=even, odd=odd, ns=args.ns[0], dt=DT,
             m0=args.m0, g2=args.g2, eta=args.eta, link_ref=lr, vac=args.vac,
             note="per-parity staggered vacuum breathing under the dt=0.5 hardware Trotterization; "
                  "the wing-anchor target for slices whose Ns=50 ideal grid is short. "
                  "Multiply by the packet's wing rescaling (see validate) for a packet card.")
    print(f"wrote {out}: ns={args.ns[0]}, {len(t)} times {t[0]}..{t[-1]}")
    for x in (0.5, 1.0, 2.0, 3.0, 6.0):
        if x in sur:
            print(f"   t={x:>4}: even {sur[x][0]:+.5f}  odd {sur[x][1]:+.5f}")


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
for name in ("validate", "build"):
    s = sub.add_parser(name)
    s.add_argument("--m0", type=float, default=0.7 if name == "validate" else 0.4)
    s.add_argument("--g2", type=float, default=1.1 if name == "validate" else 1.4)
    s.add_argument("--eta", type=float, default=1.3 if name == "validate" else 2.3)
    s.add_argument("--vac", default="data/vac_prod.npz" if name == "validate" else "data/vac_relA.npz")
    s.add_argument("--ns", type=int, nargs="+", default=[6, 8, 10] if name == "validate" else [8])
    s.add_argument("--tag", default="prod" if name == "validate" else "relA")
    if name == "validate":
        s.add_argument("--packet-grid", default="data/hw_cal_prod_ns50_j0_k1.26_s0.75_t8.npz")
        s.add_argument("--vacuum-grid", default="data/hw_cal_prod_ns50_j0_vac.npz")
        s.add_argument("--tol", type=float, default=0.002)
    else:
        s.add_argument("--out", default=None)
args = p.parse_args()
(validate if args.cmd == "validate" else build)(args)
