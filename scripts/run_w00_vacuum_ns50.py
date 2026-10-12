"""Production run: vacuum-polarization W^{00}(q0, q1) at ns=50 (101 qubits).

The vector current acting on the vacuum creates C-odd Q=0 states -- the
single-meson band -- so W^{00}_vac(q0, q1) peaks along the meson dispersion
q0 = E(q1): a 101-qubit extraction of the meson dispersion, cross-checkable
against small-volume ED.  Serves as the production warm-up for the
wavepacket hadronic tensor.

Run from repo root:
  PYTHONPATH=. .venv/bin/python scripts/run_w00_vacuum_ns50.py

New output defaults to data/mps/runs/vacuum/<UTC timestamp>/.
An optional positional output file must not already exist.
"""

import sys
import time

import numpy as np

from htensor import Z2Lattice, stateprep, backends
from htensor import currents as cur
from run_outputs import output_directory, output_file

M0, G2, ETA = 0.7, 1.1, 1.3
NS = 50
TIMES = np.arange(0.0, 6.01, 0.5)
DT_TARGET = 0.1
TRUNC = 1e-8

out_path = output_file(sys.argv[1] if len(sys.argv) > 1 else
                       output_directory("vacuum", kind="mps") / "w00_vac_ns50.npz")
print(f"output: {out_path}", flush=True)

t0 = time.time()
lat6 = Z2Lattice(6, pbc=True)
thetas = stateprep.optimize_vacuum(lat6, M0, G2, ETA, n_layers=2,
                                   restarts=2)["thetas"]
print(f"[{time.time()-t0:6.0f}s] vacuum angles from ns=6 optimization", flush=True)

lat = Z2Lattice(NS, pbc=True)
prep = stateprep.vacuum_ansatz(lat, thetas)
vc = NS // 2  # even (positron) site at mid-ring
probes = [cur.charge_density(lat, v) for v in range(NS)]
insert = cur.charge_density(lat, vc)

data = None
rows = []
for t in TIMES:  # one call per time slice -> incremental progress logging
    d = backends.hadamard_correlator_aer(
        lat, prep, insert, probes, M0, G2, ETA, [t], dt_target=DT_TARGET,
        method="matrix_product_state", mps_trunc=TRUNC,
        stationary_1pt=False if t == 0.0 else True)
    rows.append(d)
    print(f"[{time.time()-t0:6.0f}s] t={t:4.1f} done  "
          f"C(t, x=0) = {d.correlator[0, vc]:.6f}", flush=True)

corr = np.concatenate([d.correlator for d in rows], axis=0)
probe_1pt = np.concatenate([d.probe_expect for d in rows], axis=0)
np.savez(out_path, times=TIMES, corr=corr, probe_1pt=probe_1pt,
         insert_1pt=rows[0].insert_expect, ns=NS, vc=vc,
         m0=M0, g2=G2, eta=ETA, dt_target=DT_TARGET, trunc=TRUNC,
         thetas=thetas)
print(f"[{time.time()-t0:6.0f}s] saved {out_path}", flush=True)
