"""J1-insertion production: W^{01}, W^{11} (completes the tensor + FT-side Ward) at ns=50 (101 qubits).

Pipeline: ns=6-optimized vacuum angles + ns=8-trained L=3 wavepacket block
(optimized interpolator target), translated to mid-ring at ns=50; J^0
inserted at the packet center; J^0 (all sites) AND J^1 (all bonds) probes
read from the same circuits.  A matched vacuum grid is measured for the
pointwise vacuum subtraction, and <H> is saved on both preps so the packet
energy E(k0) is certified at the production volume.

  PYTHONPATH=. .venv/bin/python scripts/run_w_j1_ns50.py [k0] [out.npz]

New output defaults to data/mps/runs/meson/<UTC timestamp>/.
An explicit output file must not already exist.
"""

import os
import sys
import time

import numpy as np
from qiskit.quantum_info import Statevector

from htensor import Z2Lattice, stateprep, spectroscopy, wavepacket, \
    block_engine, backends
from htensor import currents as cur
from htensor import hamiltonian as ham
from run_outputs import output_directory, output_file

M0, G2, ETA = 0.7, 1.1, 1.3
NS = 50
CENTER = 24                      # even staggered site, spatial x0 = 12
TIMES = np.arange(0.0, 8.01, 0.5)
DT_TARGET = 0.1
TRUNC = 1e-8
K0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
OUT = output_file(sys.argv[2] if len(sys.argv) > 2 else
                  output_directory("meson", kind="mps") / f"w_meson_j1_ns50_k{K0:.2f}.npz")
print(f"output: {OUT}", flush=True)

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)


# ---- L2-regularized ns=10-trained block parameters (small-angle, MPS-
# friendly creation path; produced by scripts/regularize_wp_params.py)
lat6 = Z2Lattice(6, pbc=True)
TH = stateprep.optimize_vacuum(lat6, M0, G2, ETA, n_layers=2, restarts=2)["thetas"]
cache = f"data/wp10reg_params_k{K0:.2f}_L3.npz"
if not os.path.exists(cache):
    raise SystemExit(f"missing {cache}: run scripts/regularize_wp_params.py first")
z = np.load(cache, allow_pickle=True)
vec, offsets, n_layers, fid = z["vec"], list(z["offsets"]), int(z["L"]), float(z["F"])
log(f"loaded regularized wavepacket params (F={fid:.4f})")

params = wavepacket.params_from_vector(vec, offsets, n_layers)

# ---- ns=50 circuits
lat = Z2Lattice(NS, pbc=True)
vac_prep = stateprep.vacuum_ansatz(lat, TH)
wp_prep = vac_prep.copy()
wp_prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)

probes = [cur.charge_density(lat, v) for v in range(NS)] + \
         [cur.bond_current(lat, b, ETA) for b in range(NS)]
insert = cur.bond_current(lat, CENTER, ETA)
H50 = ham.build_hamiltonian(lat, M0, G2, ETA)

# ---- one-time high-accuracy state preparations (stored as MPS) + <H> cert
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from htensor.measure import split_current

anc_site = min(split_current(insert)[1][0][0])
stored = {}
for name, prep in (("vacuum", vac_prep), ("packet", wp_prep)):
    t1 = time.time()
    mps, perm = backends.prepare_state_mps(lat, prep, anc_site,
                                           cap=512, trunc=1e-10)
    stored[name] = (mps, perm)
    n_tot = lat.n_qubits + 1
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    qc.save_expectation_value(backends.permute_pauli(H50, perm, n_tot),
                              list(range(n_tot)), label="H")
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=TRUNC,
                       max_parallel_threads=4)
    e = sim.run(qc).result().data()["H"]
    log(f"{name}: prep+store {time.time()-t1:.0f}s, <H> = {np.real(e):.6f}")

# ---- correlator grids (every circuit starts from the stored state)
results = {}
for name, key, stat in (("wp", "packet", False), ("vac", "vacuum", True)):
    mps, perm = stored[key]
    rows = []
    for t in TIMES:
        d = backends.hadamard_correlator_aer(
            lat, None, insert, probes, M0, G2, ETA, [t], dt_target=DT_TARGET,
            method="matrix_product_state", mps_trunc=TRUNC,
            stationary_1pt=(stat and t > 0),
            initial_mps=mps, initial_perm=perm)
        rows.append(d)
        log(f"{name} t={t:4.1f} done  C00(t, x=0) = {d.correlator[0, CENTER]:.5f}")
    results[name] = rows

def stack(rows, attr):
    return np.concatenate([getattr(r, attr) for r in rows], axis=0)

np.savez(OUT,
         times=TIMES, k0=K0, ns=NS, center=CENTER,
         corr_wp=stack(results["wp"], "correlator"),
         one_pt_wp=stack(results["wp"], "probe_expect"),
         insert_1pt_wp=results["wp"][0].insert_expect,
         corr_vac=stack(results["vac"], "correlator"),
         one_pt_vac=stack(results["vac"], "probe_expect"),
         insert_1pt_vac=results["vac"][0].insert_expect,
         m0=M0, g2=G2, eta=ETA, dt_target=DT_TARGET, trunc=TRUNC,
         wp_fidelity_ns8=fid)
log(f"saved {OUT}")
