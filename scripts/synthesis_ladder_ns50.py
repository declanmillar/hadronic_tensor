"""Task 13 main run: synthesis-error ladder at 101 qubits.

For each rotation-grid spacing delta (None = exact baseline): snap ALL
rotations in prep and measurement circuits (post-transpile), re-prepare the
stored MPS, measure the on-circuit <H> certificate and a reduced rest-packet
W^{00} grid (J0 probes, t = 0..6, dt_target = 0.5 -- the larger-angle,
FT-friendlier Trotter regime).

  PYTHONPATH=. .venv/bin/python scripts/synthesis_ladder_ns50.py
"""

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from htensor import Z2Lattice, stateprep, wavepacket, backends, synthesis
from htensor import currents as cur
from htensor import hamiltonian as ham
from htensor.measure import split_current

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
TIMES = np.arange(0.0, 6.01, 0.5)
LADDER = [(None, "exact", 0)] + \
    [(np.pi / d, "stochastic", 1) for d in (128, 64, 32, 16, 8, 4, 2)] + \
    [(np.pi / d, "round", 0) for d in (64, 16)]
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                               n_layers=2, restarts=2)["thetas"]
z = np.load("data/wp10reg_params_k0.00_L3.npz", allow_pickle=True)
params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]), int(z["L"]))
lat = Z2Lattice(NS, pbc=True)
prep = stateprep.vacuum_ansatz(lat, TH)
prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)
probes = [cur.charge_density(lat, v) for v in range(NS)]
insert = cur.charge_density(lat, CENTER)
H50 = ham.build_hamiltonian(lat, M0, G2, ETA)
anc_site = min(split_current(insert)[1][0][0])

import os
import sys

# optional worker split: argv[1] in {0,1} takes alternating ladder points
# (skip-existing logic keeps workers from redoing finished points).
# single-point seed-ensemble mode: argv = point <pi_denom> <mode> <seed>
SUFFIX = ""
CHI = None                       # optional bond cap for the evolution
if len(sys.argv) >= 5 and sys.argv[1] == "point":
    _d, _m, _s = int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    LADDER = [(np.pi / _d, _m, _s)]
    SUFFIX = f"_s{_s}"
    if len(sys.argv) > 5:        # capped-chi variant: point <d> <mode> <s> <chi>
        CHI = int(sys.argv[5])
        SUFFIX += f"_chi{CHI}"
elif len(sys.argv) > 1:
    half = int(sys.argv[1])
    LADDER = [pt for i, pt in enumerate(LADDER) if i % 2 == half]

for delta, mode, seed in LADDER:
    tag = ("exact" if delta is None
           else f"{mode[:4]}_pi{int(round(np.pi/delta))}") + SUFFIX
    if os.path.exists(f"data/synthladder_{tag}.npz"):
        log(f"{tag}: already done, skipping")
        continue
    tf = None if delta is None else \
        (lambda c, d=delta, m=mode, s=seed: synthesis.snap_angles(c, d, m, s))
    mps, perm = backends.prepare_state_mps(lat, prep, anc_site, cap=512,
                                           trunc=1e-10, circuit_transform=tf)
    n_tot = lat.n_qubits + 1
    qc = QuantumCircuit(n_tot)
    qc.set_matrix_product_state(mps)
    qc.save_expectation_value(backends.permute_pauli(H50, perm, n_tot),
                              list(range(n_tot)), label="H")
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=1e-8,
                       max_parallel_threads=4)
    e = float(np.real(sim.run(qc).result().data()["H"]))
    log(f"{tag}: <H> = {e:.4f}")
    rows = []
    for t in TIMES:
        d = backends.hadamard_correlator_aer(
            lat, None, insert, probes, M0, G2, ETA, [t], dt_target=0.5,
            method="matrix_product_state", mps_trunc=1e-8, mps_max_bond=CHI,
            initial_mps=mps, initial_perm=perm, circuit_transform=tf)
        rows.append(d)
        log(f"{tag}: t={t:.1f} done")
    corr = np.concatenate([r.correlator for r in rows], axis=0)
    one = np.concatenate([r.probe_expect for r in rows], axis=0)
    np.savez(f"data/synthladder_{tag}.npz", times=TIMES, corr=corr,
             one_pt=one, insert_1pt=rows[0].insert_expect, H=e,
             delta=0.0 if delta is None else delta, mode=mode, seed=seed)
    log(f"{tag}: grid saved")
log("ladder complete")
