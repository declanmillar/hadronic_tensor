"""IBM Heron hardware path for the 101-qubit production point (paper Sec.
"Toward hardware").  Three tiers of increasing depth/risk, all reusing the
exact production circuit builders:

  tier 1  certify   : prep-only certificates -- 50 Gauss-law stabilizers
                      <G_n> (each weight-3, value 1 in the ideal state and a
                      direct local-fidelity witness), charge profile
                      <J0(v)> (packet bump on staggered background), <H>,
                      fermion parity, total charge.  100 qubits, no Trotter.
  tier 2  transport : packet charge transport <J0(v,t)> after 1-4 Trotter
                      steps (t = 0.5..2.0, dt = 0.5).  One-point functions
                      only -- no ancilla, no controlled gates.  Truth curves
                      exist in data/w_meson_ns50_k0.00_v3.npz (one_pt_wp).
  tier 3  hadamard  : W00 integrand C(t,x) at t = 0.5, 1.0 for 5 probes
                      around the packet.  J0 insertion is a SINGLE
                      controlled-Z (J0 = ((-1)^v - Z_v)/2), so the overhead
                      over tier 2 is one ancilla + one CZ + basis rotation.

Modes:
  audit [fez|torino]        offline: transpile all tiers against a fake
                            Heron target, report depth / 2q-gate counts /
                            pub sizes / suggested shots.  No credentials.
  submit <tier> [backend]   submit one tier via QiskitRuntimeService
                            (saved account or QISKIT_IBM_TOKEN env).
                            Writes data/hw/job_tier<k>_*.json.
  analyze <jobfile>         fetch results, save npz next to the job file,
                            print comparisons against MPS truth.

  PYTHONPATH=. .venv/bin/python scripts/ibm_hardware.py audit
"""

import json
import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from htensor import Z2Lattice, stateprep, wavepacket
from htensor import hamiltonian as ham
from htensor import currents as cur
from htensor import trotter
from htensor.measure import split_current, hadamard_test_circuit

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
DT = 0.5
T2_TIMES = [0.5, 1.0, 1.5, 2.0]
T3_TIMES = [0.0, 0.5, 1.0]  # t=0: known-truth slice to calibrate damping
T3_PROBES = [CENTER + o for o in (-4, -2, 0, 2, 4)]
SHOTS = {1: 8192, 2: 8192, 3: 100_000}
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def build_prep(lat):
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                   n_layers=2, restarts=2)["thetas"]
    z = np.load("data/wp10reg_params_k0.00_L3.npz", allow_pickle=True)
    params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                           int(z["L"]))
    prep = stateprep.vacuum_ansatz(lat, TH)
    prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)
    return prep


def with_ancilla(op, anc_pauli):
    return SparsePauliOp([anc_pauli + l for l in op.paulis.to_labels()],
                         op.coeffs)


def build_tier(tier, lat, prep):
    """-> list of (circ_name, circuit, [(obs_name, SparsePauliOp)])."""
    if tier == 1:
        obs = [(f"G{n}", ham.gauss_operator(lat, n)) for n in range(NS)]
        obs += [(f"J0_{v}", cur.charge_density(lat, v)) for v in range(NS)]
        obs += [("H", ham.build_hamiltonian(lat, M0, G2, ETA)),
                ("Pf", ham.fermion_parity(lat)),
                ("Q", ham.total_charge(lat))]
        return [("prep", prep, obs)]
    if tier == 2:
        out = []
        obs = [(f"J0_{v}", cur.charge_density(lat, v)) for v in range(NS)]
        for t in T2_TIMES:
            qc = prep.copy()
            qc.compose(trotter.trotter_circuit(lat, M0, G2, ETA, t,
                                               int(round(t / DT))),
                       inplace=True)
            out.append((f"t{t:.1f}", qc, obs))
        return out
    if tier == 3:
        # J0(center) = id_c + c_z * Z_{2*CENTER}: one controlled-Z insertion
        id_c, terms = split_current(cur.charge_density(lat, CENTER))
        (ins_ops, ins_coeff), = terms
        out = []
        for t in T3_TIMES:
            evo = trotter.trotter_circuit(lat, M0, G2, ETA, t,
                                          int(round(t / DT))).to_instruction()
            gadget = hadamard_test_circuit(lat.n_qubits, ins_ops, evo)
            qc = QuantumCircuit(lat.n_qubits + 1)
            qc.compose(prep, range(lat.n_qubits), inplace=True)
            qc.compose(gadget, inplace=True)
            obs = []
            for v in T3_PROBES:
                B = cur.charge_density(lat, v)
                obs += [(f"XB_{v}", with_ancilla(B, "X")),
                        (f"YB_{v}", with_ancilla(B, "Y")),
                        (f"B_{v}", with_ancilla(B, "I"))]
            obs += [("X", with_ancilla(
                SparsePauliOp("I" * lat.n_qubits), "X")),
                ("Y", with_ancilla(SparsePauliOp("I" * lat.n_qubits), "Y"))]
            qc.metadata = {"id_c": id_c, "ins_coeff": ins_coeff, "t": t}
            out.append((f"t{t:.1f}", qc, obs))
        return out
    raise SystemExit(f"unknown tier {tier}")


def fake_backend(name):
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeTorino
    return FakeFez() if name == "fez" else FakeTorino()


def isa_transpile(circs, backend):
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                      seed_transpiler=7)
    return [pm.run(c) for c in circs]


def audit(backend_name="fez"):
    be = fake_backend(backend_name)
    log(f"audit target: {be.name} ({be.num_qubits}q, "
        f"basis {be.operation_names})")
    lat = Z2Lattice(NS, pbc=True)
    prep = build_prep(lat)
    for tier in (1, 2, 3):
        entries = build_tier(tier, lat, prep)
        circs = [c for _, c, _ in entries]
        isa = isa_transpile(circs, be)
        for (name, _, obs), tc in zip(entries, isa):
            two_q = sum(v for k, v in tc.count_ops().items()
                        if k in ("cz", "ecr", "cx"))
            log(f"tier {tier} [{name}]: depth {tc.depth()}, "
                f"2q gates {two_q}, 2q depth "
                f"{tc.depth(lambda i: i.operation.num_qubits == 2)}, "
                f"{len(obs)} observables, shots {SHOTS[tier]}")
    log("audit done")


def submit(tier, backend_name=None):
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
    service = QiskitRuntimeService()
    be = (service.backend(backend_name) if backend_name
          else service.least_busy(min_num_qubits=NS * 2 + 5,
                                  operational=True))
    log(f"backend: {be.name}")
    lat = Z2Lattice(NS, pbc=True)
    prep = build_prep(lat)
    entries = build_tier(tier, lat, prep)
    isa = isa_transpile([c for _, c, _ in entries], be)
    pubs = []
    for (name, _, obs), tc in zip(entries, isa):
        layout_obs = [o.apply_layout(tc.layout) for _, o in obs]
        pubs.append((tc, layout_obs))
    est = EstimatorV2(mode=be)
    est.options.default_shots = SHOTS[tier]
    est.options.dynamical_decoupling.enable = True
    est.options.dynamical_decoupling.sequence_type = "XY4"
    est.options.twirling.enable_gates = True
    est.options.twirling.enable_measure = True
    est.options.resilience.measure_mitigation = True
    job = est.run(pubs)
    os.makedirs("data/hw", exist_ok=True)
    meta = {"job_id": job.job_id(), "tier": tier, "backend": be.name,
            "shots": SHOTS[tier],
            "circ_names": [n for n, _, _ in entries],
            "obs_names": [[on for on, _ in obs] for _, _, obs in entries],
            "circ_meta": [c.metadata for _, c, _ in entries]}
    path = f"data/hw/job_tier{tier}_{job.job_id()}.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"submitted {job.job_id()} -> {path}")


def analyze(jobfile):
    from qiskit_ibm_runtime import QiskitRuntimeService
    meta = json.load(open(jobfile))
    service = QiskitRuntimeService()
    res = service.job(meta["job_id"]).result()
    out = {}
    for names, pub in zip(meta["obs_names"], res):
        ev = np.asarray(pub.data.evs, dtype=float)
        for n, v in zip(names, np.atleast_1d(ev)):
            out[n] = v
    np.savez(jobfile.replace(".json", ".npz"), **out,
             tier=meta["tier"], backend=meta["backend"])
    tier = meta["tier"]
    if tier == 1:
        g = np.array([out[f"G{n}"] for n in range(NS)])
        log(f"Gauss stabilizers: mean {g.mean():.4f}, min {g.min():.4f} "
            f"(ideal 1; mean is a local-fidelity witness)")
        log(f"<H> = {out['H']:.3f} (MPS truth -49.37), "
            f"Pf = {out['Pf']:.3f}, Q = {out['Q']:.3f}")
    log(f"saved {jobfile.replace('.json', '.npz')}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "audit":
        audit(sys.argv[2] if len(sys.argv) > 2 else "fez")
    elif mode == "submit":
        submit(int(sys.argv[2]),
               sys.argv[3] if len(sys.argv) > 3 else None)
    elif mode == "analyze":
        analyze(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {mode}")
