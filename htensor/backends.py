"""Aer execution layer for the Hadamard-test correlator protocol.

Runs the same circuit families as measure.hadamard_correlator_sv but through
qiskit-aer, so the identical driver covers:
  - method='statevector'          exact, <= ~28 qubits
  - method='matrix_product_state' 50-200 qubits (the production engine);
    bond dimension / truncation exposed via mps_* options
  - shots=None                    exact expectation values (save_expectation_value)
(Sampled/shot-based execution with basis rotations is the hardware layer, M5.)

State preparation is a CIRCUIT here (e.g. stateprep.vacuum_ansatz output),
not a vector -- at 100 qubits there is no vector.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

from .lattice import Z2Lattice
from . import trotter
from .measure import (CorrelatorData, split_current, controlled_pauli,
                      _with_ancilla, _pauli_only)

_AER_BASIS = ["cx", "cz", "rz", "rx", "ry", "sx", "x", "h"]


# ----------------------------------------------------------- MPS ring folding
def ring_chain_order(lat: Z2Lattice, ancilla_after: int | None = None) -> list[int]:
    """Chain order (list of logical qubits) folding the PBC ring in half:
    [0, n-1, 1, n-2, ...].  Every ring-neighbor pair -- INCLUDING the seam
    (n-1, 0) -- sits <= 2 apart in the chain, so Aer's MPS never inserts
    long swap chains.  If ancilla_after is a logical system qubit, the
    ancilla (logical index n) is placed directly after it in the chain."""
    n = lat.n_qubits
    order = []
    for i in range(n // 2):
        order += [i, n - 1 - i]
    if n % 2:
        order.append(n // 2)
    if ancilla_after is not None:
        order.insert(order.index(ancilla_after) + 1, n)
    return order


def _chain_perm(order: list[int]) -> dict[int, int]:
    return {logical: chain for chain, logical in enumerate(order)}


def permute_circuit(circ: QuantumCircuit, perm: dict[int, int],
                    n_total: int) -> QuantumCircuit:
    out = QuantumCircuit(n_total)
    for inst in circ.data:
        out.append(inst.operation,
                   [perm[circ.find_bit(b).index] for b in inst.qubits])
    return out


def permute_pauli(op: SparsePauliOp, perm: dict[int, int],
                  n_total: int) -> SparsePauliOp:
    labels = []
    nq = op.num_qubits
    for lab in op.paulis.to_labels():
        new = ["I"] * n_total
        for q in range(nq):
            new[n_total - 1 - perm[q]] = lab[nq - 1 - q]
        labels.append("".join(new))
    return SparsePauliOp(labels, op.coeffs)


# ----------------------------------------------------------- chain routing
# Aer's MPS handles a non-adjacent two-qubit gate by moving one qubit next to
# its partner and back, with an SVD truncation at every move.  In the folded
# ring order the hop pair (a, b) of a bond sits four chain sites apart (the
# link qubit and the mirror-image qubits lie between), so a 50-site prep
# incurs thousands of such excursions and the stored state is truncation-
# limited even at bond cap 512 (packet <H> -49.3541 vs the converged
# -49.3690, found 2026-09-03).  Routing the circuit onto the line with
# explicit, persistent SWAPs (Sabre) keeps every gate chain-local; the
# converged energy is then independent of cap (512-2048), threshold
# (1e-10..1e-14) and routing seed, and matches exact statevectors at Ns=12.
ROUTE_CHAIN = True
ROUTE_SEED = 7


def route_chain(circ: QuantumCircuit, n_total: int, seed: int = ROUTE_SEED):
    """Sabre-route a chain-ordered circuit onto the linear chain of n_total
    sites.  -> (routed circuit, fl) with fl[virtual chain position] = final
    chain position after the inserted SWAPs."""
    from qiskit.transpiler import CouplingMap
    tqc = transpile(circ, coupling_map=CouplingMap.from_line(n_total),
                    basis_gates=_AER_BASIS, initial_layout=list(range(n_total)),
                    optimization_level=1, seed_transpiler=seed,
                    routing_method="sabre")
    fl = tqc.layout.final_index_layout()
    return tqc, {i: int(fl[i]) for i in range(n_total)}


def compose_perm(perm: dict[int, int], fl: dict[int, int]) -> dict[int, int]:
    """logical -> chain position, after the routing permutation fl."""
    return {q: fl[p] for q, p in perm.items()}


def chain_transpile(circ: QuantumCircuit, perm: dict[int, int], n_total: int,
                    circuit_transform=None, route: bool | None = None):
    """Permute a logical circuit into chain order, transpile to the Aer basis,
    apply an optional transform (e.g. a Pauli-noise trajectory -- applied
    BEFORE routing so the simulation-only SWAPs stay noiseless), then route
    onto the chain.  -> (circuit ready for Aer, perm_obs) where perm_obs maps
    logical qubits to the chain positions in which observables must be read
    (permute_pauli(op, perm_obs, n_total))."""
    route = ROUTE_CHAIN if route is None else route
    qc = permute_circuit(circ, perm, n_total)
    tqc = transpile(qc, basis_gates=_AER_BASIS, optimization_level=1)
    if circuit_transform is not None:
        tqc = circuit_transform(tqc)
    if not route:
        return tqc, perm
    tqc, fl = route_chain(tqc, n_total)
    return tqc, compose_perm(perm, fl)


def _simulator(method: str, mps_max_bond: int | None, mps_trunc: float,
               max_threads: int):
    from qiskit_aer import AerSimulator

    opts = {"method": method, "max_parallel_threads": max_threads}
    if method == "matrix_product_state":
        if mps_max_bond is not None:
            opts["matrix_product_state_max_bond_dimension"] = mps_max_bond
        opts["matrix_product_state_truncation_threshold"] = mps_trunc
    return AerSimulator(**opts)


def prepare_state_mps(lat: Z2Lattice, prep: QuantumCircuit, anc_site: int,
                      cap: int = 512, trunc: float = 1e-10,
                      max_threads: int = 4, circuit_transform=None,
                      sim_opts: dict | None = None):
    """Simulate the (expensive) preparation circuit ONCE at high accuracy and
    return (mps_data, perm) for reuse via set_matrix_product_state.

    The register includes one idle ancilla placed beside anc_site in the
    folded chain, so downstream Hadamard-test circuits can start directly
    from the stored state.  This isolates the one-time intermediate
    entanglement of the wavepacket block from the per-circuit cost.
    """
    from qiskit_aer import AerSimulator

    perm = _chain_perm(ring_chain_order(lat, ancilla_after=anc_site))
    n_tot = lat.n_qubits + 1
    # the returned perm is the chain order of the STORED state (routing SWAPs
    # composed in); every consumer must read observables through it
    tqc, perm = chain_transpile(prep, perm, n_tot, circuit_transform)
    tqc.save_matrix_product_state(label="mps")
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_max_bond_dimension=cap,
                       matrix_product_state_truncation_threshold=trunc,
                       max_parallel_threads=max_threads,
                       **(sim_opts or {}))
    data = sim.run(tqc).result().data()
    return data["mps"], perm


def hadamard_correlator_aer(lat: Z2Lattice, prep: QuantumCircuit,
                            insert_op: SparsePauliOp,
                            probe_ops: list[SparsePauliOp],
                            m0, g2, eta, times, dt_target: float = 0.05,
                            method: str = "matrix_product_state",
                            mps_max_bond: int | None = None,
                            mps_trunc: float = 1e-12,
                            max_threads: int = 4,
                            fold: bool | None = None,
                            stationary_1pt: bool = False,
                            initial_mps=None,
                            initial_perm: dict | None = None,
                            circuit_transform=None) -> CorrelatorData:
    """Hadamard-test correlator grid via Aer with exact expectation values.

    One circuit per (insertion Pauli term, time); every probe observable is
    read from the same circuit via multiple save_expectation_value slots --
    the x-multiplexing that makes this protocol cheap.

    fold=True (default for the MPS method) relabels qubits with the folded
    ring order so the PBC seam and the ancilla are chain-local.

    initial_mps/initial_perm (from prepare_state_mps): start every circuit
    from the stored prepared state instead of re-simulating `prep`; all
    circuits then run on the n+1-qubit register in the stored layout.
    """
    times = np.asarray(times, dtype=float)
    n_sys = lat.n_qubits
    if fold is None:
        fold = method == "matrix_product_state"
    use_init = initial_mps is not None
    if use_init and initial_perm is None:
        raise ValueError("initial_mps requires initial_perm")
    sim = _simulator(method, mps_max_bond, mps_trunc, max_threads)
    id_a, terms_a = split_current(insert_op)
    probes_p = [_pauli_only(op) for op in probe_ops]
    id_b = np.array([split_current(op)[0] for op in probe_ops])

    anc_site = min(terms_a[0][0]) if terms_a else 0
    if use_init:
        perm_sys = perm_anc = initial_perm
    else:
        perm_sys = _chain_perm(ring_chain_order(lat)) if fold else None
        perm_anc = _chain_perm(ring_chain_order(lat, ancilla_after=anc_site)) \
            if fold else None

    def run(circ, observables, perm):
        n_total = (n_sys + 1) if use_init else circ.num_qubits
        if perm is not None:
            # chain order + explicit routing (ROUTE_CHAIN); observables are
            # read through the composed permutation
            tqc, perm_obs = chain_transpile(circ, perm, n_total, circuit_transform)
            observables = {lbl: permute_pauli(op, perm_obs, n_total)
                           for lbl, op in observables.items()}
        else:
            # transpile first: save/set instructions cannot pass the basis
            # translator; basis-only transpilation does no routing
            tqc = transpile(circ, basis_gates=_AER_BASIS, optimization_level=1)
            if circuit_transform is not None:
                tqc = circuit_transform(tqc)
        if use_init:
            full = QuantumCircuit(n_total)
            full.set_matrix_product_state(initial_mps)
            full.compose(tqc, inplace=True)
            tqc = full
        for lbl, op in observables.items():
            tqc.save_expectation_value(op, list(range(n_total)), label=lbl)
        return sim.run(tqc).result().data()

    # ---- plain single-current expectations (disconnected/identity pieces)
    # stationary_1pt: for (near-)eigenstate preps <J(t)> is t-independent --
    # measure at t=0 and broadcast (halves the circuit count)
    probe_expect = np.empty((len(times), len(probe_ops)), dtype=complex)
    # <A(0)> must be measured at t = 0 REGARDLESS of the requested times:
    # it multiplies the probe-identity term id_b <A(0)> of <B(t) A(0)>.
    # (A per-call capture at times[0] silently became <A(t)> when callers
    # loop single-time lists -- the id_b-weighted secular bug fixed on
    # 2026-07-07; all earlier grids are repaired post hoc from one_pt.)
    qc0 = QuantumCircuit(n_sys + 1 if use_init else n_sys)
    if not use_init:
        qc0.compose(prep, inplace=True)
    insert_expect = complex(run(qc0, {"ins": insert_op}, perm_sys)["ins"])
    t_list = times[:1] if stationary_1pt else times
    for i, t in enumerate(t_list):
        qc = QuantumCircuit(n_sys + 1 if use_init else n_sys)
        if not use_init:
            qc.compose(prep, inplace=True)
        n = max(1, int(np.ceil(abs(t) / dt_target))) if t != 0 else 0
        qc.compose(trotter.trotter_circuit(lat, m0, g2, eta, t, n),
                   qubits=range(n_sys), inplace=True)
        obs = {f"p{j}": op for j, op in enumerate(probe_ops)}
        data = run(qc, obs, perm_sys)
        probe_expect[i] = [data[f"p{j}"] for j in range(len(probe_ops))]
    if stationary_1pt:
        probe_expect[1:] = probe_expect[0]
    probe_p_expect = probe_expect - id_b[None, :]

    # ---- ancilla circuits
    corr = np.zeros((len(times), len(probe_ops)), dtype=complex)
    anc = n_sys
    for ops_a, c_a in terms_a:
        for i, t in enumerate(times):
            qc = QuantumCircuit(n_sys + 1)
            if not use_init:
                qc.compose(prep, qubits=range(n_sys), inplace=True)
            qc.h(anc)
            controlled_pauli(qc, anc, ops_a)
            n = max(1, int(np.ceil(abs(t) / dt_target))) if t != 0 else 0
            qc.compose(trotter.trotter_circuit(lat, m0, g2, eta, t, n),
                       qubits=range(n_sys), inplace=True)
            obs = {}
            for j, bp in enumerate(probes_p):
                obs[f"re{j}"] = _with_ancilla(bp, "X")
                obs[f"im{j}"] = _with_ancilla(bp, "Y")
            data = run(qc, obs, perm_anc)
            for j in range(len(probes_p)):
                corr[i, j] += c_a * (np.real(data[f"re{j}"])
                                     + 1j * np.real(data[f"im{j}"]))

    for j in range(len(probe_ops)):
        corr[:, j] += (id_a * probe_p_expect[:, j]
                       + id_b[j] * (insert_expect - id_a) + id_a * id_b[j])
    return CorrelatorData(times, corr, probe_expect, insert_expect)
