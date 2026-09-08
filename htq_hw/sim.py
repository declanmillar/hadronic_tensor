"""Offline simulation: ideal grids, ISA-to-logical relabelling, cached prep
MPS, submission-path check and dress rehearsal.

check: every pub family's ISA circuits are mapped back to the logical
register through the recorded layouts (scripts/ibm_hardware.py:457-503 does
the equivalent with observable.apply_layout) and simulated with Aer,
statevector for ns <= 10 or matrix_product_state (cap 512) from a prep MPS
the package builds once per card (save_matrix_product_state); layout-mapped
expectation values must agree with the ideal grid to 5e-3.

rehearse: the same ISA circuits with their readout layers are sampled in
Aer (optionally after a Pauli-trajectory noise transform, one random Pauli
per faulty gate: p2 per 2q gate, p1 per 1q gate, scripts/hw_cal_grids.py:113-130),
written in the fetch format and pushed through analyze into slice files.
"""

import os
import pickle
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from . import DT, ETA, MIRROR_EPS, __version__
from . import analyze as A
from . import campaign as CP
from . import circuits as C
from . import target as T
from .model import Lattice, to_sparse_pauli_op, charge_terms, current_terms, hamiltonian_terms

STATEVECTOR_MAX_WIRES = 24
# Aer's MPS memory estimator mis-sizes circuits with many rxx/ryy/rzz gates
# (13 TB "required" for the 101-qubit prep); the reference pipeline's basis
# (htensor/backends.py:24) avoids it, so everything is unrolled to it first.
AER_BASIS = ["cx", "cz", "rz", "rx", "ry", "sx", "x", "h"]


# ------------------------------------------------------------------ simulators
def make_simulator(n_wires: int, method: str | None = None, cap: int = 512, trunc: float = 1e-8,
                   threads: int = 2):
    from qiskit_aer import AerSimulator

    method = method or ("statevector" if n_wires <= STATEVECTOR_MAX_WIRES else "matrix_product_state")
    opts = {"method": method, "max_parallel_threads": threads}
    if method == "matrix_product_state":
        opts["matrix_product_state_max_bond_dimension"] = cap
        opts["matrix_product_state_truncation_threshold"] = trunc
    return AerSimulator(**opts)


def _aer_ready(qc: QuantumCircuit) -> QuantumCircuit:
    from qiskit import transpile
    return transpile(qc, basis_gates=AER_BASIS, optimization_level=1)


# ------------------------------------------------------------------ relabelling
def relabel_to_logical(isa: QuantumCircuit, initial_layout, n_wires: int):
    """Physical-register ISA circuit -> circuit on the logical register: physical
    initial_layout[i] becomes wire i; other touched physical qubits get wires
    n_wires, n_wires+1, ...  -> (circuit, {physical: wire}).  Measurements are
    dropped (the rehearsal re-adds them)."""
    wire = {int(p): i for i, p in enumerate(initial_layout)}
    body = isa.remove_final_measurements(inplace=False)
    touched = sorted({isa.find_bit(q).index for inst in body.data for q in inst.qubits})
    for p in touched:
        if p not in wire:
            wire[p] = len(wire)
    out = QuantumCircuit(max(n_wires, len(wire)))
    out.global_phase = isa.global_phase
    for inst in body.data:
        out.append(inst.operation, [wire[isa.find_bit(q).index] for q in inst.qubits])
    return out, wire


def final_wires(initial_layout, final_layout, wire: dict) -> list[int]:
    """Wire holding logical j at the end of the relabelled circuit."""
    return [wire[int(final_layout[j])] for j in range(len(initial_layout))]


# ------------------------------------------------------------------ observables
def _with_anc(lat: Lattice, terms, anc_pauli: str) -> SparsePauliOp:
    t = [({**ops, lat.ancilla: anc_pauli} if anc_pauli != "I" else ops, c) for ops, c in terms]
    return to_sparse_pauli_op(lat.n_wires, t)


def probe_observables(lat: Lattice, probes=("J0", "J1"), eta: float = ETA) -> dict:
    """{'XB_v','YB_v','B_v','XT1_b',...,'X','Y'} on the logical register
    (scripts/hw_cal_grids.py:95-110; T terms carry the exact seam string)."""
    obs = {}
    if "J0" in probes:
        for v in range(lat.ns):
            for tag, ap in (("XB", "X"), ("YB", "Y"), ("B", "I")):
                obs[f"{tag}_{v}"] = _with_anc(lat, charge_terms(lat, v), ap)
    if "J1" in probes:
        for b in range(lat.ns):
            terms = current_terms(lat, b, 1.0, exact_seam=True)
            for k, (ops, c) in enumerate(terms, start=1):
                for tag, ap in ((f"XT{k}", "X"), (f"YT{k}", "Y"), (f"T{k}", "I")):
                    obs[f"{tag}_{b}"] = _with_anc(lat, [(ops, 1.0)], ap)
    for ap in ("X", "Y"):
        obs[ap] = _with_anc(lat, [({}, 1.0)], ap)
    return obs


def one_point_observables(lat: Lattice, eta: float = ETA) -> dict:
    obs = {}
    for v in range(lat.ns):
        obs[f"J0_{v}"] = _with_anc(lat, charge_terms(lat, v), "I")
        obs[f"J1_{v}"] = _with_anc(lat, current_terms(lat, v, eta, exact_seam=True), "I")
    return obs


def remap(op: SparsePauliOp, wires: list[int], n_out: int) -> SparsePauliOp:
    """Move an n_wires-qubit operator onto wires[j] of an n_out register."""
    labels = []
    for lab in op.paulis.to_labels():
        n = len(lab)
        out = ["I"] * n_out
        for j in range(n):
            ch = lab[n - 1 - j]
            if ch != "I":
                out[n_out - 1 - wires[j]] = ch
        labels.append("".join(out))
    return SparsePauliOp(labels, op.coeffs)


# ------------------------------------------------------------------ prep MPS cache
def prep_state_circuit(card: dict, lat: Lattice, center: int) -> QuantumCircuit:
    """prep on the system wires, ancilla idle (logical register)."""
    qc = QuantumCircuit(lat.n_wires)
    qc.compose(C.prep_circuit(card, lat.ns, center), range(lat.n_qubits), inplace=True)
    return qc


class PrepState:
    """Cached prep MPS in the folded chain order (+ its wire -> chain map)."""

    def __init__(self, mps, H: float, perm: list[int], order: list[int]):
        self.mps, self.H, self.perm, self.order = mps, H, list(perm), list(order)


def prep_mps(card: dict, lat: Lattice, center: int, cache_dir, cap: int = 512, trunc: float = 1e-10,
             threads: int = 2, log=None) -> PrepState:
    """Prep MPS (system + idle ancilla) in the folded ring chain order with the
    ancilla beside the centre site qubit, built once per card and cached as a
    pickle with <H> certified and the chain map recorded
    (htensor/backends.py:82-108, scripts/prep_common.py:125-154)."""
    order = chain_order(lat, ancilla_after=lat.site_qubit(center))
    perm = chain_perm(order)
    key = (f"prep_{card['name']}_ns{lat.ns}_c{center}_cap{cap}_tr{trunc:g}_foldroute_v{__version__}.pkl")
    path = os.path.join(cache_dir, key) if cache_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            ck = pickle.load(f)
        T._log(f"prep MPS from {path}  <H> = {ck['H']:.6f}", log)
        return PrepState(ck["mps"], ck["H"], ck["perm"], ck["order"])
    sim = make_simulator(lat.n_wires, "matrix_product_state", cap, trunc, threads)
    n = lat.n_wires
    t0 = time.time()
    qc, fl = route_to_chain(prep_state_circuit(card, lat, center), perm, n)
    H = to_sparse_pauli_op(n, hamiltonian_terms(lat, *[card["couplings"][k] for k in ("m0", "g2", "eta")]))
    qc.save_expectation_value(permute_pauli(H, fl, n), list(range(n)), label="H")
    qc.save_matrix_product_state(label="mps")
    d = sim.run(qc).result().data()
    mps, Hval = d["mps"], float(np.real(d["H"]))
    chi = max(int(np.asarray(l).shape[0]) for l in mps[1])
    T._log(f"prep MPS built in {time.time() - t0:.0f}s (cap {cap}, trunc {trunc:g}, max bond {chi}, "
           f"routed on the folded chain)  <H> = {Hval:.6f}", log)
    final_order = [0] * n
    for wire, pos in enumerate(fl):
        final_order[pos] = wire
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"mps": mps, "H": Hval, "perm": fl, "order": final_order, "card": card["name"],
                         "ns": lat.ns, "center": center, "cap": cap, "trunc": trunc, "max_bond": chi,
                         "initial_order": order}, f)
    return PrepState(mps, Hval, fl, final_order)


def pad_mps(mps, n_extra: int):
    """Append n_extra |0> sites to an Aer MPS tuple."""
    if n_extra <= 0:
        return mps
    g, lam = mps
    g2 = list(g) + [(np.array([[1.0 + 0j]]), np.array([[0.0 + 0j]]))] * n_extra
    lam2 = list(lam) + [np.array([1.0])] * n_extra
    return (g2, lam2)


# ------------------------------------------------------------------ folded chain order
def chain_order(lat: Lattice, ancilla_after: int | None = None) -> list[int]:
    """MPS chain order folding the PBC ring in half, [0, n-1, 1, n-2, ...],
    so every ring-neighbour pair INCLUDING the seam sits <= 2 apart; the
    ancilla (wire 2Ns) goes directly after ``ancilla_after``
    (htensor/backends.py:28-42).  Keeps the bond dimension below the cap."""
    n = lat.n_qubits
    order = []
    for i in range(n // 2):
        order += [i, n - 1 - i]
    if n % 2:
        order.append(n // 2)
    if ancilla_after is not None:
        order.insert(order.index(ancilla_after) + 1, lat.ancilla)
    return order


def chain_perm(order: list[int]) -> list[int]:
    """wire -> chain position (htensor/backends.py:45-46), as a list."""
    perm = [0] * len(order)
    for chain, wire in enumerate(order):
        perm[wire] = chain
    return perm


def full_perm(perm, n: int) -> list[int]:
    """Extend a wire->chain map to n wires: extra wires keep the identity
    beyond the folded block (they are appended |0> sites of the MPS)."""
    perm = list(perm)
    return perm + list(range(len(perm), n))


def permute_circuit(qc: QuantumCircuit, perm, n_total: int) -> QuantumCircuit:
    """Relabel wires by ``perm`` (htensor/backends.py:49-55); clbits kept."""
    out = QuantumCircuit(n_total, qc.num_clbits)
    out.global_phase = qc.global_phase
    for inst in qc.data:
        out.append(inst.operation, [perm[qc.find_bit(q).index] for q in inst.qubits],
                   [qc.find_bit(c).index for c in inst.clbits])
    return out


def permute_pauli(op: SparsePauliOp, perm, n_total: int) -> SparsePauliOp:
    """Move qubit q of ``op`` to perm[q] (htensor/backends.py:58-67)."""
    labels = []
    nq = op.num_qubits
    for lab in op.paulis.to_labels():
        new = ["I"] * n_total
        for q in range(nq):
            new[n_total - 1 - perm[q]] = lab[nq - 1 - q]
        labels.append("".join(new))
    return SparsePauliOp(labels, op.coeffs)


def route_to_chain(circuit: QuantumCircuit, perm, n_total: int, seed: int = 7):
    """Sabre-route a wire circuit onto the linear MPS chain (initial layout
    ``perm``: wire -> chain position) so every two-qubit gate is chain-local.
    Aer otherwise moves a qubit to its partner and back for each non-adjacent
    gate (the folded ring puts the a-b hop pair 4 sites apart), which is
    ~3 s/gate at Ns = 50; persistent SWAP routing is ~5 ms/gate.
    -> (routed circuit on chain positions, final wire -> chain list)."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    pl = full_perm(perm, n_total)
    tqc = transpile(circuit, coupling_map=CouplingMap.from_line(n_total), basis_gates=AER_BASIS,
                    initial_layout=pl, optimization_level=1, seed_transpiler=seed,
                    routing_method="sabre")
    return tqc, list(tqc.layout.final_index_layout())


# ------------------------------------------------------------------ expectation values
def expectations(sim, circuit: QuantumCircuit, obs: dict, initial_mps=None, noise=None,
                 perm=None) -> dict:
    """Expectation values of {label: SparsePauliOp on the circuit's wires}.
    ``perm`` (wire -> chain position) routes the circuit onto the folded MPS
    chain (route_to_chain) and reads the observables through the final
    layout; ``initial_mps`` must then be in chain order ``perm``.  Noise is
    applied to the wire circuit (before the simulation-only SWAPs)."""
    n = circuit.num_qubits
    qc = QuantumCircuit(n)
    if initial_mps is not None:
        qc.set_matrix_product_state(pad_mps(initial_mps, n - len(initial_mps[0])))
    body = noise_transform(circuit, *noise) if noise is not None else circuit
    if perm is not None:
        body, fl = route_to_chain(body, perm, n)
    else:
        body, fl = _aer_ready(body), None
    qc.compose(body, inplace=True)
    for lbl, op in obs.items():
        qc.save_expectation_value(permute_pauli(op, fl, n) if fl else op, list(range(n)), label=lbl)
    d = sim.run(qc).result().data()
    return {k: float(np.real(d[k])) for k in obs}


def noise_transform(circ: QuantumCircuit, rng, p2: float, p1: float) -> QuantumCircuit:
    """One Pauli trajectory (scripts/hw_cal_grids.py:113-130): after each 2q
    gate, with prob p2, a random Pauli (incl. I) on each of its qubits; after
    each 1q gate, with prob p1, a random non-identity Pauli."""
    out = QuantumCircuit(circ.num_qubits, circ.num_clbits)
    out.global_phase = circ.global_phase
    for inst in circ.data:
        qs = [circ.find_bit(b).index for b in inst.qubits]
        cs = [circ.find_bit(b).index for b in inst.clbits]
        out.append(inst.operation, qs, cs)
        nn = inst.operation.num_qubits
        if inst.operation.name in ("measure", "barrier"):
            continue
        if nn == 2 and rng.random() < p2:
            for q in qs:
                k = rng.integers(0, 4)
                if k:
                    (out.x, out.y, out.z)[k - 1](q)
        elif nn == 1 and rng.random() < p1:
            (out.x, out.y, out.z)[rng.integers(0, 3)](qs[0])
    return out


# ------------------------------------------------------------------ ideal grids
def one_point_grid(card: dict, ns: int, center: int, times, dt: float = DT, cap: int = 512,
                   trunc: float = 1e-10, threads: int = 2, cache_dir=None, noise=None, seed: int = 0,
                   log=None) -> dict:
    """Family-independent one-point functions <J0_v(t)>, <J1_v(t)> and the
    t = 0 insertion one-points for both insertions (shared by the three
    families).  -> {'J0': (n_t, ns), 'J1': (n_t, ns), 'ins_j0', 'ins_j1', 'H'}."""
    lat = Lattice(ns)
    eta = card["couplings"]["eta"]
    big = lat.n_wires > STATEVECTOR_MAX_WIRES
    sim = make_simulator(lat.n_wires, cap=cap, trunc=trunc, threads=threads)
    init, perm = None, None
    if big:
        ps = prep_mps(card, lat, center, cache_dir, cap, trunc, threads=threads, log=log)
        init, perm, Hval = ps.mps, ps.perm, ps.H
    else:
        Hop = to_sparse_pauli_op(lat.n_wires, hamiltonian_terms(lat, *[card["couplings"][k] for k in ("m0", "g2", "eta")]))
        Hval = expectations(sim, prep_state_circuit(card, lat, center), {"H": Hop})["H"]
    noise_t = (np.random.default_rng(seed), *noise) if noise else None
    one_obs = one_point_observables(lat, eta)
    one_obs["ins_j0"] = _with_anc(lat, charge_terms(lat, center), "I")
    one_obs["ins_j1"] = _with_anc(lat, current_terms(lat, center, eta, exact_seam=True), "I")
    out = {"J0": np.empty((len(times), ns)), "J1": np.empty((len(times), ns)), "H": Hval}
    for i, t in enumerate(times):
        n = int(round(t / dt))
        qc = QuantumCircuit(lat.n_wires)
        if not big:
            qc.compose(prep_state_circuit(card, lat, center), inplace=True)
        if n:
            qc.compose(C.trotter_block(lat, n, t, *C.card_couplings(card)), range(lat.n_qubits), inplace=True)
        t0 = time.time()
        d = expectations(sim, qc, one_obs, init, noise_t, perm)
        out["J0"][i] = [d[f"J0_{v}"] for v in range(ns)]
        out["J1"][i] = [d[f"J1_{v}"] for v in range(ns)]
        if i == 0:
            out["ins_j0"], out["ins_j1"] = d["ins_j0"], d["ins_j1"]
        T._log(f"one-point t={t:3.1f}: <J0({center})> = {out['J0'][i, center]:+.5f}  ({time.time() - t0:.0f}s)", log)
    return out


def ideal_grid(card: dict, ns: int, center: int, family: str, times, dt: float = DT,
               mirror: bool = False, cap: int = 512, threads: int = 2, cache_dir=None,
               noise=None, seed: int = 0, log=None, trunc: float = 1e-10, one_pt: dict | None = None) -> dict:
    """Noiseless (or one Pauli-trajectory, ``noise`` = (p2, p1)) hardware-
    Trotterized grid for one insertion family in the scripts/hw_cal_grids.py
    key set."""
    lat = Lattice(ns)
    p2p1 = tuple(noise) if noise else None
    noise = (np.random.default_rng(seed), *p2p1) if p2p1 else None
    eta = card["couplings"]["eta"]
    kind, off = CP.FAMILY_GADGET[family]
    gc = center + off
    id_a = (-1) ** gc / 2 if kind == "J0" else 0.0
    c_a = C.GADGET_COEFF[kind] * (eta if kind != "J0" else 1.0)
    big = lat.n_wires > STATEVECTOR_MAX_WIRES
    sim = make_simulator(lat.n_wires, cap=cap, trunc=trunc, threads=threads)
    init, perm = None, None
    if big:
        ps = prep_mps(card, lat, center, cache_dir, cap, trunc, threads=threads, log=log)
        init, perm, Hval = ps.mps, ps.perm, ps.H
    else:
        Hop = to_sparse_pauli_op(lat.n_wires, hamiltonian_terms(lat, *[card["couplings"][k] for k in ("m0", "g2", "eta")]))
        Hval = expectations(sim, prep_state_circuit(card, lat, center), {"H": Hop})["H"]
    if one_pt is None:
        one_pt = one_point_grid(card, ns, center, times, dt, cap, trunc, threads, cache_dir, p2p1, seed, log)

    def tail(t, use_mirror=False):
        qc = QuantumCircuit(lat.n_wires)
        if not big:
            qc.compose(prep_state_circuit(card, lat, center), inplace=True)
        return qc

    obs = probe_observables(lat, ("J0", "J1"), eta)
    one_pt_J0, one_pt_J1 = one_pt["J0"], one_pt["J1"]
    if kind == "J0" and off:        # dither family: its own insertion one-point
        insert_1pt = expectations(sim, tail(0.0), {"ins": _with_anc(lat, charge_terms(lat, gc), "I")}, init, noise, perm)["ins"]
    else:
        insert_1pt = one_pt["ins_j0"] if kind == "J0" else one_pt["ins_j1"]
    rows = {k: [] for k in obs}
    mrows = {f"m_{k}": [] for k in obs}
    for i, t in enumerate(times):
        n = int(round(t / dt))
        t0 = time.time()
        qa = tail(t)
        qa.h(lat.ancilla)
        qa.compose(C.insertion_gadget(lat, kind, gc, "direct"), inplace=True)
        if n:
            qa.compose(C.trotter_block(lat, n, t, *C.card_couplings(card)), range(lat.n_qubits), inplace=True)
        d = expectations(sim, qa, obs, init, noise, perm)
        for k in obs:
            rows[k].append(d[k])
        msg = f"{family} t={t:3.1f}: <XB_{center}> = {rows[f'XB_{center}'][-1]:+.5f}  ({time.time() - t0:.0f}s)"
        if mirror:
            qm = tail(t)
            qm.h(lat.ancilla)
            qm.compose(C.insertion_gadget(lat, kind, gc, "direct"), inplace=True)
            if n:
                qm.compose(C.trotter_block(lat, n, MIRROR_EPS, *C.card_couplings(card)), range(lat.n_qubits), inplace=True)
            dm = expectations(sim, qm, obs, init, noise, perm)
            for k in obs:
                mrows[f"m_{k}"].append(dm[k])
            msg += f"  mirror {mrows[f'm_XB_{center}'][-1]:+.5f}"
        T._log(msg, log)
    if mirror:
        rows.update(mrows)
    return dict(times=np.array(times, dtype=float), probes=np.arange(ns), bonds=np.arange(ns),
                id_a=id_a, c_a=c_a, H_prep=Hval, one_pt=one_pt_J0, one_pt_J0=one_pt_J0,
                one_pt_J1=one_pt_J1, insert_1pt=insert_1pt, family=family,
                insert="j0" if kind == "J0" else "j1", m0=card["couplings"]["m0"],
                g2=card["couplings"]["g2"], eta=eta, ns=ns, center=center, dt=dt,
                k0=card["block"]["k0"], sigma_x=card["block"]["sigma_x"], state=card["name"],
                noise=(np.array([p2p1[0], p2p1[1], float(seed)]) if p2p1 else np.zeros(3)),
                **{k: np.array(v) for k, v in rows.items()})


def write_ideal_grids(card, ns, center, families, times, out_template, **kw) -> dict:
    """One-point functions are computed once and shared by all families."""
    paths = {}
    op_kw = {k: kw[k] for k in ("dt", "cap", "trunc", "threads", "cache_dir", "noise", "seed", "log") if k in kw}
    one_pt = one_point_grid(card, ns, center, times, **op_kw)
    for fam in families:
        g = ideal_grid(card, ns, center, fam, times, one_pt=one_pt, **kw)
        path = out_template.format(family=fam)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **g)
        paths[fam] = path
    return paths


# ------------------------------------------------------------------ check
def check(be, lat: Lattice, card: dict, emb: T.Embedding, ideal_template: str, times,
          families=("j0", "j1p1", "j1p2"), basis: str = "cz", tol: float = 5e-3, cap: int = 512,
          threads: int = 2, cache_dir=None, seed: int = T.SEED, mirrors: bool = True, log=None) -> dict:
    """Submission-path validation.  For each family: the ISA base (mapped back
    to the logical register) from |0> vs the ideal t = 0 row; then for every t
    the ISA block (mapped back through the base's final layout) from the
    cached prep state + logical gadget, physics vs the ideal row at t and
    mirror vs the t = 0 row.  Raises AssertionError above ``tol``.
    -> {(family, name): max |diff|}."""
    eta = card["couplings"]["eta"]
    big = lat.n_wires > STATEVECTOR_MAX_WIRES
    sim = make_simulator(lat.n_wires, cap=cap, threads=threads)
    obs = probe_observables(lat, ("J0", "J1"), eta)
    ps = prep_mps(card, lat, emb.center, cache_dir, cap, threads=threads, log=log) if big else None
    init, perm = (ps.mps, ps.perm) if big else (None, None)
    steps = tuple(sorted({int(round(t / DT)) for t in times if t > 0}))
    results, worst = {}, 0.0
    for fam in families:
        ideal = A.IdealGrid(ideal_template.format(family=fam))
        kind, off = CP.FAMILY_GADGET[fam]
        b = T.transpile_bundle(be, lat, card, emb, steps, basis, kind, seed, cache_dir, log=log,
                               gadget_center=emb.center + off if off else None)
        # base from |0>
        base_rel, wire = relabel_to_logical(b.base, emb.layout if emb.layout else b.base.layout.initial_index_layout(), lat.n_wires)
        fw = final_wires(range(lat.n_wires), b.base_layout, wire)
        mapped = {k: remap(op, fw, base_rel.num_qubits) for k, op in obs.items()}
        d = expectations(sim, base_rel, mapped, perm=perm)
        results[(fam, "base")] = _compare(d, ideal, 0.0, log, f"{fam} base(t=0)")
        # blocks from the prep state + logical gadget
        gad = QuantumCircuit(lat.n_wires)
        if not big:
            gad.compose(prep_state_circuit(card, lat, emb.center), inplace=True)
        gad.h(lat.ancilla)
        gad.compose(C.insertion_gadget(lat, kind, emb.center + off, "direct"), inplace=True)
        for t in times:
            if t == 0:
                continue
            n = int(round(t / DT))
            blk = b.blocks[n]
            for label, circ, t_ref in (("physics", blk["physics"], t), ("mirror", blk["mirror"], 0.0)):
                if label == "mirror" and not mirrors:
                    continue
                rel, wire = relabel_to_logical(circ, b.base_layout, lat.n_wires)
                full = QuantumCircuit(rel.num_qubits)
                full.compose(gad, range(lat.n_wires), inplace=True)
                full.compose(rel, inplace=True)
                fw = final_wires(range(lat.n_wires), blk["layout"], wire)
                mapped = {k: remap(op, fw, rel.num_qubits) for k, op in obs.items()}
                d = expectations(sim, full, mapped, init, perm=perm)
                results[(fam, f"{label} t={t:.1f}")] = _compare(d, ideal, t_ref, log, f"{fam} {label} t={t:.1f}")
        worst = max(worst, max(v for (f, _), v in results.items() if f == fam))
    bad = {k: v for k, v in results.items() if v > tol}
    if bad:
        raise AssertionError(f"check FAILED (tol {tol}): {bad}")
    T._log(f"check PASSED: {len(results)} circuits, worst |diff| {worst:.2e} < {tol}", log)
    return results


READOUT_GROUPS = {"J0(Z)": ("XB_", "YB_", "B_"), "T1(XYA/XYB)": ("XT1_", "YT1_", "T1_"),
                  "T2(XYA/XYB)": ("XT2_", "YT2_", "T2_"), "anc": ("X", "Y")}
LAST_GROUPS: dict = {}


def _compare(d: dict, ideal: A.IdealGrid, t: float, log, tag: str) -> float:
    """Max |simulated - ideal| over the observables present in the grid; the
    per-readout-group maxima are stored in LAST_GROUPS[tag] and logged."""
    row = ideal.row(t)
    diffs = {k: abs(v - float(ideal.z[k][row])) for k, v in d.items() if ideal.has(k)}
    worst = max(diffs, key=diffs.get)
    groups = {}
    for g, prefixes in READOUT_GROUPS.items():
        sel = {k: v for k, v in diffs.items() if (k in prefixes if g == "anc" else k.startswith(prefixes))}
        if sel:
            kw = max(sel, key=sel.get)
            groups[g] = (sel[kw], kw)
    LAST_GROUPS[tag] = groups
    T._log(f"[{tag}] vs ideal t={ideal.times[row]:.1f}: max |diff| {diffs[worst]:.2e} ({worst}), "
           f"{len(diffs)} observables; " + ", ".join(f"{g} {v:.1e} ({k})" for g, (v, k) in groups.items()), log)
    return diffs[worst]


# ------------------------------------------------------------------ rehearsal
def sample_bits(sim, circuit: QuantumCircuit, shots: int, initial_mps=None, noise=None,
                seed: int | None = None, n_traj: int = 8, perm=None) -> np.ndarray:
    """Sample a measured circuit -> uint8 (shots, n_clbits), column i = clbit i.

    ``noise`` = (rng, p2, p1): the shots are split into ``n_traj`` batches and
    each batch gets its own Pauli trajectory (a fixed Pauli frame is a
    deterministic sign flip, not damping; averaging trajectories over batches
    is the stochastic channel the reference dry runs obtain by re-running
    with different seeds)."""
    n = circuit.num_qubits
    batches = [int(shots)] if noise is None else \
        [int(shots) // n_traj + (1 if k < int(shots) % n_traj else 0) for k in range(n_traj)]
    rows = []
    rng = np.random.default_rng(seed)
    clean = None
    for k, nb in enumerate(batches):
        if nb <= 0:
            continue
        qc = QuantumCircuit(n, circuit.num_clbits)
        if initial_mps is not None:
            qc.set_matrix_product_state(pad_mps(initial_mps, n - len(initial_mps[0])))
        body = noise_transform(circuit, *noise) if noise is not None else circuit
        if perm is not None:
            body = route_to_chain(body, perm, n)[0]          # measurements follow their wires
        else:
            body = _aer_ready(body)
        qc.compose(body, inplace=True)
        mem = sim.run(qc, shots=nb, memory=True, seed_simulator=int(rng.integers(2**31))).result().get_memory()
        rows.append(np.array([[int(ch) for ch in s[::-1]] for s in mem], dtype=np.uint8))
    return np.concatenate(rows, axis=0)


def relabel_pub(isa_pub: QuantumCircuit, initial_layout, n_wires: int):
    """Measured ISA pub -> logical-register circuit with the same clbits."""
    wire = {int(p): i for i, p in enumerate(initial_layout)}
    touched = sorted({isa_pub.find_bit(q).index for inst in isa_pub.data for q in inst.qubits})
    for p in touched:
        if p not in wire:
            wire[p] = len(wire)
    out = QuantumCircuit(max(n_wires, len(wire)), isa_pub.num_clbits)
    out.global_phase = isa_pub.global_phase
    for inst in isa_pub.data:
        out.append(inst.operation, [wire[isa_pub.find_bit(q).index] for q in inst.qubits],
                   [isa_pub.find_bit(c).index for c in inst.clbits])
    return out


def rehearse(be, lat: Lattice, card: dict, emb: T.Embedding, specs, shots: dict, ideal_template: str,
             out_dir: str, basis: str = "cz", noise=None, seed: int = 0, cap: int = 512, threads: int = 2,
             cache_dir=None, tag: str = "rehearsal", components=A.COMPONENTS, n_traj: int = 8,
             log=None) -> dict:
    """Sample every pub (Aer, from the cached prep MPS when large), write
    fetch-format bits + metadata, run analyze -> slice files.
    ``noise`` = (p2, p1) enables one Pauli trajectory per pub.
    -> {'bits': path, 'meta': path, 'slices': {(comp, t): path}}."""
    pubs, info, bundles = CP.build_pub_circuits(be, lat, card, emb, specs, basis, cache_dir=cache_dir, log=log)
    big = lat.n_wires > STATEVECTOR_MAX_WIRES
    sim = make_simulator(lat.n_wires, cap=cap, threads=threads)
    rng = np.random.default_rng(seed)
    noise_t = (rng, float(noise[0]), float(noise[1])) if noise else None
    init_by_fam = {}
    bits = {}
    t0 = time.time()
    for s in specs:
        qc = pubs[s.name]
        fam = "j0" if s.mirror else s.family
        bnd = bundles[(card["name"], fam)]
        if big:
            # split: cached prep state + logical gadget, then the ISA block + readout
            if fam not in init_by_fam:
                kind, off = CP.FAMILY_GADGET[fam]
                ps = prep_mps(card, lat, emb.center, cache_dir, cap, threads=threads, log=log)
                gad = QuantumCircuit(lat.n_wires)
                gad.h(lat.ancilla)
                gad.compose(C.insertion_gadget(lat, kind, emb.center + off, "direct"), inplace=True)
                q0 = QuantumCircuit(lat.n_wires)
                q0.set_matrix_product_state(ps.mps)
                routed, fl_g = route_to_chain(gad, ps.perm, lat.n_wires)
                q0.compose(routed, inplace=True)
                q0.save_matrix_product_state(label="mps")
                init_by_fam[fam] = (sim.run(q0).result().data()["mps"], fl_g)
            n = s.n_steps
            if n == 0:
                body = C.readout_layer(QuantumCircuit(be.num_qubits), lat, bnd.base_layout, s.readout, s.anc_basis)
            else:
                blk = bnd.blocks[n]["mirror" if s.mirror else "physics"] if s.dt == DT else \
                    T.assign_block(bnd, n, n * s.dt)[1 if s.mirror else 0]
                body = C.readout_layer(blk, lat, bnd.blocks[n]["layout"], s.readout, s.anc_basis)
            rel = relabel_pub(body, bnd.base_layout, lat.n_wires)
            mps_f, perm_f = init_by_fam[fam]
            arr = sample_bits(sim, rel, shots[s.name], mps_f, noise_t,
                              seed=int(rng.integers(2**31)), n_traj=n_traj, perm=perm_f)
        else:
            rel = relabel_pub(qc, emb.layout if emb.layout else list(qc.layout.initial_index_layout()), lat.n_wires)
            arr = sample_bits(sim, rel, shots[s.name], None, noise_t, seed=int(rng.integers(2**31)),
                              n_traj=n_traj)
        bits[s.name] = arr
        T._log(f"sampled {s.name}: {arr.shape} ({time.time() - t0:.0f}s)", log)
    names = [s.name for s in specs]
    meta = CP.sim_meta(lat, emb, basis, names, shots, info, backend=f"{tag}:{T.backend_label(be)}")
    os.makedirs(out_dir, exist_ok=True)
    bits_path = os.path.join(out_dir, f"htq_bits_{meta['job_id']}.npz")
    np.savez_compressed(bits_path, job_id=meta["job_id"], backend=meta["backend"],
                        pub_names=np.array(names), **bits)
    meta_path = os.path.join(out_dir, f"htq_job_{meta['job_id']}.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    slices = A.analyze([bits_path], ideal_template, os.path.join(out_dir, "slice_{comp}_t{t:.1f}.npz"),
                       lat.ns, emb.center, components=components, eta=card["couplings"]["eta"],
                       backend=meta["backend"], log=log)
    return {"bits": bits_path, "meta": meta_path, "slices": slices, "job_id": meta["job_id"]}
