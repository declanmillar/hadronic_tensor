"""Second-order (palindromic) Trotter circuits with the PBC seam parity trick.

One step of length dt:
    U1(dt/2) U2(dt/2) U3(dt) U2(dt/2) U1(dt/2)
with U1 = single-qubit mass+gauge rotations, U2/U3 = even/odd-bond hop layers.
(The three groups do NOT mutually commute -- each is only internally
commuting -- so the palindrome is required for genuine second order.)

The PBC seam hop is implemented as the ordinary 3-qubit bulk gate with
coupling eta -> (-1)^(ns/2 + 1) eta.  This is exact on Gauss-law (physical)
states, where the Jordan-Wigner string equals the c-number fermion parity;
circuits built here are therefore only valid on physical initial states.

Hop gate, all <=2-qubit operations (MPS- and hardware-safe):
    exp(-i theta (X_a X_b + Y_a Y_b) Z_l)
      = CZ(l,b) . RXX(2 theta)(a,b) . RYY(2 theta)(a,b) . CZ(l,b)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate

from .lattice import Z2Lattice
from . import hamiltonian as ham


def seam_sign(lat: Z2Lattice) -> int:
    """Fermion-parity replacement sign multiplying eta on the seam bond."""
    return (-1) ** (lat.ns // 2 + 1)


def strong_coupling_vacuum_circuit(lat: Z2Lattice,
                                   link_ref: str = "+") -> QuantumCircuit:
    """|0> (even matter), |1> (odd matter), |+> (links): Clifford, physical.

    link_ref: "+" (default, the g2 -> +infinity-favoured sigma^x = +1 state
    used by all production runs) or "-" (links |->, sigma^x = -1).  Both
    satisfy G_n = +1 (each site sees two links, so the sign squares away);
    at large m0 / small g2 (e.g. the dhk couplings (1.0, 0.6, 1.0)) the
    interacting vacuum sits closer to the |-> reference and the variational
    ansatz converges only from there (qPDF campaign, 2026-08)."""
    if link_ref not in ("+", "-"):
        raise ValueError(f"link_ref must be '+' or '-', got {link_ref!r}")
    qc = QuantumCircuit(lat.n_qubits)
    for n in range(1, lat.ns, 2):
        qc.x(lat.site_qubit(n))
    for q in lat.link_qubits:
        qc.h(q)
        if link_ref == "-":
            qc.z(q)
    return qc


def _single_qubit_layer(qc: QuantumCircuit, lat: Z2Lattice, m0, g2, dt):
    for n in range(lat.ns):
        qc.rz(-m0 * (-1) ** n * dt, lat.site_qubit(n))
    for q in lat.link_qubits:
        qc.rx(g2 * dt, q)


def _hop(qc: QuantumCircuit, lat: Z2Lattice, bond: int, theta: float):
    qa, qb = lat.site_qubit(bond), lat.site_qubit(bond + 1)
    ql = lat.link_qubit(bond)
    qc.cz(ql, qb)
    qc.rxx(2 * theta, qa, qb)
    qc.ryy(2 * theta, qa, qb)
    qc.cz(ql, qb)


def _hop_layer(qc: QuantumCircuit, lat: Z2Lattice, eta, dt, parity: int):
    for b in lat.bonds:
        if b % 2 != parity:
            continue
        s = seam_sign(lat) if lat.is_seam(b) else 1
        _hop(qc, lat, b, s * eta * dt / 4)


def trotter_step(lat: Z2Lattice, m0, g2, eta, dt) -> QuantumCircuit:
    qc = QuantumCircuit(lat.n_qubits)
    _single_qubit_layer(qc, lat, m0, g2, dt / 2)
    _hop_layer(qc, lat, eta, dt / 2, 0)
    _hop_layer(qc, lat, eta, dt, 1)
    _hop_layer(qc, lat, eta, dt / 2, 0)
    _single_qubit_layer(qc, lat, m0, g2, dt / 2)
    return qc


def trotter_circuit(lat: Z2Lattice, m0, g2, eta, t: float, n_steps: int) -> QuantumCircuit:
    qc = QuantumCircuit(lat.n_qubits)
    if n_steps == 0 or t == 0.0:
        return qc
    step = trotter_step(lat, m0, g2, eta, t / n_steps)
    for _ in range(n_steps):
        qc.compose(step, inplace=True)
    return qc


def parametric_step(lat: Z2Lattice, m0, g2, eta, name: str = "t"):
    """One second-order step with a SYMBOLIC step size (qiskit Parameter).

    Hardware recipe (2026-09): transpile this once at optimization_level=3,
    then assign_parameters({t: dt}) for the physics circuit and
    ({t: MIRROR_EPS}) for the depth-matched mirror -- the two share an
    identical two-qubit skeleton by construction (transpiling the
    near-identity mirror numerically at O3 would delete its gates; O1 keeps
    them but costs ~30% more two-qubit gates).  -> (circuit, parameter)."""
    from qiskit.circuit import Parameter
    p = Parameter(name)
    return trotter_step(lat, m0, g2, eta, p), p


def exact_evolution_gate(lat: Z2Lattice, m0, g2, eta, t: float) -> HamiltonianGate:
    """Dense exp(-iHt) as a gate, for protocol validation at small volume
    (isolates measurement-protocol errors from Trotter error)."""
    if lat.n_qubits > 12:
        raise ValueError("exact_evolution_gate is for small-volume validation only")
    H = ham.build_hamiltonian(lat, m0, g2, eta).to_matrix()
    return HamiltonianGate(H, time=t)


def make_evolution_factory(lat: Z2Lattice, m0, g2, eta, kind: str = "trotter",
                           dt_target: float = 0.05):
    """t -> Instruction implementing (approx) exp(-iHt) on the system qubits."""
    if kind == "exact":
        return lambda t: exact_evolution_gate(lat, m0, g2, eta, t) \
            if t != 0.0 else QuantumCircuit(lat.n_qubits).to_instruction()
    if kind == "trotter":
        def factory(t):
            n = max(1, int(np.ceil(abs(t) / dt_target))) if t != 0.0 else 0
            return trotter_circuit(lat, m0, g2, eta, t, n).to_instruction()
        return factory
    raise ValueError(kind)
