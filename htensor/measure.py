"""Measurement protocols for C^{mu nu}(x,t) = <psi| J^mu(x,t) J^nu(y,0) |psi>.

Three protocols (PLAN.md sec. 4):
  1. hadamard  -- single-ancilla Hadamard test, controlled-Pauli insertion of
                  J^nu at t=0, all commuting J^mu(x) probes read from the same
                  state. PRIMARY protocol.
  2. mf        -- Mitarai-Fujii ancilla-free estimator (eigen-branch
                  projection for Re, e^{-+i pi/4 P} rotations for Im).
                  Hardware cross-check.
  3. lly       -- Lamm-Lawrence-Yamauchi second-order numerical
                  differentiation of linear-response evolution; recovers
                  2 Re C_connected. Validation / historical baseline.

This module provides statevector-exact runners (validation and noiseless
simulator production); shot-based execution lives in the backend layer.

Assembly convention: with J = c_id * I + sum_a c_a P_a (all c real),
  C(x,t) = id_mu*id_nu + id_mu*<A_P>_0 + id_nu*<B_P(t)> + sum_ab c_b c_a <P_b(t) P_a>.
Single-current expectations <B_P(t)>, <A_P>_0 are returned alongside C so the
analysis layer can form connected/vacuum-subtracted combinations pointwise.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from .pauli import pauli_term


# --------------------------------------------------------------- decomposition
def split_current(op: SparsePauliOp) -> tuple[float, list[tuple[dict, float]]]:
    """SparsePauliOp -> (identity coefficient, [({qubit: pauli}, coeff), ...]).

    Coefficients must be real (Hermitian Pauli decomposition)."""
    op = op.simplify()
    id_c, terms = 0.0, []
    n = op.num_qubits
    for label, coeff in zip(op.paulis.to_labels(), op.coeffs):
        c = complex(coeff)
        if abs(c.imag) > 1e-12:
            raise ValueError(f"non-real coefficient {c} for {label}")
        ops = {n - 1 - i: ch for i, ch in enumerate(label) if ch != "I"}
        if ops:
            terms.append((ops, c.real))
        else:
            id_c += c.real
    return id_c, terms


def pauli_matrix(n_qubits: int, ops: dict) -> np.ndarray:
    return pauli_term(n_qubits, ops, 1.0).to_matrix(sparse=True).tocsr()


def _with_ancilla(op: SparsePauliOp, anc_pauli: str) -> SparsePauliOp:
    """Extend a system operator with an ancilla Pauli on a new top qubit."""
    labels = [anc_pauli + l for l in op.paulis.to_labels()]
    return SparsePauliOp(labels, op.coeffs)


def _pauli_only(op: SparsePauliOp) -> SparsePauliOp:
    """Strip the identity component."""
    id_c, terms = split_current(op)
    n = op.num_qubits
    if not terms:
        return SparsePauliOp("I" * n, coeffs=[0.0])
    return sum(pauli_term(n, ops, c) for ops, c in terms).simplify()


# --------------------------------------------------------------- circuit gadgets
def controlled_pauli(qc: QuantumCircuit, anc: int, ops: dict):
    for q, p in sorted(ops.items()):
        getattr(qc, "c" + p.lower())(anc, q)


_BASIS_PRE = {"Z": (), "X": ("h",), "Y": ("sdg", "h")}     # P -> Z
_BASIS_POST = {"Z": (), "X": ("h",), "Y": ("h", "s")}      # Z -> P


def parity_ladder_controlled_pauli(qc: QuantumCircuit, anc: int, ops: dict,
                                   path: list[tuple[int, int]]):
    """Controlled-P for a weight-w Pauli string P = prod_q P_q using ONE
    ancilla two-qubit gate: rotate every P_q to Z, accumulate the parity of
    the string onto a hub qubit with a CX ladder along `path` (list of
    (control, target) pairs; the last target is the hub, which must be the
    only qubit adjacent to the ancilla), apply CZ(anc, hub), uncompute.
    Unitarily equal to controlled_pauli(qc, anc, ops) (tests/test_m5_*).
    Two-qubit cost 2 len(path) + 1, all CX on `path` edges -- nearest
    neighbour on a ring/ladder embedding, unlike the naive controlled
    string, which needs the ancilla adjacent to every qubit of P."""
    qubits = set(ops)
    touched = {q for pair in path for q in pair}
    if path:
        hub = path[-1][1]
        if touched != qubits:
            raise ValueError(f"path qubits {sorted(touched)} != string {sorted(qubits)}")
    else:
        (hub,) = qubits
    for q, p in sorted(ops.items()):
        for g in _BASIS_PRE[p]:
            getattr(qc, g)(q)
    for c, t in path:
        qc.cx(c, t)
    qc.cz(anc, hub)
    for c, t in reversed(path):
        qc.cx(c, t)
    for q, p in sorted(ops.items()):
        for g in _BASIS_POST[p]:
            getattr(qc, g)(q)


def j1_ladder_path(ops: dict, accumulate: str = "site") -> list[tuple[int, int]]:
    """CX path for a bulk J^1 Pauli term {a: X/Y, l: Z, b: Y/X} (a < l < b,
    consecutive ring qubits).  accumulate = 'site' (hub a; ring/heavy-hex,
    ancilla next to the left site -- same ancilla position as the J^0 CZ),
    'link' (hub l), or 'pendant' (square-lattice ladder embedding where the
    link qubit is pendant to the RIGHT site b: edges (a,b) and (l,b); hub a)."""
    if len(ops) != 3:
        raise ValueError("bulk J^1 term has weight 3 (seam bond carries a JW string)")
    a, l, b = sorted(ops)
    if accumulate == "site":
        return [(b, l), (l, a)]
    if accumulate == "link":
        return [(a, l), (b, l)]
    if accumulate == "pendant":
        return [(l, b), (b, a)]
    raise ValueError(accumulate)


def j1_gadget(qc: QuantumCircuit, anc: int, ops: dict, accumulate: str = "site"):
    """Controlled J^1 Pauli term via the parity ladder (5 two-qubit gates)."""
    parity_ladder_controlled_pauli(qc, anc, ops, j1_ladder_path(ops, accumulate))


def make_ladder_gadget(accumulate: str = "site"):
    """-> gadget(qc, anc, ops) usable as `hadamard_test_circuit(..., gadget=)`:
    weight-1 strings fall back to the single controlled Pauli."""
    def gadget(qc, anc, ops):
        if len(ops) == 1:
            controlled_pauli(qc, anc, ops)
        else:
            j1_gadget(qc, anc, ops, accumulate)
    return gadget


def hadamard_test_circuit(n_sys: int, insertion: dict, evolution,
                          gadget=None) -> QuantumCircuit:
    """Ancilla (qubit n_sys) in |+>, controlled Pauli string, then evolution.
    Measure X_anc (x) M for Re<M(t) P>, Y_anc (x) M for Im<M(t) P>.
    `gadget(qc, anc, ops)` overrides the naive controlled_pauli insertion
    (e.g. make_ladder_gadget for hardware-shaped J^1 insertions)."""
    qc = QuantumCircuit(n_sys + 1)
    anc = n_sys
    qc.h(anc)
    (gadget or controlled_pauli)(qc, anc, insertion)
    qc.append(evolution, range(n_sys))
    return qc


# --------------------------------------------------------------- result record
@dataclass
class CorrelatorData:
    """Raw protocol output on a (time, probe) grid."""

    times: np.ndarray
    correlator: np.ndarray        # C[t_i, probe_j], complex
    probe_expect: np.ndarray      # <B_P^j(t_i)> + id_b^j  (full single-current)
    insert_expect: complex        # <A(0)> (full single-current, t = 0)

    @property
    def connected(self) -> np.ndarray:
        """C(t, x) - <B(t, x)><A(0)> (disconnected product subtracted pointwise)."""
        return self.correlator - self.probe_expect * self.insert_expect


# --------------------------------------------------------------- runners
def _initial_sv(psi0: np.ndarray, extra_qubits: int = 0) -> Statevector:
    full = psi0
    for _ in range(extra_qubits):
        full = np.kron([1.0, 0.0], full)  # new qubit on top, in |0>
    return Statevector(full)


def _plain_expectations(psi0, n_sys, ops: list[SparsePauliOp], evolution_factory, times):
    """<op(t)> for each op, via plain (no-ancilla) evolution."""
    out = np.empty((len(times), len(ops)), dtype=complex)
    for i, t in enumerate(times):
        qc = QuantumCircuit(n_sys)
        qc.append(evolution_factory(t), range(n_sys))
        sv = _initial_sv(psi0).evolve(qc)
        for j, op in enumerate(ops):
            out[i, j] = sv.expectation_value(op)
    return out


def hadamard_correlator_sv(psi0: np.ndarray, insert_op: SparsePauliOp,
                           probe_ops: list[SparsePauliOp], evolution_factory,
                           times, gadget=None) -> CorrelatorData:
    """Statevector-exact Hadamard-test protocol (`gadget` as in
    hadamard_test_circuit)."""
    times = np.asarray(times, dtype=float)
    n_sys = insert_op.num_qubits
    id_a, terms_a = split_current(insert_op)
    probes_p = [_pauli_only(op) for op in probe_ops]

    # single-current pieces (also the disconnected-subtraction inputs)
    probe_expect = _plain_expectations(psi0, n_sys, probe_ops, evolution_factory, times)
    insert_expect = complex(_initial_sv(psi0).expectation_value(insert_op))

    corr = np.zeros((len(times), len(probe_ops)), dtype=complex)
    probe_p_expect = probe_expect - np.array(
        [complex(split_current(op)[0]) for op in probe_ops]
    )[None, :]

    for ops_a, c_a in terms_a:
        for i, t in enumerate(times):
            qc = hadamard_test_circuit(n_sys, ops_a, evolution_factory(t), gadget)
            sv = _initial_sv(psi0, extra_qubits=1).evolve(qc)
            for j, bp in enumerate(probes_p):
                re = sv.expectation_value(_with_ancilla(bp, "X"))
                im = sv.expectation_value(_with_ancilla(bp, "Y"))
                corr[i, j] += c_a * (complex(re).real + 1j * complex(im).real)

    # identity cross terms
    for j, op in enumerate(probe_ops):
        id_b, _ = split_current(op)
        corr[:, j] += id_a * probe_p_expect[:, j] + id_b * (insert_expect - id_a) + id_a * id_b

    return CorrelatorData(times, corr, probe_expect, insert_expect)


def mf_correlator_sv(psi0: np.ndarray, insert_op: SparsePauliOp,
                     probe_ops: list[SparsePauliOp], evolution_factory,
                     times) -> CorrelatorData:
    """Mitarai-Fujii ancilla-free estimator, statevector-exact.

    Re<B(t)P_a> from sign-weighted eigen-branch states Pi_+- |psi>;
    Im<B(t)P_a> = (1/2)(<chi_-|B(t)|chi_-> - <chi_+|B(t)|chi_+>),
    chi_-+ = e^{-+i pi/4 P_a}|psi>.  All branch states are circuit-preparable
    (basis rotation + mid-circuit measurement / one Pauli rotation).
    """
    times = np.asarray(times, dtype=float)
    n_sys = insert_op.num_qubits
    id_a, terms_a = split_current(insert_op)
    probes_p = [_pauli_only(op) for op in probe_ops]

    probe_expect = _plain_expectations(psi0, n_sys, probe_ops, evolution_factory, times)
    insert_expect = complex(_initial_sv(psi0).expectation_value(insert_op))
    probe_p_expect = probe_expect - np.array(
        [complex(split_current(op)[0]) for op in probe_ops]
    )[None, :]

    corr = np.zeros((len(times), len(probe_ops)), dtype=complex)
    for ops_a, c_a in terms_a:
        P = pauli_matrix(n_sys, ops_a)
        branches = {
            "plus": 0.5 * (psi0 + P @ psi0),   # Pi_+ |psi>, unnormalized
            "minus": 0.5 * (psi0 - P @ psi0),  # Pi_- |psi>
            "chi_m": (psi0 - 1j * (P @ psi0)) / np.sqrt(2.0),  # e^{-i pi/4 P}|psi>
            "chi_p": (psi0 + 1j * (P @ psi0)) / np.sqrt(2.0),  # e^{+i pi/4 P}|psi>
        }
        for i, t in enumerate(times):
            qc = QuantumCircuit(n_sys)
            qc.append(evolution_factory(t), range(n_sys))
            evolved = {k: Statevector(v).evolve(qc) for k, v in branches.items()}
            for j, bp in enumerate(probes_p):
                re = (evolved["plus"].expectation_value(bp)
                      - evolved["minus"].expectation_value(bp))
                im = 0.5 * (evolved["chi_m"].expectation_value(bp)
                            - evolved["chi_p"].expectation_value(bp))
                corr[i, j] += c_a * (complex(re).real + 1j * complex(im).real)

    for j, op in enumerate(probe_ops):
        id_b, _ = split_current(op)
        corr[:, j] += id_a * probe_p_expect[:, j] + id_b * (insert_expect - id_a) + id_a * id_b

    return CorrelatorData(times, corr, probe_expect, insert_expect)


def lly_connected_re(psi0: np.ndarray, H_sparse, later_op: SparsePauliOp,
                     earlier_op: SparsePauliOp, t: float, eps: float) -> float:
    """LLY 4-point stencil on the return probability
        F(ex, e0) = |<psi| e^{iHt} e^{i ex A} e^{-iHt} e^{-i e0 B} |psi>|^2,
    A = later_op, B = earlier_op.  Mixed second derivative gives
    2 Re[<A(t)B> - <A(t)><B>]; returns Re C_connected (stencil / 2).
    Bias O(eps^2); under shot noise the variance scales as 1/eps^4 -- the
    reason this protocol is validation-only (PLAN.md sec. 4).
    """
    A = later_op.to_matrix(sparse=True).tocsc()
    B = earlier_op.to_matrix(sparse=True).tocsc()
    Hc = H_sparse.tocsc()

    def overlap(ex, e0):
        ket = spla.expm_multiply(-1j * e0 * B, psi0)
        ket = spla.expm_multiply(-1j * t * Hc, ket)
        ket = spla.expm_multiply(1j * ex * A, ket)
        ket = spla.expm_multiply(1j * t * Hc, ket)
        return abs(np.vdot(psi0, ket)) ** 2

    stencil = (overlap(eps, eps) - overlap(eps, -eps)
               - overlap(-eps, eps) + overlap(-eps, -eps)) / (4 * eps**2)
    return stencil / 2.0
