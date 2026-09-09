"""Exact-diagonalization reference machinery (small volumes, sparse matrices).

Ground truth for unit tests and for validating Trotter circuits, state
preparation, and the correlator/W pipeline at ns <= ~8 (16 qubits).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qiskit.quantum_info import SparsePauliOp

from .lattice import Z2Lattice
from . import hamiltonian as ham


def to_sparse(op: SparsePauliOp) -> sp.csr_matrix:
    return op.to_matrix(sparse=True).tocsr()


# ---------------------------------------------------------------- matrix-free
def _pauli_terms(op: SparsePauliOp):
    """(flip mask, sign mask, coeff*i^nY) per term; diagonal terms merged
    separately.  P|i> = i^{nY} (-1)^{popcount(i & mz)} |i ^ mx>."""
    n = op.num_qubits
    diag_terms, offdiag = [], []
    for lab, c in zip(op.paulis.to_labels(), op.coeffs):
        mx = mz = 0
        ph = complex(c)
        for q in range(n):
            ch = lab[n - 1 - q]
            if ch in "XY":
                mx |= 1 << q
            if ch in "ZY":
                mz |= 1 << q
            if ch == "Y":
                ph *= 1j
        (diag_terms if mx == 0 else offdiag).append((mx, mz, ph))
    return n, diag_terms, offdiag


def apply_pauli_sum(op: SparsePauliOp, v: np.ndarray) -> np.ndarray:
    """op @ v without building a matrix (any width that fits a statevector)."""
    n, diag_terms, offdiag = _pauli_terms(op)
    idx = np.arange(1 << n, dtype=np.uint64)
    out = np.zeros(1 << n, dtype=complex)
    for mx, mz, ph in diag_terms:
        signs = 1.0 - 2.0 * (np.bitwise_count(idx & np.uint64(mz)) % 2)
        out += ph * signs * v
    for mx, mz, ph in offdiag:
        src = idx ^ np.uint64(mx)
        signs = 1.0 - 2.0 * (np.bitwise_count(src & np.uint64(mz)) % 2)
        out += ph * signs * v[src]
    return out


def pauli_linear_operator(op: SparsePauliOp) -> spla.LinearOperator:
    """Matrix-free LinearOperator for eigsh at 20-26 qubits.  The (usually
    numerous, e.g. Q^2 penalty) diagonal terms are precomputed into a single
    diagonal vector."""
    n, diag_terms, offdiag = _pauli_terms(op)
    N = 1 << n
    idx = np.arange(N, dtype=np.uint64)
    diag = np.zeros(N, dtype=complex)
    for mx, mz, ph in diag_terms:
        diag += ph * (1.0 - 2.0 * (np.bitwise_count(idx & np.uint64(mz)) % 2))
    offdiag = [(np.uint64(mx), np.uint64(mz), ph) for mx, mz, ph in offdiag]

    def matvec(v):
        v = np.asarray(v).ravel()
        out = diag * v
        for mx, mz, ph in offdiag:
            src = idx ^ mx
            signs = 1.0 - 2.0 * (np.bitwise_count(src & mz) % 2)
            out += ph * signs * v[src]
        return out

    return spla.LinearOperator((N, N), matvec=matvec, dtype=complex)


def strong_coupling_vacuum(lat: Z2Lattice, link_ref: str = "+") -> np.ndarray:
    """Product state: even matter sites |0>, odd |1>, links |+> (or |-> for
    link_ref="-"; see trotter.strong_coupling_vacuum_circuit).

    Satisfies G_n = +1 for all n and Q = 0; the m0 -> infinity vacuum and the
    reference state for adiabatic/variational preparation.
    """
    if link_ref not in ("+", "-"):
        raise ValueError(f"link_ref must be '+' or '-', got {link_ref!r}")
    zero = np.array([1.0, 0.0])
    one = np.array([0.0, 1.0])
    plus = np.array([1.0, 1.0 if link_ref == "+" else -1.0]) / np.sqrt(2)
    single = {}
    for n in range(lat.ns):
        single[lat.site_qubit(n)] = zero if n % 2 == 0 else one
    for q in lat.link_qubits:
        single[q] = plus
    psi = np.array([1.0])
    for q in reversed(range(lat.n_qubits)):  # accumulate from highest qubit down
        psi = np.kron(psi, single[q])
    return psi


def penalized_hamiltonian(
    lat: Z2Lattice, m0: float, g2: float = 1.0, eta: float = 1.0,
    lam_gauss: float = 10.0, lam_charge: float = 10.0,
) -> SparsePauliOp:
    """H - lam_gauss * sum_n G_n + lam_charge * Q^2: pushes the physical
    (all G_n = +1), charge-zero sector to the bottom of the spectrum."""
    H = ham.build_hamiltonian(lat, m0, g2, eta)
    G = sum(ham.gauss_operator(lat, n) for n in range(lat.ns)).simplify()
    Q = ham.total_charge(lat)
    return (H - lam_gauss * G + lam_charge * (Q @ Q)).simplify()


def lowest_physical_states(
    lat: Z2Lattice, m0: float, g2: float = 1.0, eta: float = 1.0, k: int = 4,
    matrix_free: bool = False, ncv: int | None = None,
    maxiter: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(energies, states) of the k lowest eigenstates of H within the
    physical, Q=0 sector. Energies are of H itself (penalties subtracted by
    evaluating <H> on the penalized eigenvectors).

    matrix_free=True avoids building sparse matrices (mandatory above ~22
    qubits); ncv caps the Lanczos basis to bound RAM (ncv * 16 B * 2^n)."""
    Hp_op = penalized_hamiltonian(lat, m0, g2, eta)
    H_op = ham.build_hamiltonian(lat, m0, g2, eta)
    v0 = strong_coupling_vacuum(lat)
    if maxiter is None:
        # scipy's default (10*n) overflows ARPACK's int32 above ~2^27
        # dimensions (ARPACK error -4); cap explicitly
        maxiter = int(min(10 * (1 << lat.n_qubits), 100_000))
    if matrix_free:
        Hp = pauli_linear_operator(Hp_op)
        _, vecs = spla.eigsh(Hp, k=k, which="SA", v0=v0, ncv=ncv,
                             maxiter=maxiter)
        energies = np.array([np.real(np.vdot(v, apply_pauli_sum(H_op, v)))
                             for v in vecs.T])
    else:
        Hp = to_sparse(Hp_op)
        H = to_sparse(H_op)
        _, vecs = spla.eigsh(Hp, k=k, which="SA", v0=v0, ncv=ncv)
        energies = np.array([np.real(np.vdot(v, H @ v)) for v in vecs.T])
    order = np.argsort(energies)
    return energies[order], vecs[:, order]


def evolve(H: sp.spmatrix, psi: np.ndarray, t: float) -> np.ndarray:
    return spla.expm_multiply(-1j * t * H.tocsc(), psi)


def two_current_correlator(
    H: sp.spmatrix, op_later: sp.spmatrix, op_earlier: sp.spmatrix,
    psi: np.ndarray, times: np.ndarray,
) -> np.ndarray:
    """C(t) = <psi| e^{iHt} B e^{-iHt} A |psi> for B = op_later, A = op_earlier.

    Evolves |psi> and A|psi> stepwise over the (uniformly spaced or not)
    time grid; exact up to expm_multiply tolerance.
    """
    Hc = H.tocsc()
    ket = op_earlier @ psi
    bra = psi.copy()
    out = np.empty(len(times), dtype=complex)
    t_prev = 0.0
    for i, t in enumerate(times):
        dt = t - t_prev
        if dt != 0.0:
            ket = spla.expm_multiply(-1j * dt * Hc, ket)
            bra = spla.expm_multiply(-1j * dt * Hc, bra)
        out[i] = np.vdot(bra, op_later @ ket)
        t_prev = t
    return out


def expectation(op: sp.spmatrix | SparsePauliOp, psi: np.ndarray) -> complex:
    if isinstance(op, SparsePauliOp):
        op = to_sparse(op)
    return complex(np.vdot(psi, op @ psi))
