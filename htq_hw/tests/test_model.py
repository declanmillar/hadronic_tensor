"""Term tables, seam sign, Gauss law, Ward identity (qiskit only)."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector

from htq_hw import model as M
from htq_hw import circuits as C

M0, G2, ETA = 0.7, 1.1, 1.3


def _op(lat, terms):
    return M.to_sparse_pauli_op(lat.n_qubits, terms)


def _gauss_projector(lat):
    n = lat.n_qubits
    P = np.eye(2 ** n)
    for m in range(lat.ns):
        G = _op(lat, M.gauss_terms(lat, m)).to_matrix()
        P = ((np.eye(2 ** n) + G) / 2) @ P
    return P


def test_lattice_indexing():
    lat = M.Lattice(6)
    assert lat.n_qubits == 12 and lat.ancilla == 12 and lat.n_wires == 13
    assert [lat.site_qubit(n) for n in range(6)] == [0, 2, 4, 6, 8, 10]
    assert [lat.link_qubit(n) for n in range(6)] == [1, 3, 5, 7, 9, 11]
    assert lat.site_qubit(6) == 0 and lat.link_qubit(-1) == 11
    assert lat.is_seam(5) and not lat.is_seam(4)
    assert lat.seam_string_qubits() == [2, 4, 6, 8]
    assert lat.bond_qubits(5) == (10, 0, 11)


@pytest.mark.parametrize("ns", [4, 6, 8, 10, 50])
def test_seam_sign(ns):
    assert M.Lattice(ns).seam_sign == (-1) ** (ns // 2 + 1)


def test_odd_ns_rejected():
    with pytest.raises(ValueError):
        M.Lattice(5)


def test_term_tables():
    lat = M.Lattice(4)
    assert M.charge_terms(lat, 1) == [({}, -0.5), ({2: "Z"}, -0.5)]
    assert M.gauss_terms(lat, 0) == [({0: "Z", 7: "X", 1: "X"}, 1.0)]
    assert M.gauss_terms(lat, 1) == [({2: "Z", 1: "X", 3: "X"}, -1.0)]
    assert M.current_terms(lat, 0, ETA) == [({0: "Y", 2: "X", 1: "Z"}, ETA / 4),
                                            ({0: "X", 2: "Y", 1: "Z"}, -ETA / 4)]
    s = lat.seam_sign
    assert M.current_terms(lat, 3, ETA) == [({6: "Y", 0: "X", 7: "Z"}, s * ETA / 4),
                                            ({6: "X", 0: "Y", 7: "Z"}, -s * ETA / 4)]
    assert M.hop_terms(lat, 3, ETA, exact_seam=True)[0][0] == {6: "X", 0: "X", 7: "Z", 2: "Z", 4: "Z"}


@pytest.mark.parametrize("builder", [M.hop_terms, M.current_terms])
def test_seam_parity_replacement_exact_on_gauss_subspace(builder):
    lat = M.Lattice(4)
    P = _gauss_projector(lat)
    A = _op(lat, builder(lat, 3, ETA)).to_matrix()
    B = _op(lat, builder(lat, 3, ETA, exact_seam=True)).to_matrix()
    assert np.abs(A - B).max() > 1        # genuinely different operators ...
    assert np.abs((A - B) @ P).max() < 1e-12   # ... equal on physical states


def test_ward_identity_on_physical_states():
    """i[H, J0(v)] = -(J1_{v+1/2} - J1_{v-1/2}) on the Gauss subspace (all
    seam operators parity-replaced)."""
    lat = M.Lattice(4)
    P = _gauss_projector(lat)
    H = _op(lat, M.hamiltonian_terms(lat, M0, G2, ETA)).to_matrix()
    for v in range(lat.ns):
        J0 = _op(lat, M.charge_terms(lat, v)).to_matrix()
        Jp = _op(lat, M.current_terms(lat, v, ETA)).to_matrix()
        Jm = _op(lat, M.current_terms(lat, v - 1, ETA)).to_matrix()
        lhs = 1j * (H @ J0 - J0 @ H)
        assert np.abs((lhs + Jp - Jm) @ P).max() < 1e-12


def test_strong_coupling_vacuum_is_physical():
    lat = M.Lattice(6)
    sv = Statevector(C.strong_coupling_vacuum(lat))
    for n in range(lat.ns):
        assert abs(sv.expectation_value(_op(lat, M.gauss_terms(lat, n))) - 1) < 1e-12


def test_to_sparse_pauli_op_rejects_duplicates():
    with pytest.raises(ValueError):
        M.to_sparse_pauli_op(2, [({0: "X", 0: "Z"}, 1.0), ({0: "X"}, 1.0)][:1] + [({5: "X"}, 1.0)])
