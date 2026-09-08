"""Circuit builders vs qiskit-only references (Operator / Statevector)."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from htq_hw import KINDS, MIRROR_EPS
from htq_hw import circuits as C
from htq_hw import model as M


def _reference_block(lat, center, params):
    """htensor.wavepacket.block_circuit re-expressed with PauliEvolutionGate."""
    qc = QuantumCircuit(lat.n_qubits)

    def gen(bond, kind):
        qa, qb, ql = lat.bond_qubits(bond)
        s = lat.seam_sign if lat.is_seam(bond) else 1
        # local order (a, b, l) -> label 'l b a'
        if kind == "hop":
            t = [("ZXX", s / 4), ("ZYY", s / 4)]
        else:
            t = [("ZYX", s / 4), ("ZXY", -s / 4)]
        return SparsePauliOp([l for l, _ in t], [c for _, c in t]), [qa, qb, ql]

    for l in sorted({k[0] for k in params}):
        for kind in KINDS:
            for (_, _, off) in sorted(k for k in params if k[0] == l and k[1] == kind):
                ang = params[(l, kind, off)]
                if kind in ("cur", "hop"):
                    op, qs = gen((center + off) % lat.ns, kind)
                    qc.append(PauliEvolutionGate(op, time=ang), qs)
                elif kind == "site":
                    qc.rz(ang, lat.site_qubit(center + off))
                else:
                    qc.rx(ang, lat.link_qubit(center + off))
    return qc


def test_block_matches_pauli_evolution_reference():
    lat = M.Lattice(4)
    rng = np.random.default_rng(0)
    offsets = [-1, 0, 1, 2]
    params = C.params_from_vector(rng.normal(size=2 * 4 * 4), offsets, 2)
    assert Operator(C.wavepacket_block(lat, 1, params)).equiv(
        Operator(_reference_block(lat, 1, params)))


def test_bond_rotation_generators():
    lat = M.Lattice(4)
    for bond in (0, 3):           # bulk and seam
        for kind, labels in (("hop", [("ZXX", 1), ("ZYY", 1)]), ("cur", [("ZYX", 1), ("ZXY", -1)])):
            s = lat.seam_sign if lat.is_seam(bond) else 1
            op = SparsePauliOp([l for l, _ in labels], [s * c / 4 for _, c in labels])
            ref = QuantumCircuit(lat.n_qubits)
            ref.append(PauliEvolutionGate(op, time=0.37), list(lat.bond_qubits(bond)))
            qc = QuantumCircuit(lat.n_qubits)
            C.bond_rotation(qc, lat, bond, kind, 0.37)
            assert Operator(qc).equiv(Operator(ref))


@pytest.mark.parametrize("form", C.HOP_FORMS)
def test_hop_forms_equal_pauli_evolution(form):
    lat = M.Lattice(4)
    for bond in (1, 3):
        qa, qb, ql = lat.bond_qubits(bond)
        ref = QuantumCircuit(lat.n_qubits)
        ref.append(PauliEvolutionGate(SparsePauliOp(["ZXX", "ZYY"], [1, 1]), time=0.41), [qa, qb, ql])
        qc = QuantumCircuit(lat.n_qubits)
        C.hop(qc, lat, bond, 0.41, form)
        assert Operator(qc).equiv(Operator(ref))
    t = Parameter("t")
    a = C.assign(C.trotter_block(lat, 1, t, form=form), t, 0.5)
    assert Operator(a).equiv(Operator(C.trotter_block(lat, 1, 0.5)))


def test_params_from_vector_length_check():
    with pytest.raises(ValueError):
        C.params_from_vector(np.zeros(5), [0, 1], 1)


def test_prep_card_is_gauge_invariant_small_volume():
    card = C.load_card()
    lat = M.Lattice(6)
    sv = Statevector(C.prep_circuit(card, 6, 2))
    for n in range(lat.ns):
        G = M.to_sparse_pauli_op(lat.n_qubits, M.gauss_terms(lat, n))
        assert abs(sv.expectation_value(G) - 1) < 1e-10


def test_parametric_block_equals_numeric():
    lat = M.Lattice(4)
    t = Parameter("t")
    blk = C.trotter_block(lat, 2, t)
    assert Operator(C.assign(blk, t, 0.5)).equiv(Operator(C.trotter_block(lat, 2, 0.5)))
    assert Operator(C.assign(blk, "t", 0.5)).equiv(Operator(C.trotter_block(lat, 2, 0.5)))


def test_mirror_is_identity():
    lat = M.Lattice(4)
    t = Parameter("t")
    phys, mir = C.physics_and_mirror(C.trotter_block(lat, 2, t), t, 1.0)
    assert np.allclose(Operator(mir).data, np.eye(2 ** lat.n_qubits), atol=1e-6)
    assert not np.allclose(Operator(phys).data, np.eye(2 ** lat.n_qubits), atol=1e-2)
    assert phys.count_ops() == mir.count_ops()


@pytest.mark.parametrize("term", ["J1a", "J1b"])
@pytest.mark.parametrize("accumulate", C.ACCUMULATE)
def test_j1_gadget_equals_controlled_paulis(term, accumulate):
    a, l, b, anc = 0, 1, 2, 3
    pa, pl, pb = C.J1_PAULIS[term]
    ref = QuantumCircuit(4)
    C.controlled_pauli(ref, anc, {a: pa, l: pl, b: pb})
    qc = QuantumCircuit(4)
    C.j1_gadget(qc, anc, a, l, b, term, accumulate)
    assert Operator(qc).equiv(Operator(ref))


def test_gadget_skeleton_shared_and_edges():
    lat = M.Lattice(6)
    a, b, l = lat.bond_qubits(2)
    anc = lat.ancilla
    for acc, expect in (("ring", [(b, l), (l, a), (anc, a), (l, a), (b, l)]),
                        ("ladder", [(l, b), (b, a), (anc, a), (b, a), (l, b)])):
        skel = []
        for term in ("J1a", "J1b"):
            g = C.insertion_gadget(lat, term, 2, acc)
            skel.append([tuple(g.find_bit(q).index for q in i.qubits) for i in g.data
                         if i.operation.num_qubits == 2])
        assert skel[0] == skel[1] == expect
    g0 = C.insertion_gadget(lat, "J0", 2)
    assert g0.count_ops() == {"cz": 1}


def test_base_circuit_layout():
    card = C.load_card()
    lat = M.Lattice(6)
    qc = C.base_circuit(lat, card, "J1a", center=2, accumulate="ladder")
    assert qc.num_qubits == lat.n_wires
    anc_ops = [i.operation.name for i in qc.data if lat.ancilla in
               [qc.find_bit(q).index for q in i.qubits]]
    assert anc_ops == ["h", "cz"]


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_isa_readout_rotation(pauli):
    qc = QuantumCircuit(1)
    for g, ang in C.ISA_ROTATION[pauli]:
        getattr(qc, g)(*([ang, 0] if ang is not None else [0]))
    V = Operator(qc)
    assert (V @ Operator.from_label(pauli) @ V.adjoint()).equiv(Operator.from_label("Z"))


def test_basis_maps():
    lat = M.Lattice(6)
    z = C.basis_map(lat, "Z")
    assert all(z[q] == "Z" for q in lat.matter_qubits) and all(z[q] == "X" for q in lat.link_qubits)
    for n in range(lat.ns):        # every Gauss operator is diagonal in the Z basis
        (ops, _), = M.gauss_terms(lat, n)
        assert all(z[q] == p for q, p in ops.items())
    a, b = C.basis_map(lat, "XYA"), C.basis_map(lat, "XYB")
    assert a[lat.site_qubit(0)] == "Y" and a[lat.site_qubit(1)] == "X"
    assert all(a[q] != b[q] for q in lat.matter_qubits)
    assert all(a[q] == b[q] == "Z" for q in lat.link_qubits)
    for bond in lat.bonds:         # each J1 term is measurable in exactly one of XYA/XYB
        (t1, _), (t2, _) = M.current_terms(lat, bond)
        for t in (t1, t2):
            assert sum(all(m[q] == p for q, p in t.items()) for m in (a, b)) == 1


def test_readout_layer_targets_physical_qubits():
    lat = M.Lattice(4)
    isa = QuantumCircuit(12)
    layout = [11, 3, 5, 7, 0, 2, 4, 6, 8]      # 8 system wires + ancilla
    out = C.readout_layer(isa, lat, layout, "Z")
    meas = [(out.find_bit(i.qubits[0]).index, out.find_bit(i.clbits[0]).index)
            for i in out.data if i.operation.name == "measure"]
    assert meas == [(layout[i], i) for i in range(lat.n_wires)]
    rotated = {out.find_bit(i.qubits[0]).index for i in out.data if i.operation.name == "sx"}
    assert rotated == {layout[q] for q in lat.link_qubits} | {layout[lat.ancilla]}


def test_link_reference_states():
    lat = M.Lattice(6)
    for ref, xval in (("+", 1.0), ("-", -1.0)):
        sv = Statevector(C.strong_coupling_vacuum(lat, link_ref=ref))
        for q in lat.link_qubits:
            X = M.to_sparse_pauli_op(lat.n_qubits, [({q: "X"}, 1.0)])
            assert abs(sv.expectation_value(X).real - xval) < 1e-12
        for n in range(lat.ns):        # Gauss law holds for both references
            G = M.to_sparse_pauli_op(lat.n_qubits, M.gauss_terms(lat, n))
            assert abs(sv.expectation_value(G).real - 1) < 1e-12
    with pytest.raises(ValueError):
        C.strong_coupling_vacuum(lat, link_ref="x")


def test_relA_card_prep_three_layers_gauge_invariant():
    card = C.load_card("relA_k1.26_s0.75_ns50")
    assert C.card_link_ref(card) == "-" and C.card_n_layers(card) == 3
    lat = M.Lattice(6)
    qc = C.prep_circuit(card, 6, 2)
    assert qc.count_ops()["z"] == lat.ns                       # one Z per link from the |-> reference
    sv = Statevector(qc)
    for n in range(lat.ns):
        G = M.to_sparse_pauli_op(lat.n_qubits, M.gauss_terms(lat, n))
        assert abs(sv.expectation_value(G) - 1) < 1e-10
    assert T_count(qc) > T_count(C.prep_circuit(C.load_card(), 6, 2))   # extra vacuum layer


def T_count(qc):
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def test_cz_via_neighbor_equals_cz():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    ref = QuantumCircuit(3)
    ref.cz(2, 1)                    # anc = 2, target = 1, via = 0
    qc = QuantumCircuit(3)
    C.cz_via_neighbor(qc, 2, 1, 0)
    assert Operator(qc).equiv(Operator(ref))
    lat = M.Lattice(6)
    g = C.insertion_gadget(lat, "J0", 3, "ladder", anc_site=2)
    pairs = {tuple(sorted(g.find_bit(q).index for q in i.qubits)) for i in g.data if i.operation.num_qubits == 2}
    assert pairs == {(4, lat.ancilla), (4, 6)}          # only (anc, site 2) and the rail (site 2, site 3)
    d = C.insertion_gadget(lat, "J0", 3, "direct", anc_site=2)
    assert Operator(g).equiv(Operator(d))
