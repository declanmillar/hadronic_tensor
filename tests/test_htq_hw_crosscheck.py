"""htq_hw (self-contained hardware package) vs htensor (reference implementation).

  - prep statevector equality (vacuum ansatz + wavepacket block, card params)
  - Trotter step Operator equality
  - current / Gauss / Hamiltonian operator equality (incl. the seam string)
  - J1 parity-ladder gadget vs htensor.measure.controlled_pauli
  - card vacuum thetas vs stateprep.optimize_vacuum
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector

from htensor import Z2Lattice, currents, stateprep, trotter, wavepacket
from htensor import hamiltonian as ham
from htensor.measure import controlled_pauli

from htq_hw import M0, G2, ETA, VACUUM_THETAS
from htq_hw import circuits as C
from htq_hw import model as M


@pytest.fixture(scope="module")
def card():
    return C.load_card()


def _htensor_prep(card, ns, center):
    lat = Z2Lattice(ns, pbc=True)
    blk = card["block"]
    params = wavepacket.params_from_vector(np.array(blk["vec"]), list(blk["offsets"]), blk["n_layers"])
    qc = stateprep.vacuum_ansatz(lat, np.array(card["vacuum"]["thetas"]))
    qc.compose(wavepacket.block_circuit(lat, center, params), inplace=True)
    return qc


@pytest.mark.parametrize("ns,center", [(6, 2), (10, 4)])
def test_prep_statevector_matches_htensor(card, ns, center):
    ref = Statevector(_htensor_prep(card, ns, center))
    mine = Statevector(C.prep_circuit(card, ns, center))
    assert abs(ref.inner(mine)) ** 2 > 1 - 1e-10
    assert ref.equiv(mine)


def test_trotter_step_operator_matches_htensor():
    lat4, lat = Z2Lattice(4, pbc=True), M.Lattice(4)
    ref = Operator(trotter.trotter_step(lat4, M0, G2, ETA, 0.5))
    assert Operator(C.trotter_step(lat, 0.5)).equiv(ref)
    t = Parameter("t")
    for form in C.HOP_FORMS:
        blk = C.trotter_block(lat, 2, t, form=form)
        assert Operator(C.assign(blk, t, 1.0)).equiv(
            Operator(trotter.trotter_circuit(lat4, M0, G2, ETA, 1.0, 2)))
    assert trotter.seam_sign(lat4) == lat.seam_sign


def test_operators_match_htensor():
    lat4, lat = Z2Lattice(4, pbc=True), M.Lattice(4)
    n = lat.n_qubits
    for v in range(4):
        assert M.to_sparse_pauli_op(n, M.charge_terms(lat, v)) == currents.charge_density(lat4, v)
        assert M.to_sparse_pauli_op(n, M.gauss_terms(lat, v)) == ham.gauss_operator(lat4, v)
    for b in range(4):
        assert (M.to_sparse_pauli_op(n, M.current_terms(lat, b, ETA, exact_seam=True))
                == currents.bond_current(lat4, b, ETA))
        assert (M.to_sparse_pauli_op(n, M.hop_terms(lat, b, ETA, exact_seam=True))
                == ham.hop_term(lat4, b, ETA))
    assert (M.to_sparse_pauli_op(n, M.hamiltonian_terms(lat, M0, G2, ETA, exact_seam=True))
            == ham.build_hamiltonian(lat4, M0, G2, ETA))
    # parity-replaced seam current == string current on the Gauss-law subspace
    P = np.eye(2 ** n)
    for m in range(4):
        P = ((np.eye(2 ** n) + ham.gauss_operator(lat4, m).to_matrix()) / 2) @ P
    A = M.to_sparse_pauli_op(n, M.current_terms(lat, 3, ETA)).to_matrix()
    B = currents.bond_current(lat4, 3, ETA).to_matrix()
    assert np.abs((A - B) @ P).max() < 1e-12


@pytest.mark.parametrize("term", ["J1a", "J1b"])
@pytest.mark.parametrize("accumulate", C.ACCUMULATE)
def test_j1_gadget_matches_htensor_controlled_pauli(term, accumulate):
    a, l, b, anc = 0, 1, 2, 3
    pa, pl, pb = C.J1_PAULIS[term]
    ref = QuantumCircuit(4)
    controlled_pauli(ref, anc, {a: pa, l: pl, b: pb})
    qc = QuantumCircuit(4)
    C.j1_gadget(qc, anc, a, l, b, term, accumulate)
    assert Operator(qc).equiv(Operator(ref))


def test_j1_terms_match_htensor_split():
    from htensor.measure import split_current
    lat4 = Z2Lattice(50, pbc=True)
    _, terms = split_current(currents.bond_current(lat4, 24, ETA))
    got = {tuple(sorted(ops.items())): c for ops, c in terms}
    assert got == {((48, "Y"), (49, "Z"), (50, "X")): pytest.approx(C.GADGET_COEFF["J1a"] * ETA),
                   ((48, "X"), (49, "Z"), (50, "Y")): pytest.approx(C.GADGET_COEFF["J1b"] * ETA)}


VAC_FILE = "data/test/vac_prod_test.npz"   # shared convention: thetas, n_layers, link_ref, m0, g2, eta


def test_vacuum_thetas_match_reference(card):
    """Card/package vacuum angles == the shared reference file when present
    (fast), else == a fresh stateprep.optimize_vacuum at Ns=6 (~2 min)."""
    import os
    if os.path.exists(VAC_FILE):
        z = np.load(VAC_FILE, allow_pickle=True)
        assert int(z["n_layers"]) == 2 and str(z["link_ref"]) == card["link_ref"]
        assert (float(z["m0"]), float(z["g2"]), float(z["eta"])) == (M0, G2, ETA)
        ref = np.asarray(z["thetas"], dtype=float)
        assert np.allclose(ref, VACUUM_THETAS, atol=1e-5)
    else:   # fresh BFGS may land on an angle-equivalent minimum: compare states
        lat6 = Z2Lattice(6, pbc=True)
        opt = stateprep.optimize_vacuum(lat6, M0, G2, ETA, n_layers=2, restarts=2)["thetas"]
        a = Statevector(stateprep.vacuum_ansatz(lat6, opt))
        b = Statevector(stateprep.vacuum_ansatz(lat6, np.array(VACUUM_THETAS)))
        assert abs(a.inner(b)) ** 2 > 1 - 1e-6
    assert np.allclose(card["vacuum"]["thetas"], VACUUM_THETAS)


# ------------------------------------------------------------------ relA card: link_ref "-", 3 layers
RELA = "relA_k1.26_s0.75_ns50"


def _htensor_prep_card(card, ns, center):
    lat = Z2Lattice(ns, pbc=True)
    blk = card["block"]
    params = wavepacket.params_from_vector(np.array(blk["vec"]), list(blk["offsets"]), blk["n_layers"])
    qc = stateprep.vacuum_ansatz(lat, np.array(card["vacuum"]["thetas"]), link_ref=card["link_ref"])
    qc.compose(wavepacket.block_circuit(lat, center, params), inplace=True)
    return qc


@pytest.mark.parametrize("ns,center", [(6, 2), (10, 4)])
def test_relA_prep_statevector_matches_htensor(ns, center):
    card = C.load_card(RELA)
    assert card["link_ref"] == "-" and C.card_n_layers(card) == 3
    ref = Statevector(_htensor_prep_card(card, ns, center))
    mine = Statevector(C.prep_circuit(card, ns, center))
    assert abs(ref.inner(mine)) ** 2 > 1 - 1e-10
    assert ref.equiv(mine)


def test_vacuum_ansatz_three_layers_minus_reference_operator():
    lat4, lat = Z2Lattice(4, pbc=True), M.Lattice(4)
    rng = np.random.default_rng(3)
    th = rng.uniform(-2, 2, size=12)
    ref = Operator(stateprep.vacuum_ansatz(lat4, th, link_ref="-"))
    assert Operator(C.vacuum_ansatz(lat, th, link_ref="-")).equiv(ref)
    assert not Operator(C.vacuum_ansatz(lat, th, link_ref="+")).equiv(ref)
    assert Operator(C.strong_coupling_vacuum(lat, link_ref="-")).equiv(
        Operator(trotter.strong_coupling_vacuum_circuit(lat4, link_ref="-")))


def test_relA_card_matches_source_files():
    card = C.load_card(RELA)
    v = np.load("data/vac_relA.npz", allow_pickle=True)
    w = np.load("data/wp_relA_k+1.26_s0.75_L3.npz", allow_pickle=True)
    assert np.allclose(card["vacuum"]["thetas"], v["thetas"]) and str(v["link_ref"]) == card["link_ref"]
    assert np.allclose(card["block"]["vec"], w["vec"]) and card["block"]["offsets"] == [int(o) for o in w["offsets"]]
    assert (card["couplings"]["m0"], card["couplings"]["g2"], card["couplings"]["eta"]) == (0.4, 1.4, 2.3)
    assert card["certification"]["status"] == "certified" and "10" in card["certification"]["cert"]


WIDE = ["prod_k1.26_s1.00_ns50", "prod_k1.26_s1.50_ns50", "relA_k0.00_s0.75_ns50",
        "relA_k1.26_s1.00_ns50", "relA_k1.26_s1.50_ns50", "prod_k1.26_s0.75_ns58", "relA_k1.26_s0.75_ns58"]


@pytest.mark.parametrize("name", WIDE)
def test_wide_and_ns58_cards_prep_match_htensor(name):
    """Arbitrary block offsets (+-6, +-9) and L = 3/4: statevector equality
    at Ns=10 (offsets wrap identically in both implementations)."""
    card = C.load_card(name)
    blk = card["block"]
    assert len(blk["vec"]) == blk["n_layers"] * 4 * len(blk["offsets"])
    ref = Statevector(_htensor_prep_card(card, 10, 4))
    mine = Statevector(C.prep_circuit(card, 10, 4))
    assert ref.equiv(mine) and abs(ref.inner(mine)) ** 2 > 1 - 1e-10
    if name.endswith("ns58"):
        assert card["ns"] == 58 and card["center"] == 28
    status = card.get("certification", {}).get("status") if isinstance(card.get("certification"), dict) else None
    assert status in ("certified", "provisional", None)
    if name == "relA_k1.26_s1.50_ns50":
        assert status == "provisional"
