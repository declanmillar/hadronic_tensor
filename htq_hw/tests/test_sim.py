"""Ideal grids, ISA-to-logical relabelling, check and rehearsal at Ns <= 6."""

import os

import numpy as np
import pytest

from htq_hw import analyze as A
from htq_hw import campaign as CP
from htq_hw import circuits as C
from htq_hw import sim as S
from htq_hw import target as T
from htq_hw.model import Lattice

@pytest.fixture(scope="module")
def grids6(tmp_path_factory):
    scr = tmp_path_factory.mktemp("sim")
    card = C.load_card()
    tpl = os.path.join(str(scr), "ideal6_{family}.npz")
    S.write_ideal_grids(card, 6, 2, ("j0", "j1p1", "j1p2"), [0.0, 0.5, 1.0], tpl, threads=2)
    return card, tpl


def test_ideal_grid_keys_and_values(grids6):
    card, tpl = grids6
    z = np.load(tpl.format(family="j0"), allow_pickle=True)
    for k in ("times", "probes", "bonds", "id_a", "c_a", "H_prep", "one_pt", "one_pt_J0", "one_pt_J1",
              "insert_1pt", "XB_0", "YB_0", "B_0", "XT1_0", "YT1_0", "T1_0", "XT2_0", "X", "Y", "family"):
        assert k in z.files, k
    assert float(z["id_a"]) == 0.5 and float(z["c_a"]) == -0.5 and str(z["family"]) == "j0"
    assert z["XB_2"].shape == (3,)
    # J0 insertion at t=0: <X_anc> = <Z_c> = (-1)^c - 2 <J0(c)>, and <X_anc J0(c)> = id_b(c) <Z_c> - 1/2
    assert float(z["X"][0]) == pytest.approx(1.0 - 2 * float(z["insert_1pt"]), abs=1e-9)
    assert float(z["XB_2"][0]) == pytest.approx(0.5 * float(z["X"][0]) - 0.5, abs=1e-9)
    assert float(z["B_2"][0]) == pytest.approx(float(z["insert_1pt"]), abs=1e-9)
    z1 = np.load(tpl.format(family="j1p1"), allow_pickle=True)
    assert float(z1["id_a"]) == 0.0 and float(z1["c_a"]) == pytest.approx(0.25 * card["couplings"]["eta"])
    assert abs(float(z1["insert_1pt"])) < 0.1
    assert float(z1["insert_1pt"]) == pytest.approx(float(np.load(tpl.format(family="j1p2"), allow_pickle=True)["insert_1pt"]))


def test_relabel_round_trip():
    from qiskit.quantum_info import Statevector
    card = C.load_card()
    lat = Lattice(4)
    be = T.resolve_backend("grid:4x5")
    emb = T.choose_embedding(be, 4, 1)
    b = T.transpile_bundle(be, lat, card, emb, steps=(1,), basis="cz", kind="J0")
    rel, wire = S.relabel_to_logical(b.base.compose(b.blocks[1]["physics"]), emb.layout, lat.n_wires)
    fw = S.final_wires(range(lat.n_wires), b.blocks[1]["layout"], wire)
    logical = C.base_circuit(lat, card, "J0", center=1, accumulate="ladder")
    logical.compose(C.trotter_block(lat, 1, 0.5, n_wires=lat.n_wires), inplace=True)
    obs = S.probe_observables(lat, ("J0",))
    sv_rel, sv_log = Statevector(rel), Statevector(logical)
    for k in ("XB_1", "B_3", "X"):
        assert abs(sv_rel.expectation_value(S.remap(obs[k], fw, rel.num_qubits)).real
                   - sv_log.expectation_value(obs[k]).real) < 1e-5      # O3 synthesis tolerance


def test_pad_mps_matches_statevector():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector
    sim = S.make_simulator(3, "matrix_product_state")
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(0.4, 2)
    qc.save_matrix_product_state(label="mps")
    mps = sim.run(qc).result().data()["mps"]
    tail = QuantumCircuit(5)
    tail.cx(2, 4)
    tail.rx(0.7, 3)
    d = S.expectations(sim, tail, {"e": SparsePauliOp("ZIZII"), "f": SparsePauliOp("IXIZI")}, initial_mps=mps)
    ref = QuantumCircuit(5)
    ref.h(0)
    ref.cx(0, 1)
    ref.ry(0.4, 2)
    ref.cx(2, 4)
    ref.rx(0.7, 3)
    sv = Statevector(ref)
    assert d["e"] == pytest.approx(sv.expectation_value(SparsePauliOp("ZIZII")).real, abs=1e-9)
    assert d["f"] == pytest.approx(sv.expectation_value(SparsePauliOp("IXIZI")).real, abs=1e-9)


@pytest.mark.parametrize("target", ["heavyhex:5", "grid:4x5"])
def test_check_end_to_end_ns6(grids6, target):
    card, tpl = grids6
    lat = Lattice(6)
    be = T.resolve_backend(target)
    emb = T.choose_embedding(be, 6, 2)
    res = S.check(be, lat, card, emb, tpl, [0.0, 0.5, 1.0], threads=2, tol=5e-3, log=None)
    assert max(res.values()) < 5e-3
    assert {k[1] for k in res} >= {"base", "physics t=0.5", "mirror t=1.0"}


def test_check_detects_layout_bug(grids6):
    card, tpl = grids6
    lat = Lattice(6)
    be = T.resolve_backend("grid:4x5")
    emb = T.choose_embedding(be, 6, 2)
    orig = S.final_wires

    def broken(initial_layout, final_layout, wire):      # a wrong final layout: two matter sites swapped
        fw = orig(initial_layout, final_layout, wire)
        fw[0], fw[2] = fw[2], fw[0]
        return fw
    S.final_wires = broken
    try:
        with pytest.raises(AssertionError):
            S.check(be, lat, card, emb, tpl, [0.0], threads=2, log=None)
    finally:
        S.final_wires = orig


def test_rehearse_to_slices_ns6(grids6, tmp_path):
    card, tpl = grids6
    lat = Lattice(6)
    be = T.resolve_backend("grid:4x5")
    emb = T.choose_embedding(be, 6, 2)
    specs = CP.manifest(times=(0.5,))
    shots = {s.name: 800 for s in specs}
    out = S.rehearse(be, lat, card, emb, specs, shots, tpl, str(tmp_path), threads=2, log=None)
    assert os.path.exists(out["bits"]) and os.path.exists(out["meta"])
    assert set(out["slices"]) == {(c, t) for c in A.COMPONENTS for t in (0.0, 0.5)}
    z = np.load(out["slices"][("00", 0.5)], allow_pickle=True)
    assert abs(z["kappa_v"][0][2] - 1.0) < 0.1                         # noiseless: kappa ~ 1 (centre site)
    assert np.abs(z["kappa_v"][0] - 1.0).max() < 0.4                   # weak-reference sites: shot noise
    # p2 = 0.01 on this ~290-gate circuit gives ~0.45 centre-qubit X/Y errors per trajectory
    noisy = S.rehearse(be, lat, card, emb, specs, shots, tpl, str(tmp_path / "noisy"), noise=(0.01, 1e-3),
                       seed=2, threads=2, n_traj=32, log=None)
    zn = np.load(noisy["slices"][("00", 0.5)], allow_pickle=True)
    assert 0.15 < zn["kappa_v"][0][2] < 0.97                           # damping visible at the centre site
    assert str(np.load(noisy["meta"].replace("htq_job_", "htq_bits_").replace(".json", ".npz"),
                       allow_pickle=True)["backend"]).startswith("rehearsal")


def test_noise_transform_counts():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3)
    for _ in range(200):
        qc.cz(0, 1)
        qc.sx(2)
    rng = np.random.default_rng(0)
    out = S.noise_transform(qc, rng, 1.0, 0.0)
    ops = out.count_ops()
    assert ops["cz"] == 200 and sum(ops.get(g, 0) for g in ("x", "y", "z")) > 200
    out0 = S.noise_transform(qc, rng, 0.0, 0.0)
    assert out0.count_ops() == qc.count_ops()
