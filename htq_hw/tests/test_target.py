"""Embeddings, skeleton/fold/twirl helpers, and (slow) offline transpile bounds."""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from htq_hw import target as T
from htq_hw import circuits as C
from htq_hw.model import Lattice


def _valid_cycle(g, cyc, length):
    return (len(cyc) == length and len(set(cyc)) == length
            and all(g.has_edge(cyc[i], cyc[(i + 1) % length]) for i in range(length)))


def test_heavyhex5_short_cycles():
    g = T.Graph.from_coupling_map(CouplingMap.from_heavy_hex(5))
    c12 = list(T.heavyhex_cycles(g, 12))
    c20 = list(T.heavyhex_cycles(g, 20))
    assert len(c12) >= 1 and len(c20) >= 1
    assert all(_valid_cycle(g, c, 12) for c in c12)
    assert all(_valid_cycle(g, c, 20) for c in c20)
    assert list(T.heavyhex_cycles(g, 14)) == []       # heavy-hex cycles are multiples of 4


@pytest.mark.parametrize("length", [100, 116])
def test_boston_long_cycles(length):
    g = T.Graph.from_backend(T.resolve_backend("fake:boston"))
    assert g.n == 156 and g.n_edges == 176
    cs = list(T.heavyhex_cycles(g, length, max_cycles=2))
    assert len(cs) == 2 and all(_valid_cycle(g, c, length) for c in cs)


def test_grid_coordinates():
    rows, cols, coords = T.grid_coordinates(T.Graph.from_coupling_map(CouplingMap.from_grid(4, 5)))
    assert (rows, cols) == (4, 5) and coords[7] == (1, 2)
    g = T.Graph.from_backend(T.resolve_backend("fake:nighthawk"))
    assert g.n_edges == 218 and T.grid_coordinates(g)[:2] == (12, 10)
    assert T.grid_coordinates(T.Graph.from_coupling_map(CouplingMap.from_heavy_hex(3))) is None


def test_grid_ladder_and_cycle_4x5():
    g = T.Graph.from_coupling_map(CouplingMap.from_grid(4, 5))
    lad = T.grid_ladder(g, 6, 2)
    assert lad and all(e.kind == "ladder" for e in lad)
    for e in lad:
        e.validate(g)
        lat = Lattice(6)
        for n in range(6):      # pendant of link (n, n+1) adjacent to site n+1, rail adjacent sites
            assert g.has_edge(e.layout[lat.link_qubit(n)], e.layout[lat.site_qubit(n + 1)])
            assert g.has_edge(e.layout[lat.site_qubit(n)], e.layout[lat.site_qubit(n + 1)])
        assert g.has_edge(e.layout[lat.ancilla], e.layout[lat.site_qubit(2)])
    assert T.grid_ladder(g, 8, 3) == []               # no room for 8 sites + pendants
    cyc = T.grid_cycle(g, 8, 3)
    assert cyc and all(e.kind == "grid" for e in cyc)
    for e in cyc:
        e.validate(g)


def test_nighthawk_ladder_ns50():
    be = T.resolve_backend("fake:nighthawk")
    g = T.Graph.from_backend(be)
    embs = T.grid_ladder(g, 50, 24)
    assert len(embs) >= 6
    lat = Lattice(50)
    for e in embs:
        e.validate(g)
        assert len(e.required_edges()) == 101
        assert g.has_edge(e.layout[lat.ancilla], e.layout[lat.site_qubit(24)])
    best = T.choose_embedding(be, 50, 24)
    assert best.kind == "ladder" and np.isfinite(best.info["score"])


def test_choose_embedding_modes():
    boston = T.resolve_backend("fake:boston")
    e = T.choose_embedding(boston, 50, 24)
    assert e.kind == "ring" and len(e.layout) == 101
    g = T.Graph.from_backend(boston)
    free = [w for w in g.neighbors(e.layout[48]) if w not in set(e.layout[:100])]
    assert e.layout[100] in free and g.degree(e.layout[48]) == 3
    nighthawk = T.resolve_backend("fake:nighthawk")
    assert T.choose_embedding(nighthawk, 58, 28).kind == "grid"
    assert T.choose_embedding(nighthawk, 50, 24, mode="transpiler").layout is None
    with pytest.raises(ValueError):
        T.choose_embedding(nighthawk, 60, 29, mode="ladder")


def test_validate_negative():
    g = T.Graph.from_coupling_map(CouplingMap.from_grid(4, 5))
    e = T.grid_ladder(g, 6, 2)[0]
    bad = T.Embedding("ladder", 6, 2, list(e.layout))
    bad.layout[0], bad.layout[1] = bad.layout[1], bad.layout[0]
    with pytest.raises(ValueError):
        bad.validate(g)
    dup = T.Embedding("ladder", 6, 2, list(e.layout))
    dup.layout[3] = dup.layout[4]
    with pytest.raises(ValueError):
        dup.validate(g)
    with pytest.raises(ValueError):
        T.Embedding("ring", 6, 2, list(e.layout)).validate(g)   # ladder layout is not a ring


def test_skeleton_assertion():
    a = QuantumCircuit(3)
    a.cz(0, 1)
    a.rzz(0.3, 1, 2)
    b = QuantumCircuit(3)
    b.cz(0, 1)
    b.rzz(1e-8, 1, 2)
    assert T.assert_skeleton_equal(a, b) == 2
    c = QuantumCircuit(3)
    c.cz(0, 1)
    c.rzz(0.3, 2, 1)
    with pytest.raises(AssertionError):
        T.assert_skeleton_equal(a, c)
    d = QuantumCircuit(3)
    d.cz(0, 1)
    with pytest.raises(AssertionError):
        T.assert_skeleton_equal(a, d)


def test_fold_rzz_range_and_unitary():
    rng = np.random.default_rng(3)
    qc = QuantumCircuit(3)
    for k in range(12):
        qc.rz(rng.uniform(-3, 3), k % 3)
        qc.sx(k % 3)
        qc.rzz(rng.uniform(-2 * np.pi, 2 * np.pi), k % 3, (k + 1) % 3)
    folded = T.fold_rzz(qc)
    angles = [float(i.operation.params[0]) for i in folded.data if i.operation.name == "rzz"]
    assert angles and all(-1e-9 <= a <= np.pi / 2 + 1e-9 for a in angles)
    assert "global_phase" not in folded.count_ops()
    assert Operator(folded).equiv(Operator(qc))
    assert T.skeleton(folded) == T.skeleton(qc)


def test_twirl_preserves_unitary_and_skeleton():
    rng = np.random.default_rng(7)
    rng2 = np.random.default_rng(1)
    qc = QuantumCircuit(4)
    for _ in range(4):
        for q in range(4):
            qc.rz(rng2.uniform(0, 2 * np.pi), q)
            qc.sx(q)
        qc.cz(0, 1)
        qc.rzz(rng2.uniform(0, np.pi / 2), 1, 2)
        qc.cz(2, 3)
    U = Operator(qc)
    for _ in range(5):
        tq = T.twirl(qc, rng)
        assert T.skeleton(tq) == T.skeleton(qc)
        assert Operator(tq).equiv(U)


def test_resolve_backend_specs():
    g = T.resolve_backend("grid:3x4")
    assert g.num_qubits == 12 and "rzz" not in g.operation_names
    gf = T.resolve_backend("grid:3x4", fractional=True)
    assert "rzz" in gf.operation_names
    h = T.resolve_backend("heavyhex:3")
    assert h.num_qubits == CouplingMap.from_heavy_hex(3).size()
    bf = T.resolve_backend("fake:boston", fractional=True)
    assert "rzz" in bf.operation_names and T.backend_label(bf) == "fake_boston"
    with pytest.raises(ValueError):
        T.resolve_backend("fake:nowhere")


def test_small_bundle_end_to_end():
    """Full pipeline on a 4x5 grid ladder at Ns=6: layout preserved, composed
    blocks, matched skeletons, readout layer."""
    card = C.load_card()
    lat = Lattice(6)
    for basis in ("cz", "rzz"):
        be = T.resolve_backend("grid:4x5", fractional=(basis == "rzz"))
        emb = T.choose_embedding(be, 6, 2)
        b = T.transpile_bundle(be, lat, card, emb, steps=(1, 2), basis=basis, kind="J1a")
        assert b.layout_preserved and b.blocks[2]["mode"] == "composed"
        assert T.count_2q(b.step) == 9 * 4                 # 4 two-qubit gates per hop
        assert b.blocks[2]["n2q"] == 2 * b.blocks[1]["n2q"]
        ro = C.readout_layer(b.base.compose(b.blocks[1]["physics"]), lat, b.blocks[1]["layout"], "XYA")
        assert ro.count_ops()["measure"] == lat.n_wires


# ------------------------------------------------------------------ slow: 101 qubits
@pytest.mark.slow
def test_ring_transpile_bounds_boston():
    card = C.load_card()
    lat = Lattice(50)
    be = T.resolve_backend("fake:boston")
    emb = T.choose_embedding(be, 50, 24)
    b = T.transpile_bundle(be, lat, card, emb, steps=(1, 12), basis="cz", kind="J0")
    assert T.count_2q(b.base) <= 1100
    assert T.count_ops_2q(b.step) == {"cz": T.count_2q(b.step)}
    assert T.count_2q(b.step) <= 620                     # 534 observed (parametric O3)
    assert b.blocks[12]["n2q"] <= 12 * 470              # 5484 = 457/step observed
    bf = T.transpile_bundle(T.resolve_backend("fake:boston", fractional=True), lat, card, emb,
                            steps=(1, 12), basis="rzz", kind="J0")
    assert "rzz" in T.count_ops_2q(bf.step) and T.count_2q(bf.step) <= 620
    assert bf.blocks[12]["n2q"] <= 12 * 470


@pytest.mark.slow
def test_ladder_transpile_bounds_nighthawk():
    card = C.load_card()
    lat = Lattice(50)
    for basis in ("cz", "rzz"):
        be = T.resolve_backend("fake:nighthawk", fractional=(basis == "rzz"))
        emb = T.choose_embedding(be, 50, 24)
        b = T.transpile_bundle(be, lat, card, emb, steps=(1, 2), basis=basis, kind="J1a")
        assert b.layout_preserved
        assert T.count_2q(b.step) == 300
        assert b.blocks[2]["n2q"] == 600


# ------------------------------------------------------------------ end-to-end statevector
def _active_subcircuit(qc):
    """Restrict a physical-register circuit to the qubits it touches (idle
    qubits stay |0>), so small-target ISA circuits can be simulated exactly.
    -> (compact circuit, {physical qubit: compact index})."""
    qc = qc.remove_final_measurements(inplace=False)
    used = sorted({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})
    idx = {p: i for i, p in enumerate(used)}
    out = QuantumCircuit(len(used))
    out.global_phase = qc.global_phase
    for inst in qc.data:
        out.append(inst.operation, [idx[qc.find_bit(q).index] for q in inst.qubits])
    return out, idx


def _probabilities(qc, layout=None):
    from qiskit.quantum_info import Statevector
    sub, idx = _active_subcircuit(qc)
    qargs = None if layout is None else [idx[p] for p in layout]
    return Statevector(sub).probabilities(qargs=qargs)


@pytest.mark.parametrize("spec,basis", [("grid:4x5", "cz"), ("grid:4x5", "rzz"),
                                        ("heavyhex:5", "cz"), ("heavyhex:5", "rzz")])
def test_isa_pipeline_matches_logical_statevector(spec, basis):
    """base + physics block + readout on the ISA circuit, read through the
    recorded layouts, reproduces the logical circuit's outcome distribution
    (checks layout bookkeeping incl. routing on the ring, fold, compose, and
    that the mirror is the identity)."""
    card = C.load_card()
    lat = Lattice(6)
    be = T.resolve_backend(spec, fractional=(basis == "rzz"))
    emb = T.choose_embedding(be, 6, 2)
    acc = "ladder" if emb.kind == "ladder" else "ring"
    b = T.transpile_bundle(be, lat, card, emb, steps=(1, 2), basis=basis, kind="J1a")
    ident = list(range(lat.n_wires))
    for n, ro_basis in ((1, "Z"), (1, "XYA"), (2, "XYB")):
        blk = b.blocks[n]
        layout = [blk["layout"][i] for i in range(lat.n_wires)]
        isa = C.readout_layer(b.base.compose(blk["physics"]), lat, blk["layout"], ro_basis)
        logical = C.base_circuit(lat, card, "J1a", center=2, accumulate=acc)
        logical.compose(C.trotter_block(lat, n, n * 0.5, n_wires=lat.n_wires), inplace=True)
        logical = C.readout_layer(logical, lat, ident, ro_basis)
        assert np.abs(_probabilities(isa, layout) - _probabilities(logical)).max() < 1e-5
    blk = b.blocks[2]
    isa_m = C.readout_layer(b.base.compose(blk["mirror"]), lat, blk["layout"], "Z")
    base_only = C.readout_layer(C.base_circuit(lat, card, "J1a", center=2, accumulate=acc), lat, ident, "Z")
    p_m = _probabilities(isa_m, [blk["layout"][i] for i in range(lat.n_wires)])
    assert np.abs(p_m - _probabilities(base_only)).max() < 1e-5
