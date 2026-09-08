"""Campaign presets: counts, pairing, stretch ordering, vacuum cards, qPDF estimator."""

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector

from htq_hw import analyze as A
from htq_hw import campaign as CP
from htq_hw import circuits as C
from htq_hw import sim as S
from htq_hw import target as T
from htq_hw.model import Lattice, gauss_terms, to_sparse_pauli_op


def test_preset_counts_and_structure():
    core = CP.preset_specs("relA-core")
    assert len(core) == 12 * 18 + 9 + 2 * 18 + 4 * 18 == 333
    assert sum(s.stretch for s in core) == 72 and {s.t for s in core if s.stretch} == set(CP.STRETCH_TIMES)
    per = [s for s in core if s.t == 0.5 and abs(s.dt - 0.5) < 1e-9]
    assert len(per) == 18 and sum(s.mirror for s in per) == 6 and sum(s.family == "j0d" for s in per) == 3
    assert sum(1 for s in core if s.t == 0) == 9                      # no dither refs
    assert len([s for s in core if abs(s.dt - 0.25) < 1e-9]) == 36
    bridge = CP.preset_specs("prod-bridge")
    assert len(bridge) == 16 * 9 + 6 and {s.family for s in bridge} == {"j0", "j0d"}
    vac = CP.preset_specs("vac-w00")
    assert len(vac) == 50 and {s.card for s in vac} == set(CP.VAC_CARDS.values()) and {s.readout for s in vac} == {"Z"}
    q = CP.preset_specs("qpdf-scan")
    assert len(q) == 8 * 20 and all(s.family == "qpdf" and s.n_steps == 0 for s in q)
    assert all(s.name.startswith(f"{s.preset}.{s.card}:") for s in core + bridge + vac + q)
    p = CP.parse_pub_name(core[-1].name)
    assert p["prefix"] == "relA-core.relA_k1.26_s0.75_ns50:" and p["mirror"] and p["t"] == 8.0


def test_compose_stretch_last_and_jobs():
    specs = CP.compose_presets(("relA-core", "prod-bridge", "vac-w00", "qpdf-scan"))
    names = [s.name for s in specs]
    assert len(set(names)) == len(names) == 693
    first = next(i for i, s in enumerate(specs) if s.stretch)
    assert all(s.stretch for s in specs[first:]) and not any(s.stretch for s in specs[:first])
    jobs = CP.group_jobs(specs, 8)
    by = {s.name: s for s in specs}
    assert sum(len(j) for j in jobs) == len(specs)
    flags = [by[j[0]].stretch for j in jobs]
    assert flags == sorted(flags)                                    # stretch jobs last
    for j in jobs:
        assert len({by[n].card for n in j}) == 1 and len({by[n].stretch for n in j}) == 1
        for n in j:                                                  # mirrors ride with their physics
            s = by[n]
            if s.family in ("j1p1", "j1p2", "j0d") and not s.mirror and s.t > 0:
                assert CP.pub_name("j0", s.t, True, s.readout, s.anc_basis, s.dt, s.prefix) in j
    plan = CP.shots_plan(specs, {n: 1000 for n in names}, 180, 250e-6, weighting="equal")
    sp = plan["split"]
    assert abs(sp["committed"] - 128.5) < 1 and abs(sp["stretch"] - 18.0) < 0.1 and sp["contingency"] > 30
    assert all(plan["shots"][s.name] >= 30000 for s in specs if s.mirror)
    assert plan["shots"][[s for s in specs if s.family == "qpdf"][0].name] == 20000


def test_vacuum_cards_build_and_check_ns6():
    """Vacuum cards: no block, gauge-invariant prep, ideal grid + check at Ns=6."""
    import os
    scr = "/tmp/claude-1000/-home-hlamm-Desktop-QC-hadronic-tensor/4a179643-b4b8-42be-b43f-ffcd49ec9721/scratchpad/htq_tests"
    os.makedirs(scr, exist_ok=True)
    for name in CP.VAC_CARDS.values():
        card = C.load_card(name)
        assert C.card_block_params(card) == {}
        lat = Lattice(6)
        qc = C.prep_circuit(card, 6, 2)
        sv = Statevector(qc)
        for n in range(6):
            assert abs(sv.expectation_value(to_sparse_pauli_op(lat.n_qubits, gauss_terms(lat, n))) - 1) < 1e-10
    card = C.load_card("prod_vac_ns50")
    tpl = os.path.join(scr, "vac6_ideal_{family}.npz")
    S.write_ideal_grids(card, 6, 2, ("j0",), [0.0, 0.5], tpl, threads=2)
    be = T.resolve_backend("grid:4x5")
    emb = T.choose_embedding(be, 6, 2)
    res = S.check(be, Lattice(6), card, emb, tpl, [0.0, 0.5], families=("j0",), threads=2, log=None)
    assert max(res.values()) < 5e-3


def test_qpdf_estimator_vs_statevector_ns6():
    """qXX/qYY/qXY/qYX settings sampled noiselessly at Ns=6 reproduce
    <psi-bar(z) W psi(0)> from the exact statevector for z = +-2 (m=1)."""
    card = C.load_card()
    lat, center = Lattice(6), 2
    prep = C.prep_only_circuit(lat, card, center)
    sv = Statevector(prep)
    sim = S.make_simulator(lat.n_wires)
    bits = {}
    for kind in CP.QPDF_KINDS if hasattr(CP, "QPDF_KINDS") else ("XX", "YY", "XY", "YX"):
        bm = C.qpdf_basis_map(lat, center, 1, kind)
        qc = C.readout_layer(prep, lat, list(range(lat.n_wires)), bm, "Z")
        bits[f"q{kind}m1"] = S.sample_bits(sim, qc, 20000, seed=3)
    est = A.estimate_qpdf(bits, lat, center, ms=(1,))
    for z in (2, -2):
        va, vb = (center, center + z) if z > 0 else (center + z, center)
        qa, qb = lat.site_qubit(va), lat.site_qubit(vb)
        string = {q: "Z" for q in range(qa + 1, qb)}
        def op(pa, pb, c):
            return to_sparse_pauli_op(lat.n_wires, [({qa: pa, **string, qb: pb}, c)])
        o_r = op("X", "X", 0.5) + op("Y", "Y", 0.5)
        o_i = op("X", "Y", 0.5) - op("Y", "X", 0.5)
        h_exact = complex(sv.expectation_value(o_r).real + 1j * sv.expectation_value(o_i).real)
        h, err = est[z]
        assert abs(h.real - h_exact.real) < 4 * max(err.real, 0.01) and abs(h.imag - h_exact.imag) < 4 * max(err.imag, 0.01), (z, h, h_exact)
        assert abs(h_exact) > 0.02 or abs(h) < 0.05


def test_phoenix_alias_and_per_shot_time():
    import warnings
    warnings.simplefilter("ignore")
    be = T.resolve_backend("phoenix")
    assert not T.is_real_backend(be)
    assert getattr(be, "_htq_standin_for", None) == "ibm_phoenix" or be.name == "ibm_phoenix"
    assert "nighthawk" in T.backend_label(be) or be.name == "ibm_phoenix"
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(be.num_qubits, 2)
    qc.sx(0)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])
    pst = T.per_shot_time(be, qc)
    assert pst["total_s"] > 0 and (pst["reset_us"] is None or pst["reset_us"] > 0)


def test_qpdf_amplitude_reference_convention_ns6():
    """A(z) = (C_R - i C_I)/2 for z != 0 and A(0) = <n(centre)> (connected),
    the convention of scripts/quasipdf_analysis.py and data/qpdf_card_refs.npz."""
    from htq_hw.model import to_sparse_pauli_op

    card = C.load_card()
    lat, center = Lattice(6), 2
    prep = C.prep_only_circuit(lat, card, center)
    sv = Statevector(prep)
    sim = S.make_simulator(lat.n_wires)
    bits = {}
    for kind in ("XX", "YY", "XY", "YX"):
        bm = C.qpdf_basis_map(lat, center, 1, kind)
        bits[f"q{kind}m1"] = S.sample_bits(sim, C.readout_layer(prep, lat, list(range(lat.n_wires)), bm, "Z"),
                                           40000, seed=7)
    z_bits = S.sample_bits(sim, C.readout_layer(prep, lat, list(range(lat.n_wires)), "Z", "Z"), 40000, seed=8)
    amp = A.qpdf_amplitude(bits, lat, center, ms=(1,), z_bits=z_bits)
    for z in (2, -2):
        va, vb = (center, center + z) if z > 0 else (center + z, center)
        qa, qb = lat.site_qubit(va), lat.site_qubit(vb)
        string = {q: "Z" for q in range(qa + 1, qb)}

        def op(pa, pb, c):
            return to_sparse_pauli_op(lat.n_wires, [({qa: pa, **string, qb: pb}, c)])

        C_R = sv.expectation_value(op("X", "X", 0.5) + op("Y", "Y", 0.5)).real
        C_I = sv.expectation_value(op("X", "Y", 0.5) - op("Y", "X", 0.5)).real
        exact = (C_R - 1j * C_I) / 2
        got, err = amp[z]
        assert abs(got.real - exact.real) < max(4 * err.real, 0.01), (z, got, exact)
        assert abs(got.imag - exact.imag) < max(4 * err.imag, 0.01), (z, got, exact)
    n_exact = sv.expectation_value(to_sparse_pauli_op(lat.n_wires, [({}, 0.5), ({lat.site_qubit(center): "Z"}, -0.5)])).real
    assert abs(amp[0].real if isinstance(amp[0], complex) else amp[0][0].real - n_exact) < 0.02
    h, e = A.qpdf_h_of_m(amp, ms=(-1, 0, 1))
    assert h.shape == (3,) and np.isfinite(h).all() and np.isfinite(e).all()
    assert h[1] == amp[0][0] and h[2] == amp[2][0]


def test_every_trotter_path_uses_the_card_couplings():
    """Regression: the module-level M0/G2/ETA are the production point, so a
    card at another coupling (relA 0.4/1.4/2.3) must not be evolved with them.
    Caught by comparing against the main-repo relA grids, where the prepared
    state agreed to 1e-6 in <H> but t > 0 observables differed by 0.3."""
    from qiskit.quantum_info import Operator, Statevector

    relA = C.load_card("relA_k1.26_s0.75_ns50")
    assert C.card_couplings(relA) == (0.4, 1.4, 2.3)
    assert C.card_couplings(C.load_card()) == (0.7, 1.1, 1.3)
    lat = Lattice(4)
    prod_blk = C.trotter_block(lat, 1, 0.5)
    relA_blk = C.trotter_block(lat, 1, 0.5, *C.card_couplings(relA))
    assert not Operator(prod_blk).equiv(Operator(relA_blk))

    be = T.resolve_backend("grid:4x5")
    emb = T.choose_embedding(be, 4, 1)
    b = T.transpile_bundle(be, lat, relA, emb, (1,), "cz", "J0")
    assert b.card["name"] == relA["name"]
    isa = b.base.compose(b.blocks[1]["physics"])
    rel, wire = S.relabel_to_logical(isa, emb.layout, lat.n_wires)
    fw = S.final_wires(range(lat.n_wires), b.blocks[1]["layout"], wire)
    obs = S.probe_observables(lat, ("J0",))
    sv_isa = Statevector(rel)

    def logical(couplings):
        qc = C.base_circuit(lat, relA, "J0", center=1, accumulate="ladder")
        qc.compose(C.trotter_block(lat, 1, 0.5, *couplings, n_wires=lat.n_wires), inplace=True)
        return Statevector(qc)

    sv_right, sv_wrong = logical(C.card_couplings(relA)), logical((0.7, 1.1, 1.3))
    dr = max(abs(sv_isa.expectation_value(S.remap(obs[k], fw, rel.num_qubits)).real
                 - sv_right.expectation_value(obs[k]).real) for k in ("B_0", "B_1", "XB_1", "X"))
    dw = max(abs(sv_isa.expectation_value(S.remap(obs[k], fw, rel.num_qubits)).real
                 - sv_wrong.expectation_value(obs[k]).real) for k in ("B_0", "B_1", "XB_1", "X"))
    assert dr < 1e-5, dr
    assert dw > 1e-3, dw
