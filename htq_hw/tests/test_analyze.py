"""Estimators and calibration on synthetic bits; slice-file contract."""

import os

import numpy as np
import pytest

from htq_hw import analyze as A
from htq_hw import campaign as CP
from htq_hw import circuits as C
from htq_hw import sim as S
from htq_hw import target as T
from htq_hw.model import Lattice

SCR = "/tmp/claude-1000/-home-hlamm-Desktop-QC-hadronic-tensor/4a179643-b4b8-42be-b43f-ffcd49ec9721/scratchpad/htq_tests"
NS, CENTER, TIMES = 6, 2, (0.5,)


@pytest.fixture(scope="module")
def setup():
    os.makedirs(SCR, exist_ok=True)
    card = C.load_card()
    lat = Lattice(NS)
    tpl = os.path.join(SCR, "ideal_{family}.npz")
    S.write_ideal_grids(card, NS, CENTER, ("j0", "j1p1", "j1p2"), [0.0] + list(TIMES), tpl, threads=2)
    return card, lat, tpl


def _ideal_C(ideals, comp, t, ns, eta):
    fams, probe = A.COMPONENT_LAYOUT[comp]
    g0 = ideals["j0"]
    A0, id_a = ideals[fams[0]].A0, ideals[fams[0]].id_a
    idb = g0.id_b() if probe == "J0" else np.zeros(ns)
    tot = np.zeros(ns)
    for f in fams:
        g = ideals[f]
        c_a = A.FAMILY_COEFF[f] * (eta if f.startswith("j1") else 1.0)
        if probe == "J0":
            sx = g.vec("XB", t) - idb * g.scalar("X", t)
            one = g.vec("B", t)
        else:
            sx = eta / 4 * (g.vec("XT1", t) - g.vec("XT2", t))
            one = eta / 4 * (g.vec("T1", t) - g.vec("T2", t))
        tot += c_a * sx
    return tot + id_a * (one - idb) + idb * (A0 - id_a) + id_a * idb


def _synthetic_bits(card, lat, specs, kappa, shots, seed=5):
    """Noiseless logical sampling of every pub, then the ancilla column of
    physics and mirror pubs at t > 0 flipped with probability (1-kappa)/2:
    every ancilla-weighted signal is damped by exactly kappa."""
    rng = np.random.default_rng(seed)
    sim = S.make_simulator(lat.n_wires)
    bits = {}
    for s in specs:
        kind, off = CP.FAMILY_GADGET["j0" if s.mirror else s.family]
        qc = C.base_circuit(lat, card, kind, center=CENTER, accumulate="direct", gadget_center=CENTER + off)
        if s.n_steps:
            qc.compose(C.trotter_block(lat, s.n_steps, 1e-8 if s.mirror else s.t, n_wires=lat.n_wires), inplace=True)
        qc = C.readout_layer(qc, lat, list(range(lat.n_wires)), s.readout, s.anc_basis)
        arr = S.sample_bits(sim, qc, shots, seed=int(rng.integers(2**31)))
        if s.t > 0:
            flip = rng.random(shots) < (1 - kappa[s.readout]) / 2
            arr[flip, lat.ancilla] ^= 1
        bits[s.name] = arr
    return bits


def test_estimators_match_reference_formulas():
    lat = Lattice(4)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(500, lat.n_wires), dtype=np.uint8)
    s = 1.0 - 2.0 * bits
    r = A.estimate_probes(bits, lat, "Z")
    v = 1
    J0 = (-1) ** v / 2 - s[:, lat.site_qubit(v)] / 2
    assert r["J0"][v] == pytest.approx(J0.mean()) and r["xJ0"][v] == pytest.approx((s[:, lat.ancilla] * J0).mean())
    G1 = -s[:, 2] * s[:, 1] * s[:, 3]
    assert r["gauss"][1] == pytest.approx(G1.mean())
    rA = A.estimate_probes(bits, lat, "XYA")
    b = 3                                    # seam bond: sign = seam_sign(4) = -1
    T3 = lat.seam_sign * s[:, 6] * s[:, 7] * s[:, 0]
    assert rA["xT"][b] == pytest.approx((s[:, lat.ancilla] * T3).mean())
    assert list(rA["term"]) == [1, 2, 1, 2]
    rB = A.estimate_probes(bits, lat, "XYB")
    assert list(rB["term"]) == [2, 1, 2, 1]
    val, err = A.combine_j1(rA, rB, 1.3)
    assert val[0] == pytest.approx(1.3 / 4 * (rA["xT"][0] - rB["xT"][0]))


def test_analyze_recovers_injected_kappa_and_ideal_C(setup, tmp_path):
    card, lat, tpl = setup
    eta = card["couplings"]["eta"]
    specs = CP.manifest(times=TIMES, dt_half=True, dt_half_times=TIMES)
    kappa = {"Z": 0.7, "XYA": 0.55, "XYB": 0.55}
    shots = 6000
    bits = _synthetic_bits(card, lat, specs, kappa, shots)
    bpath = str(tmp_path / "htq_bits_synth.npz")
    np.savez_compressed(bpath, job_id="synth", backend="synthetic", pub_names=np.array(list(bits)), **bits)
    out_tpl = str(tmp_path / "slice_{comp}_t{t:.1f}.npz")
    written = A.analyze([bpath], tpl, out_tpl, NS, CENTER, eta=eta, backend="synthetic", log=None)
    ideals = A.load_ideal_grids(tpl, ("j0", "j1p1", "j1p2"))
    assert set(written) == ({(c, t) for c in A.COMPONENTS for t in (0.0, 0.5)}
                            | {(c, 0.5, 0.25) for c in A.COMPONENTS})
    for key, path in written.items():
        comp, t = key[0], key[1]
        z = np.load(path, allow_pickle=True)
        for k in A.SLICE_KEYS:
            assert k in z.files, (comp, t, k)
        assert str(z["component"]) == comp and float(z["t"]) == t and z["C_cal"].shape == (1, NS)
        assert z["C_cal"].dtype == np.complex128 and int(z["tier"]) == 4
        assert bool(z["raw_reference"]) == (t == 0) and float(z["dt"]) == (0.25 if len(key) == 3 else 0.5)
        g0 = ideals["j0"]
        ref = (np.abs(g0.sx_ideal0("J0")) if comp in ("00", "01")
               else np.minimum(np.abs(g0.sx_ideal0("J1", 1)), np.abs(g0.sx_ideal0("J1", 2))))
        if len(key) == 3:                                   # dt = 0.25 control: own file, equal t
            assert path.endswith("_dt0.25.npz")
            base = np.load(written[(comp, t)], allow_pickle=True)
            assert np.abs(z["C_cal"][0].real - base["C_cal"][0].real)[ref > 0.1].max() < 0.1
            continue
        if comp in ("00", "01"):
            assert "b_cal" in z.files
        else:
            assert "b_cal" not in z.files
        kap = z["kappa_v"][0]
        if t > 0:
            expect = kappa["Z"] if comp in ("00", "01") else kappa["XYA"]
            g0 = ideals["j0"]
            if comp in ("00", "01"):
                ref = np.abs(g0.sx_ideal0("J0"))
            else:
                ref = np.minimum(np.abs(g0.sx_ideal0("J1", 1)), np.abs(g0.sx_ideal0("J1", 2)))
            strong = ref > 0.1                          # sites with a usable reference signal
            kerr = 1.0 / np.sqrt(shots) / np.maximum(ref, 0.02) / expect
            assert np.all(np.abs(kap - expect)[strong] < 5 * kerr[strong]), (comp, kap, kerr)
            assert abs(kap[strong].mean() - expect) < 0.04, (comp, kap)
        else:
            assert np.allclose(kap, 1.0)
        ci = _ideal_C(ideals, comp, t, NS, eta)
        dev = (z["C_cal"][0].real - ci) / z["C_err"][0]
        assert np.mean(dev ** 2) < 4.0, (comp, t, dev)
        assert np.abs(z["C_cal"][0].real - ci)[ref > 0.1].max() < 0.06
        if comp == "00" and t > 0:      # raw is damped, calibrated is not
            assert np.abs(z["C"][0].real - ci).max() > np.abs(z["C_cal"][0].real - ci).max()


def test_merge_across_jobs_inflates_on_disagreement():
    a = {"anc_cal": np.array([1.0, 2.0]), "var": np.array([0.01, 0.01])}
    b = {"anc_cal": np.array([1.0, 3.0]), "var": np.array([0.01, 0.01])}
    m, e = A._merge([a, b])
    assert np.allclose(m, [1.0, 2.5])
    assert e[0] == pytest.approx(np.sqrt(0.005)) and e[1] > e[0]


def test_wing_anchor_and_rebuild():
    ns, c = 20, 9
    b_ideal = np.zeros(ns)
    b_cal = b_ideal + np.where(np.arange(ns) % 2 == 0, 0.07, -0.02)
    out, syst = A.wing_anchor(b_cal, b_ideal, c)
    wing = np.abs(np.arange(ns) - c) >= A.WING_MIN
    assert np.abs(out[wing]).max() < 1e-12 and np.all(syst >= 0)
    anc = np.ones(ns)
    assert np.allclose(A.rebuild_C(anc, b_cal, 0.5, np.full(ns, 0.5), 0.3),
                       anc + 0.5 * (b_cal - 0.5) + 0.5 * (0.3 - 0.5) + 0.25)


def test_kappa_never_transferred_between_probe_weights(setup, tmp_path):
    """Weight-3 (J1) probes damp faster than weight-1 (J0) probes in the same
    circuit (measured law kappa_w ~ kappa_1 exp(-0.05 (w-1) t)).  The
    calibration must therefore take each probe's kappa from the mirror's OWN
    probe of that weight: J0 probes from the Z-readout mirror, T1/T2 probes
    from the XY-readout mirrors.  Injecting different damping per readout and
    recovering both proves no J0 kappa leaks onto J1 probes."""
    card, lat, tpl = setup
    eta = card["couplings"]["eta"]
    specs = CP.manifest(times=TIMES)
    kappa = {"Z": 0.90, "XYA": 0.60, "XYB": 0.60}          # weight-1 vs weight-3 damping
    bits = _synthetic_bits(card, lat, specs, kappa, 6000, seed=11)
    bpath = str(tmp_path / "htq_bits_weights.npz")
    np.savez_compressed(bpath, job_id="w", backend="synthetic", pub_names=np.array(list(bits)), **bits)
    written = A.analyze([bpath], tpl, str(tmp_path / "slice_{comp}_t{t:.1f}.npz"), NS, CENTER,
                        eta=eta, backend="synthetic", log=None)
    ideals = A.load_ideal_grids(tpl, ("j0", "j1p1", "j1p2"))
    g0 = ideals["j0"]
    for comp in A.COMPONENTS:
        z = np.load(written[(comp, 0.5)], allow_pickle=True)
        kap = z["kappa_v"][0]
        if comp in ("00", "01"):
            ref, expect, wrong = np.abs(g0.sx_ideal0("J0")), kappa["Z"], kappa["XYA"]
        else:
            ref = np.minimum(np.abs(g0.sx_ideal0("J1", 1)), np.abs(g0.sx_ideal0("J1", 2)))
            expect, wrong = kappa["XYA"], kappa["Z"]
        strong = ref > 0.1
        assert strong.any()
        got = kap[strong].mean()
        assert abs(got - expect) < 0.05, (comp, got, expect)
        assert abs(got - wrong) > 0.15, (comp, got, wrong)     # the other weight's kappa is NOT used


def test_dither_family_uses_its_own_ideal_grid(tmp_path):
    """The j0d (dither) pubs must be assembled against the j0d ideal grid --
    its insertion sits on the ODD site CENTER+1, so id_a = -1/2 and A0 is its
    own insert_1pt -- while the damping still comes from the j0 mirror they
    share a job with (hence kappa keeps using the j0 grid's t=0 row)."""
    card = C.load_card()
    lat = Lattice(NS)
    tpl = str(tmp_path / "ideal_{family}.npz")
    S.write_ideal_grids(card, NS, CENTER, ("j0", "j0d"), [0.0, 0.5], tpl, threads=2)
    g0, gd = A.IdealGrid(tpl.format(family="j0")), A.IdealGrid(tpl.format(family="j0d"))
    assert g0.id_a == 0.5 and gd.id_a == -0.5 and abs(g0.A0 - gd.A0) > 0.1

    specs = CP.manifest(times=TIMES, dither=True)
    assert any(s.family == "j0d" for s in specs)
    kappa = {"Z": 0.8, "XYA": 0.8, "XYB": 0.8}
    bits = _synthetic_bits(card, lat, specs, kappa, 4000, seed=13)
    bpath = str(tmp_path / "htq_bits_dither.npz")
    np.savez_compressed(bpath, job_id="d", backend="synthetic", pub_names=np.array(list(bits)), **bits)
    written = A.analyze([bpath], tpl, str(tmp_path / "slice_{comp}_t{t:.1f}.npz"), NS, CENTER,
                        components=("00", "00d"), eta=card["couplings"]["eta"], log=None)
    assert ("00d", 0.5) in written and ("00", 0.5) in written
    zd = np.load(written[("00d", 0.5)], allow_pickle=True)
    z0 = np.load(written[("00", 0.5)], allow_pickle=True)
    assert float(zd["id_a"]) == -0.5 and float(z0["id_a"]) == 0.5      # from the j0d / j0 grids
    assert str(zd["component"]) == "00d"
    idb = g0.id_b()
    for z, g in ((zd, gd), (z0, g0)):                                  # C_cal vs that family's ideal
        sx = g.vec("XB", 0.5) - idb * g.scalar("X", 0.5)
        ci = -0.5 * sx + g.id_a * (g.vec("B", 0.5) - idb) + idb * (g.A0 - g.id_a) + g.id_a * idb
        ok = np.abs(g0.sx_ideal0("J0")) > 0.1
        assert np.abs(z["C_cal"][0].real - ci)[ok].max() < 0.08, str(z["component"])
    ok = np.abs(g0.sx_ideal0("J0")) > 0.1
    assert abs(zd["kappa_v"][0][ok].mean() - kappa["Z"]) < 0.05        # shared j0 mirror
