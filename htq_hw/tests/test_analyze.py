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

NS, CENTER, TIMES = 6, 2, (0.5,)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    scr = tmp_path_factory.mktemp("analyze")
    card = C.load_card()
    lat = Lattice(NS)
    tpl = os.path.join(str(scr), "ideal_{family}.npz")
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


def test_grouped_rela_campaign_retains_split_xy_slices(setup, tmp_path):
    """The real packer's per-readout groups must not drop W10/W11.

    Keep the relA-core grouping (dither, j1 mirrors, plain Z and half-step
    controls), but only one positive time and six sites for an offline test.
    The split and unsplit runs consume exactly the same recorded bits.
    """
    card, lat, tpl = setup
    specs = CP.manifest(times=TIMES, j1_mirrors=True, dither=True,
                        dither_refs=False, dt_half=True, dt_half_times=TIMES,
                        plain=True, card="synthetic_ns6", preset="relA-core")
    groups = CP.group_jobs(specs)
    prefix = CP.name_prefix("relA-core", "synthetic_ns6")
    for dt in (0.5, 0.25):
        names = [CP.pub_name("j0", 0.5, False, r, dt=dt, prefix=prefix)
                 for r in ("XYA", "XYB")]
        assert not any(all(n in group for n in names) for group in groups)
    bits = _synthetic_bits(card, lat, specs, {"Z": 0.8, "XYA": 0.6, "XYB": 0.75},
                           2000, seed=71)
    split_paths = []
    for i, names in enumerate(groups):
        path = tmp_path / f"bits_{i}.npz"
        np.savez_compressed(path, job_id=f"recorded-{i}", **{n: bits[n] for n in names})
        split_paths.append(str(path))
    whole_path = tmp_path / "bits_whole.npz"
    np.savez_compressed(whole_path, job_id="whole", **bits)
    kwargs = dict(eta=card["couplings"]["eta"], prefix=prefix, log=None)
    whole = A.analyze([str(whole_path)], tpl, str(tmp_path / "whole_{comp}_t{t:.1f}.npz"),
                      NS, CENTER, **kwargs)
    split = A.analyze(split_paths, tpl, str(tmp_path / "split_{comp}_t{t:.1f}.npz"),
                      NS, CENTER, **kwargs)
    assert set(split) == set(whole)
    g0 = A.IdealGrid(tpl.format(family="j0"))
    strong = ((np.abs(g0.sx_ideal0("J1", 1)) > A.KAP_GUARD)
              & (np.abs(g0.sx_ideal0("J1", 2)) > A.KAP_GUARD))
    assert strong.any()
    for comp in ("10", "11"):
        for key in ((comp, 0.0), (comp, 0.5), (comp, 0.5, 0.25)):
            with np.load(split[key]) as got, np.load(whole[key]) as ref:
                for field in ("C", "C_cal", "C_err", "kappa_v", "kappa1_v", "kappa2_v"):
                    # A split job fills weak mirror sites only from its own
                    # observed bonds; the old complete-job convention fills
                    # across a full term. Strong-reference sites are exact.
                    np.testing.assert_allclose(got[field][:, strong], ref[field][:, strong],
                                               rtol=1e-12, atol=1e-12)
                t, dt = key[1], key[2] if len(key) == 3 else 0.5
                assert float(got["dt"]) == dt
                fams = A.COMPONENT_LAYOUT[comp][0]
                relevant = {CP.pub_name(f, t, False, r, dt=dt, prefix=prefix)
                            for f in fams for r in ("XYA", "XYB")}
                source_ids = {f"recorded-{i}" for i, names in enumerate(groups)
                              if relevant.intersection(names)}
                assert set(got["merged_jobs"].tolist()) == source_ids
                assert len(got["merged_jobs"]) == len(source_ids)


class _XYIdeal:
    """Nonzero mirror references avoid statistical/median-fill ambiguity."""

    def __init__(self, family):
        self.ns = NS
        self.id_a = 0.5 if family == "j0" else 0.0
        self.A0 = 0.3

    def id_b(self):
        return (-1.0) ** np.arange(NS) / 2

    def sx_ideal0(self, probe, term=None):
        assert probe == "J1"
        return np.full(NS, 0.4 if term == 1 else -0.3)


def _xy_cached_job(job_id, setting, *, kappa=1.0, families=("j0",),
                   signal=0.16, error=0.02, mirror=True, dt=0.5):
    """Exact estimator fixture: isolate job joining from shot fluctuations."""
    job = A.JobBits({}, Lattice(NS), job_id, eta=1.3)
    term = np.array([A.j1_term_in_setting(b, setting) for b in range(NS)])

    def record(name, sx, sxerr):
        job.bits[name] = np.zeros((100, job.lat.n_wires), np.uint8)
        job._cache[name] = dict(term=term, xT=np.broadcast_to(sx, (NS,)).copy(),
                                xT_err=np.full(NS, sxerr), T=np.zeros(NS),
                                T_err=np.full(NS, error), N=100, setting=setting,
                                sx=np.broadcast_to(sx, (NS,)).copy(), sxe=np.full(NS, sxerr),
                                xa=0.0, xa_err=0.01)

    for i, family in enumerate(families):
        record(CP.pub_name(family, 0.5, False, setting, dt=dt),
               kappa * signal * (i + 1), error)
    if mirror:
        ref = np.where(term == 1, 0.4, -0.3)
        record(CP.pub_name("j0", 0.5, True, setting, dt=dt), kappa * ref, 0.0)
    return job


@pytest.mark.parametrize("comp", ["10", "11"])
def test_split_xy_settings_use_each_jobs_own_mirror(comp):
    fams = A.COMPONENT_LAYOUT[comp][0]
    ideals = {f: _XYIdeal(f) for f in set(fams) | {"j0"}}
    jobs = [_xy_cached_job("source-A", "XYA", kappa=0.35, families=fams, signal=0.16),
            _xy_cached_job("source-B", "XYB", kappa=0.85, families=fams, signal=0.06)]
    got = A.component_slice(comp, 0.5, jobs, ideals, jobs[0].lat, CENTER, wing=False)
    assert got is not None
    coeff = sum(A.FAMILY_COEFF[f] * (1.3 if f.startswith("j1") else 1.0) * (i + 1)
                for i, f in enumerate(fams))
    sign = (-1.0) ** np.arange(NS)
    expected = coeff * 1.3 / 4 * (0.16 - 0.06) * sign
    raw = coeff * 1.3 / 4 * (0.35 * 0.16 - 0.85 * 0.06) * sign
    np.testing.assert_allclose(got["C_cal"][0], expected, atol=1e-15)
    np.testing.assert_allclose(got["C"][0], raw, atol=1e-15)
    np.testing.assert_allclose(got["kappa1_v"][0], np.where(sign == 1, 0.35, 0.85))
    np.testing.assert_allclose(got["kappa2_v"][0], np.where(sign == 1, 0.85, 0.35))
    assert set(got["merged_jobs"].tolist()) == {"source-A", "source-B"}
    if comp == "10":
        summary = A.kappa_summary(jobs, ideals, [0.5], jobs[0].lat)
        for term in (1, 2):
            kap = got[f"kappa{term}_v"][0]
            np.testing.assert_array_equal(summary[f"j0_J1T{term}_kappa_t0.5"], kap)
            np.testing.assert_array_equal(summary[f"j0_J1T{term}_mask_t0.5"], kap < A.KAP_MASK)


@pytest.mark.parametrize("all_weak", [False, True], ids=["local-term-fill", "all-below-guard"])
def test_split_xy_weak_reference_fill_stays_with_its_job_and_term(all_weak):
    class GuardedIdeal(_XYIdeal):
        def sx_ideal0(self, probe, term=None):
            ref = super().sx_ideal0(probe, term)
            # Otherwise one weak and two strong reference bonds for each
            # term in each setting: local medians then differ by job.
            ref[slice(None) if all_weak else slice(0, 2)] = A.KAP_GUARD / 2
            return ref

    ideal = GuardedIdeal("j0")
    jobs = [_xy_cached_job("A", "XYA"), _xy_cached_job("B", "XYB")]
    observed, expected_kap = [], []
    for job, setting, damping, signal in zip(jobs, ("XYA", "XYB"),
                                            ((0.35, 0.55), (0.85, 0.70)), (0.16, 0.06)):
        p = job.est(CP.pub_name("j0", 0.5, False, setting), setting)
        m = job.est(CP.pub_name("j0", 0.5, True, setting), setting)
        kap = np.where(p["term"] == 1, damping[0], damping[1])
        ref = np.where(p["term"] == 1, ideal.sx_ideal0("J1", 1), ideal.sx_ideal0("J1", 2))
        p["xT"] = p["sx"] = kap * signal
        m["xT"] = m["sx"] = kap * ref
        # At weak-reference sites even a large measured mirror value must
        # not be divided by a near-zero ideal reference.
        m["xT"][np.abs(ref) <= A.KAP_GUARD] = 0.9
        m["xT_err"] = m["sxe"] = np.full(NS, 0.006)
        observed.append(p["xT"])
        expected_kap.append(np.ones(NS) if all_weak else kap)
    got = A.component_slice("10", 0.5, jobs, {"j0": ideal}, jobs[0].lat, CENTER, wing=False)
    assert got is not None
    even = np.arange(NS) % 2 == 0
    np.testing.assert_allclose(got["kappa1_v"][0], np.where(even, expected_kap[0], expected_kap[1]))
    np.testing.assert_allclose(got["kappa2_v"][0], np.where(even, expected_kap[1], expected_kap[0]))
    coeff = A.FAMILY_COEFF["j0"] * jobs[0].eta / 4
    expected = coeff * np.where(even, 1.0, -1.0) * (observed[0] / expected_kap[0]
                                                    - observed[1] / expected_kap[1])
    np.testing.assert_allclose(got["C_cal"][0], expected, atol=1e-15)
    assert np.isfinite(got["C_err"]).all()
    if all_weak:
        # The established guard falls back to kappa=1, with relative mirror
        # error median(sigma_m)/KAP_GUARD, when no reference bond is usable.
        kerr = 0.006 / A.KAP_GUARD
        expected_err = abs(coeff) * np.sqrt(2 * 0.02 ** 2
                                           + kerr ** 2 * (observed[0] ** 2 + observed[1] ** 2))
        np.testing.assert_allclose(got["C_err"][0], expected_err, rtol=1e-12)


def test_split_xy_missing_mirror_cannot_borrow_from_another_job():
    a = _xy_cached_job("A", "XYA", kappa=0.35)
    b = _xy_cached_job("B", "XYB", kappa=0.85, mirror=False)
    mirror_only = _xy_cached_job("unrelated-B-mirror", "XYB", kappa=0.2, families=())
    got = A.component_slice("10", 0.5, [a, b, mirror_only], {"j0": _XYIdeal("j0")},
                            a.lat, CENTER, wing=False)
    assert got is None


def test_split_xy_incomplete_insertion_families_are_reported():
    """Present but unusable pubs must not vanish without a diagnostic."""
    jobs = [_xy_cached_job(f"{setting}-{family}", setting, families=(family,))
            for setting in ("XYA", "XYB") for family in ("j1p1", "j1p2")]
    ideals = {f: _XYIdeal(f) for f in ("j0", "j1p1", "j1p2")}
    messages = []
    got = A.component_slice("11", 0.5, jobs, ideals, jobs[0].lat, CENTER,
                            wing=False, log=messages.append)
    assert got is None
    assert any("j1p1+j1p2" in msg and "XYA/XYB" in msg and "incomplete" in msg
               and "t=0.5" in msg and "dt=0.5" in msg for msg in messages), messages


def test_split_xy_repeated_setting_does_not_double_count_its_partner():
    """Two independent A samples improve only A's error, not the single B."""
    a1 = _xy_cached_job("A-1", "XYA", signal=0.16)
    a2 = _xy_cached_job("A-2", "XYA", signal=0.16)
    b = _xy_cached_job("B", "XYB", signal=0.06)
    got = A.component_slice("10", 0.5, [a1, a2, b], {"j0": _XYIdeal("j0")},
                            a1.lat, CENTER, wing=False)
    assert got is not None
    prefactor = abs(A.FAMILY_COEFF["j0"] * a1.eta / 4)
    expected_error = prefactor * np.sqrt(0.02 ** 2 / 2 + 0.02 ** 2)
    np.testing.assert_allclose(got["C_err"], expected_error, rtol=1e-12)
    assert set(got["merged_jobs"].tolist()) == {"A-1", "A-2", "B"}
    assert len(got["merged_jobs"]) == 3


def test_split_xy_does_not_join_different_evolution_steps():
    a = _xy_cached_job("A-dt05", "XYA", dt=0.5)
    b = _xy_cached_job("B-dt025", "XYB", dt=0.25)
    for dt in (0.5, 0.25):
        got = A.component_slice("10", 0.5, [a, b], {"j0": _XYIdeal("j0")},
                                a.lat, CENTER, wing=False, dt=dt)
        assert got is None


def test_split_xy_does_not_reuse_a_complete_jobs_setting():
    a = _xy_cached_job("A", "XYA", signal=0.16)
    b = _xy_cached_job("B", "XYB", signal=0.06)
    complete = A.JobBits({**a.bits, **b.bits}, a.lat, "complete", eta=a.eta)
    complete._cache = {**a._cache, **b._cache}
    extra_a = _xy_cached_job("extra-A", "XYA", signal=0.24)
    kwargs = dict(ideals={"j0": _XYIdeal("j0")}, lat=a.lat, center=CENTER, wing=False)
    ref = A.component_slice("10", 0.5, [complete], **kwargs)
    got = A.component_slice("10", 0.5, [complete, extra_a], **kwargs)
    for field in ("C", "C_cal", "C_err", "kappa_v", "one_pt", "nshot"):
        np.testing.assert_allclose(got[field], ref[field], rtol=1e-12, atol=1e-12)
    assert got["merged_jobs"].tolist() == ["complete"]


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


# ---- quasi-PDF reduction ---------------------------------------------------

def test_qpdf_distribution_reproduces_the_reference_transform():
    """The convention, pinned against the pipeline that produced the ideal
    references: taste phase (-1)^m, Gaussian window sigma_m = 5, FT against
    P = k0 per site, normalize then take the moment.  The stored h already
    carries the phase, so it is removed before feeding it back in."""
    import pathlib
    from htq_hw import circuits as C
    p = pathlib.Path(C.ref_path("qpdf_card_refs.npz"))
    if not p.exists():
        pytest.skip("ideal qpdf references not present")
    z = np.load(p, allow_pickle=True)
    ms = np.asarray(z["ms"])
    for key in ("prod_s0.75", "relA_s1.50"):
        h = np.asarray(z[f"{key}_h"]) * (-1.0) ** ms
        qt, norm, x = A.qpdf_distribution(h, ms, float(z[f"{key}_k0"]), xs=np.asarray(z["xs"]))
        assert np.allclose(qt, np.asarray(z[f"{key}_qt"]), atol=1e-12)
        assert norm == pytest.approx(float(z[f"{key}_norm"]), abs=1e-12)
        assert x == pytest.approx(float(z[f"{key}_x"]), abs=1e-12)


def test_qpdf_bits_grouping_and_vacuum_pairing():
    cards = ["prod_k1.26_s0.75_ns50", "prod_vac_ns50", "relA_k1.26_s1.50_ns50", "relA_vac_ns50"]
    assert A.qpdf_vacuum_card(cards[0], cards) == "prod_vac_ns50"
    assert A.qpdf_vacuum_card(cards[2], cards) == "relA_vac_ns50"      # never the prod vacuum
    assert A.qpdf_vacuum_card("prod_vac_ns50", cards) is None
    assert A.qpdf_vacuum_card(cards[2], ["prod_vac_ns50"]) is None     # no wrong-coupling fallback


def test_qpdf_bits_by_card_round_trip(scratch, tmp_path):
    from htq_hw import campaign as CP
    specs = CP.qpdf_specs("prod_k1.26_s0.75_ns50") + CP.qpdf_specs("prod_vac_ns50")
    names = [s.name for s in specs]
    path = str(tmp_path / "htq_bits_x.npz")
    np.savez(path, job_id="x", backend="b", pub_names=np.array(names),
             **{n: np.zeros((4, 101), np.uint8) for n in names})
    by = A.qpdf_bits_by_card([path])
    assert set(by) == {"prod_k1.26_s0.75_ns50", "prod_vac_ns50"}
    assert len(by["prod_vac_ns50"]) == 21                       # 4 kinds x 5 separations + qZ
    assert "qZ" in by["prod_k1.26_s0.75_ns50"] and "qXYm3" in by["prod_vac_ns50"]


def test_shipped_references_are_present_and_self_describing():
    """A bundle carries its own references: without the surrogate the relA
    anchor silently degrades, and without the qpdf refs there is nothing to
    compare <x> against."""
    import pathlib
    from htq_hw import circuits as C
    for name, eta in (("wing_surrogate_prod.npz", 1.3), ("wing_surrogate_relA.npz", 2.3)):
        p = pathlib.Path(C.REF_DIR / name)
        assert p.exists(), f"{name} is not shipped in htq_hw/refs"
        sur = A.load_wing_surrogate(str(p), eta=eta)
        assert sur["couplings"][2] == eta and sur["times"].max() >= 8.0
        with pytest.raises(ValueError):
            A.load_wing_surrogate(str(p), eta=eta + 1.0)
    z = np.load(C.REF_DIR / "qpdf_card_refs.npz", allow_pickle=True)
    assert {f"{t}_s{w}_x" for t in ("prod", "relA") for w in ("0.75", "1.00", "1.50")} <= set(z.files)


# ---- components a card can actually produce --------------------------------

@pytest.mark.parametrize("fams,want", [
    ({"j0"}, ["00", "10"]),                       # a vacuum card: no J1 insertion, ever
    ({"j0", "j1p1", "j1p2"}, ["00", "10", "01", "11"]),
    ({"j1p1", "j1p2"}, ["01", "11"]),
    (set(), []),
])
def test_components_for(fams, want):
    assert A.components_for(fams) == want


def test_analyze_drops_components_the_pubs_cannot_build(tmp_path):
    """The defect the acceptance rehearsal caught: analysing a vacuum card
    with the default four components tried to load ideal_j1p1.npz, which a
    j0-only card has never had and never will."""
    from htq_hw import campaign as CP
    lat = Lattice(6)
    specs = [s for s in CP.manifest(times=(0.5,), families=("j0",), readouts=("Z",))]
    names = [s.name for s in specs]
    bits = str(tmp_path / "htq_bits_v.npz")
    np.savez(bits, job_id="v", backend="b", pub_names=np.array(names),
             **{n: np.zeros((8, lat.n_wires), np.uint8) for n in names})
    tpl = str(tmp_path / "ideal_{family}.npz")          # only j0 will exist
    from htq_hw import sim as S
    S.write_ideal_grids(C.load_card(), 6, 2, ("j0",), [0.0, 0.5], tpl, threads=2)
    out = A.analyze([bits], tpl, str(tmp_path / "slice_{comp}_t{t:.1f}.npz"), 6, 2,
                    components=["00", "10", "01", "11"], log=None)
    assert {k[0] for k in out} <= {"00", "10"}          # no j1p1 grid was demanded
    with pytest.raises(ValueError, match="can be built"):
        A.analyze([bits], tpl, str(tmp_path / "x_{comp}_t{t:.1f}.npz"), 6, 2,
                  components=["01", "11"], log=None)


def test_card_selection_spans_presets(tmp_path):
    """The production packet is in both prod-bridge and qpdf-scan. Selecting
    on one preset's prefix hides the other's pubs, which is how a full
    rehearsal came back with zero slices: the card's j0 pubs were in
    prod-bridge while the prefix picked qpdf-scan."""
    from htq_hw import campaign as CP
    lat = Lattice(6)
    card = "prod_k1.26_s0.75_ns50"
    names = ([f"prod-bridge.{card}:j0_t0.0_Z", f"prod-bridge.{card}:j0_t0.5_Z"]
             + [f"qpdf-scan.{card}:qpdf_t0.0_qXXm1", f"qpdf-scan.{card}:qpdf_t0.0_qZ"]
             + [f"vac-w00.prod_vac_ns50:j0_t0.0_Z"])
    p = str(tmp_path / "htq_bits_m.npz")
    np.savez(p, job_id="m", backend="b", pub_names=np.array(names),
             **{n: np.zeros((4, lat.n_wires), np.uint8) for n in names})
    one = A.load_job_bits(p, lat, prefix=f"qpdf-scan.{card}:")
    assert set(one.bits) == {"qpdf_t0.0_qXXm1", "qpdf_t0.0_qZ"}      # one preset only
    both = A.load_job_bits(p, lat, card=card)
    assert set(both.bits) == {"j0_t0.0_Z", "j0_t0.5_Z", "qpdf_t0.0_qXXm1", "qpdf_t0.0_qZ"}
    assert "j0_t0.0_Z" in A.load_job_bits(p, lat, card="prod_vac_ns50").bits
    assert len(A.load_job_bits(p, lat, card="prod_vac_ns50").bits) == 1


# ---- plain-evolution probe one-point ---------------------------------------

def test_calib_prefers_the_plain_pub_and_falls_back(tmp_path):
    """B_v should come from the insertion-free circuit when the job carries
    one. The Hadamard marginal is (B_uninserted + B_inserted)/2, which is
    harmless for a packet but wrong for the vacuum by more than the signal."""
    from htq_hw import campaign as CP
    from htq_hw import sim as S
    lat = Lattice(6)
    tpl = str(tmp_path / "ideal_{family}.npz")
    S.write_ideal_grids(C.load_card(), 6, 2, ("j0",), [0.0, 0.5], tpl, threads=2)
    grid = A.IdealGrid(tpl.format(family="j0"))
    assert grid.b_plain(0.0) is not None and grid.b_plain(0.5) is not None
    assert grid.b_plain(0.5).shape == grid.b_ideal(0.5).shape

    def job_with(names):
        bits = {n: np.zeros((16, lat.n_wires), np.uint8) for n in names}
        return A.JobBits(bits, lat, "j", A.ETA)

    had = [A.pub_name(f, t, m, "Z", "X", A.DT)
           for f in ("j0",) for t, m in ((0.0, False), (0.5, False), (0.5, True))]
    plain = [A.pub_name("plain", t, m, "Z", "X", A.DT)
             for t, m in ((0.0, False), (0.5, False), (0.5, True))]

    both = A._calib_J0(job_with(had + plain), "j0", 0.5, grid, grid, 2, wing=False)
    assert both is not None and both["b_source"] == "plain"
    only = A._calib_J0(job_with(had), "j0", 0.5, grid, grid, 2, wing=False)
    assert only is not None and only["b_source"] == "marginal"      # old data still analyses
