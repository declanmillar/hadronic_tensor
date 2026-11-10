"""The mirror diagnostic knobs: --mirror-eps, --twirl-gates and the mirror-diag preset.

Written after the 6 November 2026 ibm_chicago rzz run, whose 1e-8 mirrors lost
their ancilla correlation at the odd sites around the insertion while the
physics circuits damped normally.  The diagnostic reruns the mirror with gate
twirling on and with a larger mirror angle; both must be settable from the
command line, recorded with the jobs, and must leave the skeleton invariant.
"""

import argparse

import pytest

import htq_hw
from htq_hw import MIRROR_EPS
from htq_hw import campaign as CP
from htq_hw import circuits as C
from htq_hw import target as T
from htq_hw.__main__ import _twirl_gates, main
from htq_hw.model import Lattice


@pytest.fixture
def restore_eps():
    yield
    htq_hw.set_mirror_eps(MIRROR_EPS)


def test_set_mirror_eps_rebinds_every_module_and_validates(restore_eps):
    from htq_hw import sim as S
    assert htq_hw.set_mirror_eps(0.05) == 0.05
    for mod in (htq_hw, C, CP, T, S):
        assert mod.MIRROR_EPS == 0.05
    for bad in (0.0, -1e-3, 0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            htq_hw.set_mirror_eps(bad)
    htq_hw.set_mirror_eps(MIRROR_EPS)
    assert T.MIRROR_EPS == MIRROR_EPS == 1e-8


def test_mirror_diag_preset_is_seven_j0_z_pubs_on_the_production_card():
    specs = CP.preset_specs("mirror-diag")
    assert len(specs) == 7
    assert {s.card for s in specs} == {"prod_k1.26_s0.75_ns50"}
    assert all(s.family == "j0" and s.readout == "Z" and s.anc_basis == "X" for s in specs)
    assert [s.t for s in specs if s.t == 0] == [0.0]
    assert sorted(s.t for s in specs if not s.mirror and s.t > 0) == list(CP.MIRROR_DIAG_TIMES)
    assert sorted(s.t for s in specs if s.mirror) == list(CP.MIRROR_DIAG_TIMES)
    assert all(s.name.startswith("mirror-diag.prod_k1.26_s0.75_ns50:") for s in specs)
    assert CP.compose_presets(["mirror-diag"]) == specs
    assert len(CP.group_jobs(specs, 8)) <= 2


def test_mirror_angle_reaches_every_block_and_keeps_the_skeleton(scratch, restore_eps):
    lat = Lattice(6)
    be = T.resolve_backend("grid:4x5")
    emb = T.choose_embedding(be, 6, 2)
    card = C.load_card()
    htq_hw.set_mirror_eps(0.05)
    b = T.transpile_bundle(be, lat, card, emb, (1, 2), "cz", "J0", T.SEED, None,
                           tails=((1, 0.25),))
    for n in (1, 2):
        blk = b.blocks[n]
        assert blk["mirror_eps"] == 0.05
        per_step = blk["mode"] == "composed"
        assert blk["mirror"] == C.assign(blk["parametric"], "t", 0.05 / n if per_step else 0.05)
        assert T.skeleton(blk["mirror"]) == T.skeleton(blk["physics"])
        phys, mir = T.assign_block(b, n, n * 0.5)
        assert mir == blk["mirror"] and phys == blk["physics"]
        _, mir_explicit = T.assign_block(b, n, n * 0.5, eps=1e-8)
        assert mir_explicit != mir
    tail = b.blocks[(1, 0.25)]
    assert tail["mirror_eps"] == 0.05
    assert T.skeleton(tail["mirror"]) == T.skeleton(tail["physics"])
    # the default argument follows the setting, it is not frozen at import
    assert T.tail_block(b, be, 1, 0.25)["mirror"] == tail["mirror"]
    htq_hw.set_mirror_eps(MIRROR_EPS)
    assert T.tail_block(b, be, 1, 0.25)["mirror"] != tail["mirror"]


def test_check_validates_a_raised_mirror_angle_against_the_logical_mirror(tmp_path, monkeypatch,
                                                                           restore_eps):
    """At 1e-8 the mirror is checked against the ideal t = 0 row.  At a raised
    angle it is a real evolution to t = eps, with no grid row, so the ISA mirror
    is checked against the logical mirror block on the same gadget state; the
    physics check is unchanged and still reads the grid."""
    import os
    from htq_hw import sim as S
    card = C.load_card()
    lat = Lattice(6)
    tpl = os.path.join(str(tmp_path), "ideal6e_{family}.npz")
    S.write_ideal_grids(card, 6, 2, ("j0",), [0.0, 0.5], tpl, threads=2)
    be = T.resolve_backend("grid:4x5")
    emb = T.choose_embedding(be, 6, 2)
    htq_hw.set_mirror_eps(0.05)
    res = S.check(be, lat, card, emb, tpl, [0.0, 0.5], families=("j0",), threads=2, tol=5e-3, log=None)
    assert max(res.values()) < 5e-3 and ("j0", "mirror t=0.5") in res
    assert S.LAST_GROUPS["j0 mirror t=0.5"]           # reported like every other circuit
    # the same mirror against the t = 0 row is off by the evolution to t = 0.05
    monkeypatch.setattr(S, "MIRROR_EPS_IDENTITY", 1.0)      # force the t = 0 comparison
    with pytest.raises(AssertionError, match="check FAILED"):
        S.check(be, lat, card, emb, tpl, [0.0, 0.5], families=("j0",), threads=2, tol=5e-3, log=None)


@pytest.mark.parametrize("fractional,flag,pre,expected", [
    (False, None, 0, True), (True, None, 0, False),
    (False, False, 0, False), (True, False, 0, False), (False, True, 0, True),
    (False, None, 4, False), (True, None, 4, False), (True, False, 4, False),
])
def test_gate_twirling_rule(fractional, flag, pre, expected):
    assert CP.gate_twirling(fractional, flag, pre) is expected


def test_gate_twirling_refuses_what_the_runtime_refuses():
    """ibm_chicago, 8 Nov 2026: the Sampler's gate twirling on rzz circuits
    passed the acceptance and failed at submission (error 1519)."""
    with pytest.raises(ValueError, match="1519"):
        CP.gate_twirling(True, True)
    with pytest.raises(ValueError, match="exclusive"):
        CP.gate_twirling(False, True, 4)
    with pytest.raises(SystemExit, match="1519"):
        main(["--twirl-gates", "on", "plan", "--target", "grid:4x5", "--basis", "rzz"])
    with pytest.raises(SystemExit, match="K >= 2"):
        main(["--pre-twirl", "1", "plan", "--target", "grid:4x5", "--basis", "rzz"])


def test_twirl_names_and_shot_split():
    assert CP.twirl_name("j0_t2.0_Z", 3) == "j0_t2.0_Z#3"
    assert CP.split_twirl_name("j0_t2.0_Z#3") == ("j0_t2.0_Z", 3)
    assert CP.split_twirl_name("mirror-diag.prod_k1.26_s0.75_ns50:j0_m2.0_Z#0") == \
        ("mirror-diag.prod_k1.26_s0.75_ns50:j0_m2.0_Z", 0)
    assert CP.split_twirl_name("j0_t2.0_Z") == ("j0_t2.0_Z", None)
    assert CP.split_shots(60000, 4) == [15000] * 4
    assert CP.split_shots(10, 4) == [3, 3, 2, 2]
    with pytest.raises(ValueError):
        CP.split_shots(3, 4)


def test_pre_twirl_job_keeps_skeletons_and_draws_reproducibly():
    from htq_hw.tests.test_campaign import _small_pubs
    card, lat, center, be, emb, specs, pubs, info, bundles = _small_pubs(basis="rzz", ns=6, times=(0.5,),
                                                                          target="grid:4x5")
    names = [s.name for s in specs]
    shots = {n: 100 for n in names}
    insts = CP.pre_twirl_job(pubs, info, shots, names, 3, T.SEED, job_index=5)
    assert [i["instance"] for i in insts] == [0, 1, 2]
    for inst in insts:
        assert inst["names"] == [CP.twirl_name(n, inst["instance"]) for n in names]
        for n in names:
            tn = CP.twirl_name(n, inst["instance"])
            assert T.skeleton(inst["pubs"][tn]) == T.skeleton(pubs[n])
            assert inst["pubs"][tn].count_ops().get("rzz", 0) == pubs[n].count_ops().get("rzz", 0)
            assert inst["info"][tn]["twirl_of"] == n and inst["info"][tn]["twirl_instances"] == 3
            assert inst["info"][tn]["skeleton_full"] == info[n]["skeleton_full"]
    assert [sum(i["shots"][CP.twirl_name(names[0], i["instance"])] for i in insts)] == [100]
    # instances differ from each other, and a resumed job re-draws the same ones
    assert insts[0]["pubs"][CP.twirl_name(names[0], 0)] != insts[1]["pubs"][CP.twirl_name(names[0], 1)]
    again = CP.pre_twirl_job(pubs, info, shots, names, 3, T.SEED, job_index=5)
    assert again[1]["pubs"][CP.twirl_name(names[0], 1)] == insts[1]["pubs"][CP.twirl_name(names[0], 1)]
    with pytest.raises(ValueError):
        CP.pre_twirl_job(pubs, info, shots, names, 1, T.SEED)


def test_pre_twirled_submission_round_trip(tmp_path):
    """Local testing mode on FakeNighthawk, rzz, two instances: K instance jobs per
    job, one twirl group each; fetch merges a group back under the pubs' own
    names with the shots summed, and load_job_bits reads it unchanged."""
    import json
    import numpy as np
    from htq_hw import analyze as A
    from htq_hw.tests.test_campaign import _small_pubs
    card, lat, center, be, emb, specs, pubs, info, bundles = _small_pubs(basis="rzz", ns=6, times=(0.5,),
                                                                          target="fake:nighthawk")
    shots = {n: 64 for n in pubs}
    jobs = CP.group_jobs(specs, max_pubs=8)
    recs = CP.submit(be, pubs, info, shots, jobs, lat, emb, "rzz", str(tmp_path), tag="tw", pre_twirl=2)
    assert len(recs) == 2 * len(jobs)
    groups = {}
    for r in recs:
        meta = json.load(open(r["path"]))
        assert meta["options"]["pre_twirl"] == 2 and meta["options"]["twirling_gates"] is False
        tw = meta["twirl"]
        assert tw["instances"] == 2 and tw["of"] == jobs[meta["job_index"]]
        assert meta["pub_names"] == [CP.twirl_name(n, tw["instance"]) for n in tw["of"]]
        assert all(meta["shots"][n] == 32 for n in meta["pub_names"])
        groups.setdefault(tw["group"], []).append(meta)
    assert len(groups) == len(jobs)
    for g, metas in groups.items():
        assert sorted(m["twirl"]["instance"] for m in metas) == [0, 1] and metas[0]["job_id"] == g
    paths = CP.fetch_all([(r["job"], r["meta"]) for r in recs], str(tmp_path))
    assert len(paths) == len(jobs)
    for path, names in zip(paths, jobs):
        z = np.load(path, allow_pickle=True)
        assert list(z["pub_names"]) == names and len(z["twirl_jobs"]) == 2 and int(z["twirl_instances"]) == 2
        for n in names:
            assert z[n].shape == (64, lat.n_wires) and z[n].dtype == np.uint8
        jb = A.load_job_bits(path, lat)
        assert set(jb.bits) == set(names)
    # a single instance record finds its siblings' records on disk (their jobs
    # are then fetched by id), and a group with an instance missing is refused
    one = recs[0]
    metas = CP.twirl_group_metas(one["meta"], str(tmp_path))
    assert [m["twirl"]["instance"] for m in metas] == [0, 1]
    assert metas[0]["job_id"] == one["meta"]["twirl"]["group"] and metas[1]["job_id"] == recs[1]["meta"]["job_id"]
    with pytest.raises(ValueError, match="instances \\[0\\] of 2"):
        CP.fetch(one["job"], one["meta"], str(tmp_path / "again"), meta_dir=str(tmp_path / "nowhere"))


@pytest.mark.parametrize("use_preset", [False, True])
def test_check_passes_on_twirled_rzz_circuits(tmp_path, monkeypatch, use_preset):
    import os
    from htq_hw import sim as S
    card = C.load_card()
    lat = Lattice(6)
    tpl = os.path.join(str(tmp_path), "ideal6t_{family}.npz")
    S.write_ideal_grids(card, 6, 2, ("j0",), [0.0, 0.5], tpl, threads=2)
    be = T.resolve_backend("grid:4x5", fractional=True)
    emb = T.choose_embedding(be, 6, 2)
    twirled = []
    original_twirl = T.twirl

    def track_twirl(qc, rng):
        twirled.append(qc)
        return original_twirl(qc, rng)

    monkeypatch.setattr(T, "twirl", track_twirl)
    specs = CP.preset_specs("mirror-diag") if use_preset else None
    res = S.check(be, lat, card, emb, tpl, [0.0, 0.5], families=("j0",), basis="rzz", threads=2,
                  tol=5e-3, log=None, twirl=True, specs=specs)
    assert max(res.values()) < 5e-3 and any(k[-2:] == ("j0", "mirror t=0.5") for k in res)
    # Preset dispatch must check twirled base, physics and mirror circuits too.
    assert len(twirled) == 3


def test_twirl_flag_parses_to_override():
    assert _twirl_gates(argparse.Namespace(twirl_gates="auto")) is None
    assert _twirl_gates(argparse.Namespace(twirl_gates="on")) is True
    assert _twirl_gates(argparse.Namespace(twirl_gates="off")) is False
    assert _twirl_gates(argparse.Namespace()) is None


class _StubJob:
    def __init__(self, jid):
        self._id, self.cancelled, self.usage_estimation = jid, False, None

    def job_id(self):
        return self._id

    def cancel(self):
        self.cancelled = True


def test_submission_records_twirling_and_mirror_angle(tmp_path, monkeypatch, restore_eps):
    seen = {}

    class _Sampler:
        def run(self, pubs):
            return _StubJob(f"loc-{len(seen)}")

    def sampler(mode, fractional, shots_default=None, twirl_gates=None):
        seen["twirl"] = twirl_gates
        return _Sampler()

    monkeypatch.setattr(CP, "_sampler", sampler)
    monkeypatch.setattr(CP, "preflight_runtime", lambda *a, **k: None)
    lat, emb = Lattice(4), T.choose_embedding(T.resolve_backend("grid:4x5"), 4, 1)
    common = dict(pubs={"p0": None}, info={"p0": {"n2q": 1}}, shots={"p0": 100}, jobs=[["p0"]],
                  lat=lat, emb=emb, log=None)
    htq_hw.set_mirror_eps(0.05)
    with pytest.raises(ValueError, match="1519"):          # the Runtime would refuse it
        CP.submit("fake", basis="rzz", out_dir=str(tmp_path / "a"), twirl_gates=True, **common)
    recs = CP.submit("fake", basis="cz", out_dir=str(tmp_path / "a"), twirl_gates=True, **common)
    assert seen["twirl"] is True
    assert recs[0]["meta"]["options"] == {"twirling_gates": True, "twirling_measure": True,
                                           "dd": "XY4", "mirror_eps": 0.05, "pre_twirl": 0}
    assert "twirl" not in recs[0]["meta"]
    recs = CP.submit("fake", basis="rzz", out_dir=str(tmp_path / "b"), **common)
    assert seen["twirl"] is False and recs[0]["meta"]["options"]["twirling_gates"] is False
    recs = CP.submit("fake", basis="cz", out_dir=str(tmp_path / "c"), **common)
    assert seen["twirl"] is True and recs[0]["meta"]["options"]["twirling_gates"] is True
