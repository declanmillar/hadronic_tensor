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
from htq_hw.__main__ import _twirl_gates
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


@pytest.mark.parametrize("fractional,flag,expected", [
    (False, None, True), (True, None, False),
    (False, False, False), (True, True, True), (True, False, False),
])
def test_gate_twirling_rule_and_cli_mapping(fractional, flag, expected):
    assert CP.gate_twirling(fractional, flag) is expected


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
    recs = CP.submit("fake", basis="rzz", out_dir=str(tmp_path / "a"), twirl_gates=True, **common)
    assert seen["twirl"] is True
    assert recs[0]["meta"]["options"] == {"twirling_gates": True, "twirling_measure": True,
                                           "dd": "XY4", "mirror_eps": 0.05}
    recs = CP.submit("fake", basis="rzz", out_dir=str(tmp_path / "b"), **common)
    assert seen["twirl"] is False and recs[0]["meta"]["options"]["twirling_gates"] is False
    recs = CP.submit("fake", basis="cz", out_dir=str(tmp_path / "c"), **common)
    assert seen["twirl"] is True and recs[0]["meta"]["options"]["twirling_gates"] is True
