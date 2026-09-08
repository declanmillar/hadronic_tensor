"""The gate itself: what it must refuse, and what a record must carry.

These are cheap unit tests of the gate's decision logic.  The expensive part
(the actual Ns=50 acceptance run) is a command, not a test; what is tested
here is that every way of shipping something unvalidated is refused.
"""

import argparse

import pytest

from htq_hw import acceptance as AC
from htq_hw import analyze as A
from htq_hw import campaign as CP
from htq_hw import record as R
from htq_hw.__main__ import _require_acceptance


def test_families_needed_includes_j0_for_mirror_calibration():
    specs = CP.compose_presets(["relA-core"])
    cname = next(s.card for s in specs if s.family == "j1p1")
    fams = AC._families_for(specs, cname)
    assert "j1p1" in fams and "j0" in fams          # every mirror calibrates against J0
    assert "qpdf" not in fams                       # qpdf pubs need no ideal grid
    qcard = next((s.card for s in CP.compose_presets(["qpdf-scan"]) if s.family == "qpdf"), None)
    assert AC._families_for(CP.compose_presets(["qpdf-scan"]), qcard) == set()


def test_rehearse_times_are_t0_plus_the_shortest_slice_per_card():
    specs = CP.compose_presets(["relA-core", "prod-bridge"])
    ts = AC._rehearse_times(specs)
    assert ts[0] == 0.0
    for c in {s.card or "" for s in specs}:
        pos = sorted({s.t for s in specs if (s.card or "") == c and s.t > 0})
        if pos:
            assert pos[0] in ts
        # and nothing deep: the rehearsal is a path test, not a physics run
    assert max(ts) <= 1.0


@pytest.mark.parametrize("card,tag", [("relA_k1.26_s0.75_ns50", "relA"),
                                      ("prod_vac_ns50", "prod"),
                                      ("relA_vac_ns50", "relA")])
def test_wing_path_resolves_per_coupling(card, tag):
    assert A.wing_path_for("d/wing_{tag}.npz", card) == f"d/wing_{tag}.npz"
    assert A.wing_path_for(None, card) is None
    assert A.wing_path_for("d/fixed.npz", card) == "d/fixed.npz"


def test_surrogate_refuses_the_wrong_coupling(scratch):
    """A prod surrogate serving a relA card would bias every anchored slice by
    the difference of two breathing amplitudes, silently."""
    import numpy as np
    p = str(scratch / "sur_prod.npz")
    np.savez(p, times=np.array([0.0, 0.5]), even=np.zeros(2), odd=np.zeros(2),
             m0=0.7, g2=1.1, eta=1.3)
    assert A.load_wing_surrogate(p, eta=1.3)["couplings"] == (0.7, 1.1, 1.3)
    with pytest.raises(ValueError, match="coupling specific"):
        A.load_wing_surrogate(p, eta=2.3)
    assert A.load_wing_surrogate(p)["couplings"] == (0.7, 1.1, 1.3)   # unchecked when eta is unknown
    sur, why = AC._surrogate_for("relA_vac_ns50", p, {"relA_vac_ns50": {"couplings": {"eta": 2.3}}})
    assert sur is None and "eta=1.3" in why
    sur, why = AC._surrogate_for("prod_vac_ns50", p, {"prod_vac_ns50": {"couplings": {"eta": 1.3}}})
    assert sur is not None and why == ""


def test_record_round_trip_and_summary(scratch):
    p = str(scratch / "rec.json")
    R.write_record(p, "acceptance", {"level": "full", "steps": [
        {"name": "env", "status": R.PASS, "detail": "pinned"},
        {"name": "plan", "status": R.WARN, "detail": "OVER BUDGET"}],
        "warnings": ["plan: OVER BUDGET"], "failures": []}, R.WARN)
    rec = R.load_record(p)
    assert rec["status"] == R.WARN and rec["env"]["htq_hw"] and "git" in rec
    txt = R.summarize(rec)
    assert "OVER BUDGET" in txt and "acceptance: WARN" in txt


def test_file_hashes_track_content(scratch):
    d = scratch / "pkg"
    d.mkdir(exist_ok=True)
    (d / "a.py").write_text("x = 1\n")
    (d / "skip.pyc").write_text("nope")
    h1 = R.file_hashes(d)
    assert set(h1) == {"a.py"}
    (d / "a.py").write_text("x = 2\n")
    assert R.file_hashes(d)["a.py"] != h1["a.py"]


def _args(path, **kw):
    return argparse.Namespace(acceptance=str(path), ns=50, allow_fast=False,
                              accept_warnings=False, **kw)


def _pkg(scratch, name="tree"):
    d = scratch / name
    d.mkdir(exist_ok=True)
    (d / "mod.py").write_text("v = 1\n")
    return d


def test_bundle_refuses_without_a_record(scratch):
    with pytest.raises(SystemExit, match="no acceptance record"):
        _require_acceptance(_args(scratch / "absent.json"), _pkg(scratch))


def test_bundle_refuses_a_failed_record(scratch):
    d = _pkg(scratch, "failtree")
    p = scratch / "fail.json"
    R.write_record(str(p), "acceptance", {"level": "full", "files": R.file_hashes(d),
                                          "failures": ["cards:relA: grids missing"], "warnings": []},
                   R.FAIL)
    with pytest.raises(SystemExit, match="FAILED"):
        _require_acceptance(_args(p), d)


def test_bundle_refuses_a_fast_record_then_allows_it_explicitly(scratch):
    d = _pkg(scratch, "fasttree")
    p = scratch / "fast.json"
    R.write_record(str(p), "acceptance", {"level": "fast", "files": R.file_hashes(d),
                                          "failures": [], "warnings": []}, R.PASS)
    with pytest.raises(SystemExit, match="level='fast'"):
        _require_acceptance(_args(p), d)
    rec, hashes = _require_acceptance(
        argparse.Namespace(acceptance=str(p), ns=50, allow_fast=True, accept_warnings=False), d)
    assert rec["status"] == R.PASS and set(hashes) == {"mod.py"}


def test_bundle_refuses_a_stale_record(scratch):
    """The defect this exists for: editing a file after acceptance passed and
    shipping the zip anyway."""
    d = _pkg(scratch, "staletree")
    p = scratch / "stale.json"
    R.write_record(str(p), "acceptance", {"level": "full", "files": R.file_hashes(d),
                                          "failures": [], "warnings": []}, R.PASS)
    assert _require_acceptance(_args(p), d)[0]["status"] == R.PASS
    (d / "mod.py").write_text("v = 2\n")
    with pytest.raises(SystemExit, match="1 modified"):
        _require_acceptance(_args(p), d)
    (d / "mod.py").write_text("v = 1\n")
    (d / "new.py").write_text("w = 0\n")
    with pytest.raises(SystemExit, match="1 new"):
        _require_acceptance(_args(p), d)


def test_bundle_refuses_warnings_unless_accepted(scratch):
    d = _pkg(scratch, "warntree")
    p = scratch / "warn.json"
    R.write_record(str(p), "acceptance", {"level": "full", "files": R.file_hashes(d),
                                          "failures": [], "warnings": ["target: offline stand-in"]},
                   R.WARN)
    with pytest.raises(SystemExit, match="offline stand-in"):
        _require_acceptance(_args(p), d)
    rec, _ = _require_acceptance(
        argparse.Namespace(acceptance=str(p), ns=50, allow_fast=False, accept_warnings=True), d)
    assert rec["warnings"] == ["target: offline stand-in"]
