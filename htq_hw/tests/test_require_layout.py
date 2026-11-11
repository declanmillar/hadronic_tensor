"""A matched diagnostic must stop before submission if its layout changes."""

import json
from types import SimpleNamespace

import pytest

from htq_hw import __main__ as cli
from htq_hw import acceptance as AC
from htq_hw import campaign as CP
from htq_hw import target as T
from htq_hw.model import Lattice


class ReachedBuild(Exception):
    """The layout guard passed; stop before any circuit building or hardware."""


@pytest.fixture
def layout_cli(monkeypatch, tmp_path):
    backend = SimpleNamespace(name="ibm_test")
    lat = Lattice(4)
    emb = T.Embedding("ladder", 4, 1, list(range(9)))
    record = {
        "kind": "acceptance", "level": "full", "status": "PASS", "failures": [],
        "target": {"name": backend.name, "is_real": True}, "ns": lat.ns, "basis": "rzz",
        "steps": [{"name": "cards:prod_k1.26_s0.75_ns50" if name == "cards" else name,
                   "status": "PASS"} for name in AC.STEPS_FULL],
        "embedding": {"kind": emb.kind, "layout": list(emb.layout)},
        "options": {"pre_twirl": 0},
    }
    path = tmp_path / "acceptance_control.json"
    monkeypatch.setattr(cli, "check_versions", lambda: [])
    monkeypatch.setattr(cli, "_setup", lambda *a, **k: (None, lat, emb.center, backend, emb, []))
    monkeypatch.setattr(T, "is_real_backend", lambda be: be is backend)
    monkeypatch.setattr(CP, "build_pub_circuits", lambda *a, **k: pytest.fail("built before guard passed"))
    monkeypatch.setattr(CP, "submit", lambda *a, **k: pytest.fail("unexpected hardware submission"))

    def run(pre_twirl=0, require=True, raw=None):
        path.write_text(json.dumps(record) if raw is None else raw)
        args = ["--ns", "4", "--pre-twirl", str(pre_twirl), "submit", "--target", backend.name,
                "--basis", "rzz", "--real", "--confirm", backend.name]
        if require:
            args += ["--require-layout", str(path)]
        return cli.main(args)

    return record, emb, run


def test_changed_layout_aborts_before_build_or_submission(layout_cli):
    _, emb, run = layout_cli
    # Changing a noncentral wire is enough: centre and ancilla alone do not
    # establish the matched physical embedding needed by the comparison.
    emb.layout[0], emb.layout[1] = emb.layout[1], emb.layout[0]
    with pytest.raises(SystemExit, match="selected embedding differs"):
        run()


@pytest.mark.parametrize("pre_twirl", [0, 4])
@pytest.mark.parametrize("status", ["PASS", "WARN"])
def test_matching_layout_allows_control_and_twirled_build(layout_cli, monkeypatch, pre_twirl, status):
    record, _, run = layout_cli
    record["status"] = status
    if status == "WARN":
        record["steps"].append({"name": "git", "status": "WARN"})
        record["warnings"] = ["git: uncommitted changes"]

    def build(*args, **kwargs):
        raise ReachedBuild

    monkeypatch.setattr(CP, "build_pub_circuits", build)
    with pytest.raises(ReachedBuild):
        run(pre_twirl=pre_twirl)


@pytest.mark.parametrize("change", [
    lambda r: r.update(kind="check"),
    lambda r: r.update(level="fast"),
    lambda r: r.update(status="FAIL"),
    lambda r: r.update(failures=["runtime: failed"]),
    lambda r: r.pop("failures"),
    lambda r: r["steps"].pop(),
    lambda r: r["steps"].pop(AC.STEPS_FULL.index("cards")),
    lambda r: r["steps"][0].update(status="FAIL"),
    lambda r: r.update(steps=[None]),
    lambda r: r["target"].update(name="ibm_elsewhere"),
    lambda r: r["target"].update(is_real=False),
    lambda r: r.update(ns=6),
    lambda r: r.update(basis="cz"),
    lambda r: r["embedding"].update(kind="grid"),
    lambda r: r["embedding"].update(layout=None),
    lambda r: r["embedding"].update(layout=[0] * 9),
    lambda r: r["embedding"].update(layout=list(range(8))),
    lambda r: r["embedding"].update(layout=[False] + list(range(1, 9))),
])
def test_invalid_or_incompatible_acceptance_aborts_before_build(layout_cli, change):
    record, _, run = layout_cli
    change(record)
    with pytest.raises(SystemExit, match="refusing submission"):
        run()


@pytest.mark.parametrize("raw", ["{invalid json", "[]"])
def test_malformed_record_aborts_before_build(layout_cli, raw):
    _, _, run = layout_cli
    with pytest.raises(SystemExit, match="refusing submission"):
        run(raw=raw)


def test_default_submit_does_not_require_an_acceptance_record(layout_cli, monkeypatch):
    record, _, run = layout_cli
    record.clear()

    def build(*args, **kwargs):
        raise ReachedBuild

    monkeypatch.setattr(CP, "build_pub_circuits", build)
    with pytest.raises(ReachedBuild):
        run(require=False)
