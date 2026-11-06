"""Reject invalid Runtime inputs before acceptance passes or any job is sent."""

from types import SimpleNamespace

import pytest
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Parameter

from htq_hw import acceptance as AC
from htq_hw import campaign as CP
from htq_hw import sim as S
from htq_hw import target as T


def _pub(angle=0.2, register="meas"):
    qc = QuantumCircuit(2)
    qc.add_register(ClassicalRegister(2, register))
    qc.rzz(angle, 0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture
def backend():
    return T.resolve_backend("grid:2x2", fractional=True)


@pytest.mark.parametrize("defect,match", [
    ("rzz", "rzz.*outside"),
    ("isa", "instruction h.*not supported"),
    ("register", "Python keywords"),
    ("parameters", "parameters"),
    ("shots", "shots"),
])
def test_runtime_input_failures_include_job_and_pub(backend, defect, match):
    qc, shots = _pub(), 64
    if defect == "rzz":
        qc = _pub(-1.5448765000000009)
    elif defect == "isa":
        qc.h(0)
    elif defect == "register":
        qc = _pub(register="for")
    elif defect == "parameters":
        qc = _pub(Parameter("theta"))
    else:
        shots = 0
    with pytest.raises(ValueError, match=match) as error:
        CP.preflight_runtime(backend, {"late": qc}, {"late": shots}, [["late"]],
                             job_indices=[12])
    assert "job 12, pub 'late'" in str(error.value)


def test_runtime_preflight_checks_faulty_hardware_when_available(backend):
    def faulty(qc):
        raise ValueError("faulty qubit 0")

    be = SimpleNamespace(target=backend.target, check_faulty=faulty)
    with pytest.raises(ValueError, match="job 0, pub 'p'.*faulty qubit 0"):
        CP.preflight_runtime(be, {"p": _pub()}, {"p": 64}, [["p"]])


def test_bad_late_selected_job_prevents_batch_sampler_and_budget_query(
        backend, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid Runtime input reached a batch, sampler or budget query")

    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    for name in ("_batch_ctx", "_sampler", "preflight_budget"):
        monkeypatch.setattr(CP, name, forbidden)
    pubs = {"first": _pub(), "late": _pub(-0.1)}
    with pytest.raises(ValueError, match="job 12, pub 'late'.*rzz"):
        CP.submit(backend, pubs, {}, {name: 64 for name in pubs},
                  [["first"], ["late"]], None, None, "rzz",
                  out_dir=str(tmp_path / "jobs"), est_seconds=1.0,
                  job_indices=[7, 12])
    assert not (tmp_path / "jobs").exists()


def test_submit_ignores_invalid_unselected_pub(backend, tmp_path, monkeypatch):
    sent = []

    class Sampler:
        def run(self, pubs):
            sent.append(pubs)
            return SimpleNamespace(job_id=lambda: "local-test")

    monkeypatch.setattr(CP, "_sampler", lambda *args, **kwargs: Sampler())
    pubs = {"selected": _pub(), "unselected": _pub(-0.1)}
    lat = SimpleNamespace(ns=2, n_wires=5)
    emb = SimpleNamespace(center=0, kind="test", layout=[0, 1])
    records = CP.submit(backend, pubs, {"selected": {"n2q": 1}},
                        {name: 64 for name in pubs}, [["selected"]], lat, emb, "rzz",
                        out_dir=str(tmp_path), job_indices=[7])
    assert len(sent) == len(records) == 1
    assert sent[0] == [(pubs["selected"], None, 64)]
    assert records[0]["meta"]["job_index"] == 7


def test_full_acceptance_rejects_bad_pub_outside_rehearsed_times(monkeypatch):
    specs = CP.manifest(times=(0.5, 6.0), families=("j0",), readouts=("Z",))
    bad = next(s for s in specs if s.t == 6.0 and not s.mirror)
    assert bad.t not in AC._rehearse_times(specs)
    pubs = {s.name: _pub(-0.1 if s is bad else 0.2) for s in specs}
    info = {s.name: {"n2q": 1} for s in specs}
    monkeypatch.setattr(CP, "compose_presets", lambda presets: specs)
    monkeypatch.setattr(CP, "load_cards", lambda *args: {})
    monkeypatch.setattr(CP, "build_pub_circuits", lambda *args, **kwargs: (pubs, info, {}))

    def forbidden(*args, **kwargs):
        pytest.fail("known-invalid Runtime input reached numerical checks or rehearsal")

    monkeypatch.setattr(S, "check", forbidden)
    monkeypatch.setattr(S, "rehearse", forbidden)
    result = AC.run("grid:4x5", ["prod-im"], ns=6, basis="rzz", level="full",
                    log=lambda message: None)
    steps = {step["name"]: step for step in result["steps"]}
    assert steps["circuits"]["status"] == steps["plan"]["status"] == "PASS"
    assert result["status"] == steps["runtime"]["status"] == "FAIL"
    assert len(result["failures"]) == 1 and bad.name in result["failures"][0]
    assert "check" not in steps and "rehearse" not in steps
