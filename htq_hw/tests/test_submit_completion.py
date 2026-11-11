"""An incomplete submission must fail the CLI so shell sequences stop."""

from types import SimpleNamespace

import pytest
import qiskit_ibm_runtime

from htq_hw import __main__ as cli
from htq_hw import campaign as CP
from htq_hw import target as T
from htq_hw.model import Lattice


@pytest.fixture
def submit_cli(monkeypatch, tmp_path):
    backend = SimpleNamespace(name="ibm_test", num_qubits=120)
    lat = Lattice(50)
    emb = T.Embedding("ladder", 50, 24, list(range(101)))
    specs = CP.preset_specs("mirror-diag")
    pubs = {spec.name: object() for spec in specs}
    info = {name: {"n2q": 1} for name in pubs}
    service = object()
    calls = []
    monkeypatch.setattr(cli, "check_versions", lambda: [])
    monkeypatch.setattr(cli, "_setup", lambda *a, **k: (None, lat, emb.center, backend, emb, specs))
    monkeypatch.setattr(CP, "build_pub_circuits", lambda *a, **k: (pubs, info, None))
    monkeypatch.setattr(T, "is_real_backend", lambda be: be is backend)
    monkeypatch.setattr(T, "require_real_backend", lambda *a, **k: {
        "name": backend.name, "num_qubits": backend.num_qubits})
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", lambda: service)

    def run(kept, pre_twirl=0, only_jobs=None):
        def submit(*args, **kwargs):
            assert args[0] is backend and args[9] is service
            calls.append((args[4], kwargs))
            return [{"job": object(), "meta": {}, "path": str(tmp_path / f"job_{i}.json")}
                    for i in range(kept)]

        monkeypatch.setattr(CP, "submit", submit)
        args = ["--pre-twirl", str(pre_twirl), "submit", "--target", backend.name,
                "--basis", "rzz", "--preset", "mirror-diag", "--rep-time", "6.2e-5",
                "--real", "--confirm", backend.name, "--out", str(tmp_path)]
        if only_jobs is not None:
            args += ["--only-jobs", *map(str, only_jobs)]
        return cli.main(args)

    return run, calls


@pytest.mark.parametrize("pre_twirl,kept,expected", [
    (0, 0, 2),
    (0, 1, 2),
    (4, 0, 8),
    (4, 1, 8),
    (4, 3, 8),
    (4, 4, 8),  # one whole twirl group still leaves the second group missing
    (4, 7, 8),
])
def test_incomplete_submission_exits_nonzero(submit_cli, pre_twirl, kept, expected):
    run, calls = submit_cli
    with pytest.raises(SystemExit, match=f"{kept} of {expected} job\\(s\\) kept") as error:
        run(kept, pre_twirl)
    assert error.value.code != 0
    jobs, options = calls[0]
    assert len(jobs) == 2
    assert options["strict_guard"] is True
    assert options["estimate_wait"] == CP.ESTIMATE_WAIT_SECONDS


@pytest.mark.parametrize("pre_twirl,kept", [(0, 2), (4, 8)])
def test_complete_submission_returns_normally(submit_cli, pre_twirl, kept):
    run, _ = submit_cli
    assert run(kept, pre_twirl) is None


@pytest.mark.parametrize("pre_twirl,kept", [(0, 1), (4, 4)])
def test_selected_jobs_only_need_the_requested_instances(submit_cli, pre_twirl, kept):
    run, calls = submit_cli
    assert run(kept, pre_twirl, only_jobs=[1]) is None
    jobs, options = calls[0]
    assert len(jobs) == 1 and options["job_indices"] == [1]


def test_incomplete_selected_twirl_group_exits_nonzero(submit_cli):
    run, _ = submit_cli
    with pytest.raises(SystemExit, match="3 of 4 job\\(s\\) kept"):
        run(3, pre_twirl=4, only_jobs=[1])
