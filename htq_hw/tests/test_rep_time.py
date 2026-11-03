"""CLI shot-time assumptions must survive planning without contacting hardware."""

from types import SimpleNamespace

import pytest

from htq_hw import __main__ as cli
from htq_hw import acceptance as AC
from htq_hw import campaign as CP
from htq_hw import target as T


class PlanComplete(Exception):
    """Stop a command after its real planner, before simulation or submission."""


@pytest.fixture
def plan_cli(monkeypatch):
    specs = CP.preset_specs("prod-im")
    pubs = {s.name: object() for s in specs}
    info = {s.name: {"n2q": i + 1} for i, s in enumerate(specs)}
    backend = SimpleNamespace(name="ibm_test")
    actual_plan = cli._plan
    results, estimates = [], []

    def estimate(be, circuit):
        assert be is backend and circuit is pubs[specs[-1].name]
        estimates.append(circuit)
        return {"circuit_us": 40.0, "reset_us": 10.0,
                "overhead_us": 100.0, "total_s": 150e-6}

    def finish_plan(*args):
        results.append(actual_plan(*args))
        raise PlanComplete

    monkeypatch.setattr(cli, "check_versions", lambda: [])
    monkeypatch.setattr(cli, "_setup", lambda *a, **k: (None, None, None, backend, None, specs))
    monkeypatch.setattr(CP, "build_pub_circuits", lambda *a, **k: (pubs, info, None))
    monkeypatch.setattr(T, "backend_label", lambda be: be.name)
    monkeypatch.setattr(T, "per_shot_time", estimate)
    monkeypatch.setattr(cli, "_plan", finish_plan)

    def run(command, rep_time=None, real=True):
        monkeypatch.setattr(T, "is_real_backend", lambda be: real)
        argv = [command, "--target", backend.name, "--preset", "prod-im"]
        if rep_time is not None:
            argv += ["--rep-time", str(rep_time)]
        with pytest.raises(PlanComplete):
            cli.main(argv)
        return results[0], estimates

    return run


@pytest.mark.parametrize("command", ["plan", "rehearse", "submit"])
@pytest.mark.parametrize("rep_time", [62e-6, 250e-6])
def test_explicit_rep_time_survives_real_backend_planning(plan_cli, command, rep_time):
    (plan, rep), estimates = plan_cli(command, rep_time)
    assert rep == rep_time
    assert plan["rep_time_s"] == rep_time
    assert plan["total"] == 6_660_000
    assert plan["minutes"][rep] == pytest.approx(plan["total"] * rep_time / 60)
    assert estimates == []


@pytest.mark.parametrize("command", ["plan", "rehearse", "submit"])
@pytest.mark.parametrize("real", [False, True])
def test_omitted_rep_time_preserves_backend_defaults(plan_cli, capsys, command, real):
    (plan, rep), estimates = plan_cli(command, real=real)
    assert rep == (150e-6 if real else 250e-6)
    assert plan["rep_time_s"] == rep
    assert len(estimates) == int(real)
    output = capsys.readouterr().out
    if real:
        assert "estimated per-shot time" in output
        assert "assumed overhead" in output
        assert "measured per-shot time" not in output


@pytest.mark.parametrize("rep_time", [None, 62e-6, 250e-6])
def test_acceptance_keeps_its_default_and_explicit_rep_time(monkeypatch, rep_time):
    def run(*args, **kwargs):
        assert kwargs["rep_time"] == (250e-6 if rep_time is None else rep_time)
        raise PlanComplete

    monkeypatch.setattr(cli, "check_versions", lambda: [])
    monkeypatch.setattr(AC, "run", run)
    argv = ["acceptance"]
    if rep_time is not None:
        argv += ["--rep-time", str(rep_time)]
    with pytest.raises(PlanComplete):
        cli.main(argv)


@pytest.mark.parametrize("command", ["plan", "rehearse", "submit", "acceptance"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_invalid_rep_time_is_rejected_before_command(monkeypatch, capsys, command, value):
    monkeypatch.setattr(cli, f"cmd_{command}", lambda args: pytest.fail("invalid time reached command"))
    with pytest.raises(SystemExit) as exc:
        cli.main([command, f"--rep-time={value}"])
    assert exc.value.code == 2
    assert "finite, positive" in capsys.readouterr().err
