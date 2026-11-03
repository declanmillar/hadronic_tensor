"""Diagnostic layouts must use the same calibration cuts as a campaign."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from qiskit.circuit import Parameter
from qiskit.circuit.library import CZGate, Measure, RZZGate
from qiskit.transpiler import CouplingMap, InstructionProperties, Target
from qiskit_ibm_runtime.models import BackendProperties

from htq_hw import __main__ as cli
from htq_hw import target as T


@pytest.fixture
def backends():
    """One bad readout and an edge usable for CZ but above the RZZ cut."""
    edges = [(q, (q + 1) % 6) for q in range(6)]
    pairs = edges + [(v, u) for u, v in edges]
    date = datetime(2026, 11, 2, tzinfo=timezone.utc)
    readout = {q: 0.2 if q == 1 else 0.001 for q in range(6)}
    errors = {
        gate: {pair: 0.1 if gate == "rzz" and set(pair) == {2, 3} else 0.002
               for pair in pairs}
        for gate in ("cz", "rzz")
    }
    props = BackendProperties.from_dict({
        "backend_name": "diagnostic_fixture", "backend_version": "1.0",
        "last_update_date": date,
        "qubits": [[{"name": "readout_error", "value": readout[q], "unit": "", "date": date}]
                   for q in range(6)],
        "gates": [{"gate": gate, "name": f"{gate}_{u}_{v}", "qubits": [u, v],
                   "parameters": [{"name": "gate_error", "value": error, "unit": "", "date": date}]}
                  for gate, rates in errors.items() for (u, v), error in rates.items()],
        "general": [],
    })
    result = {}
    for basis in ("cz", "rzz"):
        target = Target(num_qubits=6)
        target.add_instruction(CZGate(), {pair: InstructionProperties(error=error)
                                         for pair, error in errors["cz"].items()})
        if basis == "rzz":
            target.add_instruction(RZZGate(Parameter("theta")), {
                pair: InstructionProperties(error=error) for pair, error in errors["rzz"].items()})
        target.add_instruction(Measure(), {(q,): InstructionProperties(error=error)
                                           for q, error in readout.items()})
        result[basis] = SimpleNamespace(name=basis, num_qubits=6, target=target,
                                       coupling_map=CouplingMap(pairs), properties=lambda: props)
    return result


def _check_graph(graph, basis):
    assert graph is not None, "diagnostic bypassed the filtered graph"
    assert not graph.adj[0], "explicit qubit exclusion was ignored"
    assert not graph.adj[1], "readout cutoff was ignored"
    assert not graph.has_edge(3, 4), "explicit edge exclusion was ignored"
    assert graph.has_edge(2, 3) == (basis == "cz"), "wrong basis used for edge cutoff"


def _install_selection(monkeypatch, backends, selected):
    monkeypatch.setattr(cli, "check_versions", lambda: [])
    monkeypatch.setattr(T, "resolve_backend", lambda spec, fractional=False: backends["rzz" if fractional else "cz"])

    def choose(be, ns, center, *, graph=None, basis=None, allow_transpiler=False, **kwargs):
        assert basis == be.name
        _check_graph(graph, basis)
        selected.append((basis, allow_transpiler))
        return T.Embedding("transpiler", ns, center, None, info={"basis": basis})

    monkeypatch.setattr(T, "choose_embedding", choose)


FILTER_ARGS = ["--ns", "2", "--exclude-qubits", "0", "--exclude-edges", "3-4",
               "--max-readout-error", "0.05", "--max-2q-error", "0.02", "--allow-transpiler"]


@pytest.mark.parametrize("basis", ["cz", "rzz"])
def test_embed_cli_uses_filtered_graph_and_requested_basis(monkeypatch, backends, basis):
    selected = []
    _install_selection(monkeypatch, backends, selected)
    cli.main(FILTER_ARGS + ["embed", "--targets", "fixture", "--basis", basis])
    assert selected == [(basis, True)]


def test_report_selects_a_filtered_layout_for_each_basis(monkeypatch, backends):
    selected, compiled = [], []
    _install_selection(monkeypatch, backends, selected)

    def transpile(be, lat, card, emb, steps, basis, kind, *args, **kwargs):
        assert be is backends[basis]
        assert emb.info["basis"] == basis
        compiled.append(basis)
        return SimpleNamespace(basis=basis)

    monkeypatch.setattr(T, "transpile_bundle", transpile)
    monkeypatch.setattr(T, "_row", lambda spec, emb, bundle, be: {"basis": bundle.basis})
    rows = T.report(["fixture"], ns=2, steps=(1,), bases=("cz", "rzz"), gadgets=("J0",),
                    exclude_qubits=(0,), exclude_edges=((3, 4),),
                    max_readout_error=0.05, max_2q_error=0.02, log=None)
    assert rows == [{"basis": "cz"}, {"basis": "rzz"}]
    assert selected == [("cz", False), ("rzz", False)]
    assert compiled == ["cz", "rzz"]


def test_report_cli_applies_filters_to_table_and_timing(monkeypatch, backends, capsys):
    selected, estimated = [], []
    _install_selection(monkeypatch, backends, selected)

    def report(targets, **kwargs):
        assert kwargs["exclude_qubits"] == (0,)
        assert kwargs["exclude_edges"] == ((3, 4),)
        assert kwargs["max_readout_error"] == 0.05
        assert kwargs["max_2q_error"] == 0.02
        assert kwargs["allow_transpiler"] is True
        return []

    def transpile(be, lat, card, emb, steps, basis, kind, *args, **kwargs):
        assert be is backends[basis] and emb.info["basis"] == basis
        return SimpleNamespace(base=SimpleNamespace(compose=lambda block: basis),
                               blocks={1: {"physics": None, "layout": None}})

    def estimate(be, circuit):
        assert be.name == circuit
        estimated.append(be.name)
        return {"total_s": 150e-6}

    monkeypatch.setattr(T, "report", report)
    monkeypatch.setattr(T, "format_table", lambda rows: "")
    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    monkeypatch.setattr(T, "transpile_bundle", transpile)
    monkeypatch.setattr(cli.C, "readout_layer", lambda full, *args: full)
    monkeypatch.setattr(T, "per_shot_time", estimate)
    cli.main(FILTER_ARGS + ["report", "--targets", "fixture", "--steps", "1", "--basis", "rzz", "cz"])
    assert selected == [("rzz", True), ("cz", True)]
    assert estimated == ["rzz", "cz"]
    output = capsys.readouterr().out
    assert "estimated per-shot time" in output
    assert "measured per-shot time" not in output
