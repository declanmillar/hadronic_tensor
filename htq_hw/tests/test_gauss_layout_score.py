"""The relay coupling must affect selection between otherwise equal ladders."""

from types import SimpleNamespace

import pytest
from qiskit.circuit import Measure
from qiskit.circuit.library import CZGate
from qiskit.transpiler import InstructionProperties, Target

from htq_hw import target as T


def test_relay_coupling_changes_layout_ranking(monkeypatch):
    bad = T.Embedding("ladder", 4, 1, list(range(10)),
                      info={"gauss_ancillas": {2: 9}})
    good = T.Embedding("ladder", 4, 1, list(range(9)) + [10],
                       info={"gauss_ancillas": {2: 9}})
    edges = [(0, 2), (2, 4), (4, 6), (6, 0),
             (1, 2), (3, 4), (5, 6), (7, 0), (8, 2)]
    errors = {edge: 0.002 for edge in edges}
    errors.update({(9, 4): 0.04, (10, 4): 0.003})
    graph = T.Graph(11, errors)
    target = Target(num_qubits=11)
    target.add_instruction(CZGate(), {
        direction: InstructionProperties(error=error)
        for edge, error in errors.items() for direction in (edge, edge[::-1])
    })
    target.add_instruction(Measure(), {
        (q,): InstructionProperties(error=0.0) for q in range(11)
    })
    backend = SimpleNamespace(target=target, name="relay-test")
    bad.validate(graph)
    good.validate(graph)
    monkeypatch.setattr(T, "grid_coordinates", lambda _g: None)
    monkeypatch.setattr(T, "search_ladder", lambda *a, **k: [bad, good])

    selected = T.choose_embedding(backend, 4, 1, mode="ladder", graph=graph,
                                  gauss_sites=[2], basis="cz")
    assert selected.layout[-1] == 10
    assert T.score_embedding(bad, backend, basis="cz") - selected.info["score"] == pytest.approx(0.037)


def test_relay_connection_remains_required_for_validation():
    emb = T.Embedding("ladder", 4, 1, list(range(10)),
                      info={"gauss_ancillas": {"2": 9}})
    edges = [(0, 2), (2, 4), (4, 6), (6, 0),
             (1, 2), (3, 4), (5, 6), (7, 0), (8, 2)]
    with pytest.raises(ValueError, match="1 required edges missing"):
        emb.validate(T.Graph(10, edges))
