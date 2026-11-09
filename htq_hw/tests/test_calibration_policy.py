"""Calibration cuts and ranking must not treat unavailable errors as zero."""
from datetime import datetime, timezone
from math import isfinite

import pytest
from qiskit.circuit import Measure, Parameter
from qiskit.circuit.library import CZGate, RZZGate
from qiskit.transpiler import CouplingMap, InstructionProperties, Target
from qiskit_ibm_runtime.models import BackendProperties

from htq_hw import target as T


EDGES = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4)]
NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)
INVALID = [float("nan"), float("inf"), -float("inf"), -0.01, 1.01]


def _property(name, value):
    return {"name": name, "value": value, "date": NOW, "unit": ""}


def _properties(readout=0.01, gate_error=0.002):
    return BackendProperties.from_dict({
        "backend_name": "calibration_test", "backend_version": "1",
        "last_update_date": NOW,
        "qubits": [[_property("readout_error", readout if q == 0 else 0.01)]
                   for q in range(5)],
        "gates": [{"gate": "cz", "name": f"cz{u}_{v}", "qubits": [u, v],
                   "parameters": [_property("gate_error", gate_error if (u, v) == EDGES[0] else 0.002)]}
                  for u, v in EDGES],
        "general": [],
    })


def _target(readout=0.01, gate_error=0.002, rzz=False):
    target = Target(num_qubits=5)
    target.add_instruction(CZGate(), {
        edge: InstructionProperties(error=gate_error if edge == EDGES[0] else 0.002)
        for edge in EDGES})
    target.add_instruction(Measure(), {
        (q,): InstructionProperties(error=readout if q == 0 else 0.01) for q in range(5)})
    if rzz:
        # Configured instructions remain supported even when properties are absent.
        target.add_instruction(RZZGate(Parameter("theta")), dict.fromkeys(EDGES))
    return target


class Backend:
    coupling_map = CouplingMap(EDGES + [(v, u) for u, v in EDGES])

    def __init__(self, properties, target=None):
        self._properties = properties
        self.target = _target() if target is None else target

    def properties(self):
        if isinstance(self._properties, Exception):
            raise self._properties
        return self._properties


def _properties_with_t2(t2_us):
    """One T2 per qubit in microseconds (the unit the device reports); None
    leaves that qubit without a T2 entry."""
    return BackendProperties.from_dict({
        "backend_name": "calibration_test", "backend_version": "1",
        "last_update_date": NOW,
        "qubits": [[_property("readout_error", 0.01)]
                   + ([{"name": "T2", "value": v, "date": NOW, "unit": "us"}] if v is not None else [])
                   for v in t2_us],
        "gates": [{"gate": "cz", "name": f"cz{u}_{v}", "qubits": [u, v],
                   "parameters": [_property("gate_error", 0.002)]} for u, v in EDGES],
        "general": [],
    })


def _ladder(ns, center, layout):
    return T.Embedding("ladder", ns, center, list(layout))


def test_core_wires_are_the_hadamard_test_neighbourhood():
    w = T.core_wires(4, 1)
    assert w["centre"] == 2 and w["ancilla"] == 8          # site n -> wire 2n, ancilla 2ns
    assert w["neighbours"] == [0, 4] and w["links"] == [1, 3]
    assert T.core_wires(4, 0)["neighbours"] == [6, 2]        # periodic: site -1 is site 3


def test_core_t2_reads_microseconds_and_flags_missing():
    props = _properties_with_t2([50.0, 200.0, 200.0, 200.0, None, 300.0, 300.0, 300.0, 120.0])
    assert T._t2_us(props, 1) == pytest.approx(200.0)      # the device reports seconds
    core = T.core_t2_us(props, list(range(9)), 4, 1)        # identity layout: wires 0..8
    assert core["centre"] == [2, pytest.approx(200.0)] and core["ancilla"] == [8, pytest.approx(120.0)]
    assert set(core["qubits"]) == {0, 1, 2, 3, 4, 8}
    assert core["min"] is None                              # wire 4 has no T2: the core is not vouched for
    full = T.core_t2_us(props, [0, 1, 2, 3, 5, 6, 7, 4, 8], 4, 1)  # q4 moved out of the core
    assert full["min"] == pytest.approx(50.0)


def test_core_t2_floor_keeps_score_order_among_passing_candidates():
    props = _properties_with_t2([50.0, 200.0, 200.0, 200.0, 180.0, 300.0, 300.0, 300.0, 120.0])
    good = _ladder(4, 1, [1, 2, 3, 4, 5, 6, 7, 0, 8])        # core on 1,2,3,4,5 and ancilla 8
    short = _ladder(4, 1, list(range(9)))                    # core contains q0 (50 us)
    scored = [(1.0, 0, short), (1.1, 1, good)]
    kept = T._core_t2_floor(scored, props, 4, 1, 100.0)
    assert [e for _, _, e in kept] == [good]
    assert short.info["core_t2_us"]["min"] == pytest.approx(50.0)
    loose = T._core_t2_floor(scored, props, 4, 1, 40.0)
    assert [e for _, _, e in loose] == [short, good]        # nothing rejected: score order kept
    with pytest.raises(T.EmbeddingError, match="T2 >= 250") as err:
        T._core_t2_floor(scored, props, 4, 1, 250.0)
    assert "FAIL" in str(err.value) and "centre q2" in str(err.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -5.0])
def test_t2_floor_must_be_positive_microseconds(value):
    with pytest.raises(T.EmbeddingError, match="min_core_t2_us"):
        T._duration_cut(value, "min_core_t2_us")


@pytest.mark.parametrize("properties", [None, RuntimeError("properties unavailable")])
@pytest.mark.parametrize("cut", ["max_readout_error", "max_2q_error"])
def test_requested_cuts_require_backend_properties(properties, cut):
    with pytest.raises(T.EmbeddingError, match="backend properties"):
        T.operational_graph(Backend(properties), basis="cz", **{cut: 0.05})


@pytest.mark.parametrize("value", INVALID)
@pytest.mark.parametrize("cut", [None, 0.05])
def test_readout_cut_rejects_invalid_calibration(value, cut):
    _, report = T.operational_graph(Backend(_properties(readout=value)), basis="cz",
                                   max_readout_error=cut)
    assert 0 in report["dropped_qubits"]


@pytest.mark.parametrize("value", INVALID)
@pytest.mark.parametrize("basis", [None, "cz"])
def test_gate_cut_rejects_invalid_calibration(value, basis):
    _, report = T.operational_graph(Backend(_properties(gate_error=value)), basis=basis,
                                   max_2q_error=0.05)
    assert (0, 1) in report["dropped_edges"]


def test_missing_properties_do_not_make_supported_rzz_unavailable():
    backend = Backend(_properties(), _target(rzz=True))
    assert backend.target.instruction_supported(operation_name="rzz", qargs=(0, 1))
    _, report = T.operational_graph(backend, basis="rzz", max_2q_error=0.05)
    assert (0, 1) in report["dropped_edges"]


def test_genuine_cz_only_target_remains_usable_for_rzz_basis():
    backend = Backend(_properties())
    graph, report = T.operational_graph(backend, basis="rzz", max_2q_error=0.05)
    assert graph.n_edges == len(EDGES)
    assert report["dropped_edges"] == []
    embedding = T.Embedding("ring", 2, 0, list(range(5)))
    assert isfinite(T.score_embedding(embedding, backend, basis="rzz"))


@pytest.mark.parametrize("value", [None, *INVALID, 1.0])
def test_missing_or_invalid_target_gate_error_is_not_free(value):
    backend = Backend(_properties(), _target(gate_error=value))
    embedding = T.Embedding("ring", 2, 0, list(range(5)))
    assert T.score_embedding(embedding, backend, basis="cz") == float("inf")


@pytest.mark.parametrize("value", [None, *INVALID])
def test_missing_or_invalid_target_readout_error_is_not_free(value):
    backend = Backend(_properties(), _target(readout=value))
    embedding = T.Embedding("ring", 2, 0, list(range(5)))
    assert T.score_embedding(embedding, backend, basis="cz") == float("inf")


def test_supported_rzz_without_target_calibration_is_not_free():
    backend = Backend(_properties(), _target(rzz=True))
    embedding = T.Embedding("ring", 2, 0, list(range(5)))
    assert T.score_embedding(embedding, backend, basis="rzz") == float("inf")


@pytest.mark.parametrize("value", INVALID)
@pytest.mark.parametrize("cut", ["max_readout_error", "max_2q_error"])
def test_cuts_are_finite_probabilities(value, cut):
    with pytest.raises(ValueError, match=cut):
        T.operational_graph(Backend(_properties()), basis="cz", **{cut: value})
