"""Failed Gauss searches must diagnose the same ancilla requirements."""

from types import SimpleNamespace

import pytest

from htq_hw import target as T


@pytest.mark.parametrize("sites,needed", [(None, 101), ([25], 102)])
def test_failed_layout_diagnostic_keeps_ancillas_and_search_limits(monkeypatch, sites, needed):
    calls = []

    def search(g, ns, center, **kw):
        calls.append((ns, kw))
        if "report" in kw:
            kw["report"].update(stopped_by="exhausted", expansions=100, cycles=0)
        return [object()] if ns == 48 else []

    monkeypatch.setattr(T, "grid_coordinates", lambda g: None)
    monkeypatch.setattr(T, "search_ladder", search)
    with pytest.raises(T.EmbeddingError, match=f"needs {needed} qubits") as raised:
        T.choose_embedding(SimpleNamespace(name="test"), 50, 24, mode="ladder",
                           graph=T.Graph(120, []), gauss_sites=sites,
                           seed=19, max_expansions=250, time_budget_s=12)
    error = raised.value
    assert error.diagnostics["needed_qubits"] == needed
    assert error.diagnostics["gauss_sites"] == (sites or [])
    assert error.diagnostics["largest_feasible_ns"] == 48
    assert error.diagnostics["search"]["stopped_by"] == "exhausted"
    assert "largest Ns found by the diagnostic search is 48" in str(error)
    assert [ns for ns, _ in calls] == [50, 50, 48]
    for _, kw in calls:
        assert kw["gauss_sites"] == sites
        assert kw["seed"] == 19 and kw["max_expansions"] == 250
    assert calls[1][1]["time_budget_s"] == 6


def test_smaller_gauss_diagnostic_does_not_wrap_requested_sites(monkeypatch):
    tried = []

    def search(g, ns, **kw):
        tried.append(ns)
        return [] if ns == 8 else [object()]

    monkeypatch.setattr(T, "search_ladder", search)
    assert T.largest_feasible_ns(T.Graph(20, []), center=6, gauss_sites=[7],
                                 ns_max=8, ns_min=4) is None
    assert tried == [8]
