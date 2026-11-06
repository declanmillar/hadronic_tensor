"""An Im-only hardware manifest still needs X physics in local rehearsal."""

import json

import numpy as np
import pytest

from htq_hw import acceptance as AC
from htq_hw import analyze as A
from htq_hw import campaign as CP
from htq_hw import circuits as C
from htq_hw import sim as S


def test_im_rehearsal_adds_real_companions_without_changing_the_campaign():
    specs = CP.compose_presets(["prod-im", "relA-im"])
    original = tuple(specs)
    selected = [s for s in specs if s.t in (0.0, 0.5)]
    rehearsal = AC._rehearse_specs(specs, [0.0, 0.5])

    assert tuple(specs) == original and len(specs) == 372
    assert len(selected) == 42 and len(rehearsal) == 66
    assert all(s in rehearsal for s in selected)
    assert len({s.name for s in rehearsal}) == len(rehearsal)
    added = [s for s in rehearsal if s not in selected]
    assert all(s.anc_basis == "X" and not s.mirror for s in added)
    assert {(s.card, s.family, s.t, s.readout, s.dt) for s in added} == {
        (s.card, s.family, s.t, s.readout, s.dt)
        for s in selected if s.anc_basis == "Y" and not s.mirror
    }
    sources = {s.name: s for s in selected}
    for s in added:
        source = sources[CP.pub_name(s.family, s.t, False, s.readout, "Y", s.dt, s.prefix)]
        assert (s.n_steps, s.group, s.preset, s.stretch) == (
            source.n_steps, source.group, source.preset, source.stretch)


@pytest.mark.parametrize("presets,times", [
    (["prod-bridge", "relA-core", "prod-im", "relA-im"], [0.0, 0.5]),
    (["prod-bridge", "relA-core"], [0.0, 0.5]),
    (["gauss-midcircuit"], [0.0, 1.0, 2.0]),
    (["qpdf-scan"], [0.0]),
])
def test_rehearsal_keeps_existing_real_pubs_and_other_presets(presets, times):
    specs = CP.compose_presets(presets)
    assert AC._rehearse_specs(specs, times) == [s for s in specs if s.t in times]


def test_standalone_im_acceptance_analyzes_physical_samples_without_expanding_plan(
        tmp_path, monkeypatch):
    """Exercise the failed acceptance path on 13 wires, with its real analyzer.

    Production building and planning must still see the Y-only manifest;
    only the local rehearsal receives the auxiliary X physics/reference pubs.
    """
    card = C.load_card("prod_k1.26_s0.75_ns50")
    card.update(ns=6, center=2)
    card_dir = tmp_path / "cards" / card["name"]
    card_dir.mkdir(parents=True)
    (card_dir / "card.json").write_text(json.dumps(card))
    ideal_template = str(card_dir / "ideal_{family}.npz")
    S.write_ideal_grids(card, 6, 2, ("j0",), [0.0, 0.5], ideal_template, threads=2)
    monkeypatch.setattr(C, "CARD_DIR", card_dir.parent)
    specs = CP.manifest(times=(0.5,), families=("j0",), readouts=("Z",),
                        im_only=True, card=card["name"], preset="prod-im")
    original = tuple(specs)
    monkeypatch.setattr(CP, "compose_presets", lambda _presets: list(specs))

    built, planned = [], []
    build_impl, plan_impl = CP.build_pub_circuits, CP.shots_plan

    def build(be, lat, default_card, emb, selected, basis, **kwargs):
        built.append(tuple(selected))
        return build_impl(be, lat, default_card, emb, selected, basis, **kwargs)

    def plan(selected, *args, **kwargs):
        planned.append(tuple(selected))
        return plan_impl(selected, *args, **kwargs)

    monkeypatch.setattr(CP, "build_pub_circuits", build)
    monkeypatch.setattr(CP, "shots_plan", plan)
    rehearsal_dir = tmp_path / "rehearsal"
    result = AC.run("grid:4x5", ["prod-im"], ns=6, basis="rzz", level="full",
                    times=[0.0, 0.5], rep_time=62e-6, threads=2,
                    cache_dir=str(tmp_path / "cache"), keep_rehearsal=str(rehearsal_dir),
                    log=lambda _message: None)

    steps = {step["name"]: step for step in result["steps"]}
    for name in ("circuits", "plan", "check", "rehearse"):
        assert steps[name]["status"] == "PASS", steps[name]["detail"]
    assert result["failures"] == []
    assert tuple(specs) == original and built[0] == original and planned == [original]
    assert len(original) == 4 and len(built) == 2 and len(built[1]) == 6
    assert steps["plan"]["minutes"] == pytest.approx(4 * 60000 * 62e-6 / 60)
    assert steps["rehearse"]["real_companion_pubs"] == 2

    with np.load(rehearsal_dir / f"{card['name']}_slice_00_t0.5.npz") as z:
        assert all(key in z.files for key in A.SLICE_KEYS_IM)
        grid = A.IdealGrid(ideal_template.format(family="j0"), expect_ns=6)
        strong = np.abs(grid.sx_ideal0("J0")) > A.KAP_GUARD
        expected = A.FAMILY_COEFF["j0"] * grid.s_ideal("Y", 0.5)
        assert strong.any() and np.max(np.abs(expected[strong])) > 0.01
        measured, error = z["C_cal"][0].imag, z["C_err_im"][0]
        assert np.all(np.isfinite(measured[strong]))
        assert np.all(np.isfinite(error[strong]))
        assert np.median(np.abs(measured - expected)[strong]
                         / np.maximum(error[strong], 1e-12)) < 4.0
