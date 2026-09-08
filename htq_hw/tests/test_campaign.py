"""Manifest, shot plan, job grouping, submission/fetch round trips."""

import json
import os

import numpy as np
import pytest

from htq_hw import campaign as CP
from htq_hw import circuits as C
from htq_hw import target as T
from htq_hw import analyze as A
from htq_hw.model import Lattice


def test_manifest_counts():
    specs = CP.manifest()
    assert len(specs) == 12 * 12 + 9
    names = [s.name for s in specs]
    assert len(set(names)) == len(names)
    assert "j0_t0.5_Z" in names and "j0_m0.5_XYA" in names and "j1p1_t1.0_XYB" in names
    assert "j1p2_t0.0_Z" in names and "j0_t6.0_XYB" in names
    assert sum(s.mirror for s in specs) == 36 and all(s.family == "j0" for s in specs if s.mirror)
    assert len(CP.manifest(j1_mirrors=True)) == 153 + 36
    assert len(CP.manifest(dither=True)) == 153 + 12 * 3 + 3
    assert len(CP.manifest(im=True)) == 2 * 153
    assert any(s.name.endswith("_Y") for s in CP.manifest(im=True))
    half = CP.manifest(dt_half=True)
    assert len(half) == 153 + 2 * 12                                # t = 0.5, 1.0 repeated at dt = 0.25
    ctrl = [s for s in half if s.name.endswith("_dt0.25")]
    assert len(ctrl) == 24 and {s.dt for s in ctrl} == {0.25} and {s.t for s in ctrl} == {0.5, 1.0}
    assert all(s.n_steps == int(round(s.t / 0.25)) for s in ctrl) and sum(s.mirror for s in ctrl) == 6
    assert "j0_m0.5_Z_dt0.25" in {s.name for s in ctrl}
    assert {s.group for s in ctrl}.isdisjoint({s.group for s in half if s not in ctrl})
    assert all(s.n_steps == int(round(s.t / s.dt)) for s in half)
    assert len(CP.manifest(dt_half=True, dt_half_times=(1.0,))) == 153 + 12
    assert CP.parse_pub_name("j1p1_m1.0_XYB_Y_dt0.25") == {"family": "j1p1", "t": 1.0, "mirror": True,
                                                         "readout": "XYB", "anc_basis": "Y", "dt": 0.25, "prefix": ""}


def test_group_jobs_pairing():
    specs = CP.manifest()
    jobs = CP.group_jobs(specs, max_pubs=8)
    by = {s.name: s for s in specs}
    assert jobs[0] == [s.name for s in specs if s.t == 0]           # refs in one job
    for job in jobs[1:]:
        assert len(job) <= 8
        groups = {by[n].group for n in job}
        for g in groups:                                           # no group split
            members = [s.name for s in specs if s.group == g]
            assert all(m in job for m in members)
        for n in job:
            s = by[n]
            if s.mirror:
                continue
            mirror = CP.pub_name("j0", s.t, True, s.readout, s.anc_basis)
            assert mirror in job                                   # mirror rides with its physics
    assert sum(len(j) for j in jobs) == len(specs)
    big = CP.group_jobs(CP.manifest(j1_mirrors=True), max_pubs=8)
    assert all(len(j) <= 8 or j == big[0] for j in big)


def _small_pubs(basis="cz", ns=6, times=(0.5, 1.0), target="grid:4x5"):
    card = C.load_card()
    lat = Lattice(ns)
    center = ns // 2 - 1
    be = T.resolve_backend(target, fractional=(basis == "rzz"))
    emb = T.choose_embedding(be, ns, center)
    specs = CP.manifest(times=times)
    pubs, info, bundles = CP.build_pub_circuits(be, lat, card, emb, specs, basis)
    return card, lat, center, be, emb, specs, pubs, info, bundles


def test_pub_circuits_and_shots_plan():
    card, lat, center, be, emb, specs, pubs, info, bundles = _small_pubs()
    assert set(pubs) == {s.name for s in specs}
    for s in specs:
        assert pubs[s.name].count_ops()["measure"] == lat.n_wires
        assert info[s.name]["clbit_to_logical"] == list(range(lat.n_wires))
        assert info[s.name]["basis_map"][str(lat.ancilla)] == "X"
    for g in {s.group for s in specs if s.group != "refs"}:
        hashes = {info[s.name]["skeleton_block"] for s in specs if s.group == g}
        assert len(hashes) == 1                                    # physics and mirror share the block skeleton
    n2q = {n: info[n]["n2q"] for n in pubs}
    plan = CP.shots_plan(specs, n2q, budget_minutes=2.0, rep_time_s=250e-6, mirror_floor=30000)
    budget = 2.0 * 60 / 250e-6
    assert abs(plan["total"] - budget) < 8 * len(specs)
    assert all(plan["shots"][s.name] >= 30000 for s in specs if s.mirror)
    kap = plan["kappa"]
    assert kap["j0_t1.0_Z"] < kap["j0_t0.5_Z"] < kap["j0_t0.0_Z"]
    assert plan["shots"]["j0_t1.0_Z"] > plan["shots"]["j0_t0.5_Z"]      # 1/kappa^2 weighting
    assert plan["minutes"][4e-3] == pytest.approx(plan["total"] * 4e-3 / 60)
    txt = CP.format_plan(plan)
    assert "min at 250 us" in txt and "Nighthawk" in txt
    # dt-half control rows are visible in the plan with their own cost
    specs_h = CP.manifest(times=(0.5, 1.0), dt_half=True, dt_half_times=(0.5,))
    pubs_h, info_h, _ = CP.build_pub_circuits(be, lat, card, emb, specs_h, "cz")
    plan_h = CP.shots_plan(specs_h, {n: info_h[n]["n2q"] for n in pubs_h}, budget_minutes=2.0)
    rows = {(r["t"], r["dt"]): r for r in plan_h["rows"]}
    assert (0.5, 0.25) in rows and rows[(0.5, 0.25)]["n2q"] > rows[(0.5, 0.5)]["n2q"]
    assert abs(plan_h["total"] - budget) < 8 * len(specs_h)


def test_default_kappa_model_reproduces_measured():
    n2q_by_t = {0.5: 1300, 1.0: 1750, 1.5: 2200, 2.0: 2650, 3.0: 3550}
    model, (a, b) = CP.default_kappa_model(n2q_by_t)
    assert b < 0
    for t, k in CP.KINGSTON_KAPPA.items():
        assert abs(model(n2q_by_t[t]) - k) < 0.05


def test_local_submit_fetch_round_trip_fakeboston(tmp_path):
    """qiskit-ibm-runtime local testing mode on FakeBoston at Ns=4: submit
    the grouped jobs, fetch, bit arrays in the logical column order."""
    card, lat, center, be, emb, specs, pubs, info, bundles = _small_pubs(ns=4, times=(0.5,), target="fake:boston")
    shots = {n: 64 for n in pubs}
    jobs = CP.group_jobs(specs, max_pubs=8)
    recs = CP.submit(be, pubs, info, shots, jobs, lat, emb, "cz", str(tmp_path), tag="test")
    assert len(recs) == len(jobs)
    for r in recs:
        meta = json.load(open(r["path"]))
        assert meta["pub_names"] == r["meta"]["pub_names"] and meta["n_logical"] == lat.n_wires
        assert meta["embedding"] == emb.kind and meta["initial_layout"] == emb.layout
        assert all(meta["pubs"][n]["skeleton_full"] for n in meta["pub_names"])
        path = CP.fetch(r["job"], meta, str(tmp_path))
        z = np.load(path, allow_pickle=True)
        for n in meta["pub_names"]:
            assert z[n].shape == (64, lat.n_wires) and z[n].dtype == np.uint8
    legacy = CP.to_legacy_losch_bits(path) if "j0_t0.5_Z" in z.files else CP.to_legacy_losch_bits(
        CP.fetch(recs[1]["job"], recs[1]["meta"], str(tmp_path)))
    assert set(legacy) >= {"t0.5", "m0.5"}
    assert np.array_equal(legacy["t0.5"][:, ::-1], np.load(
        os.path.join(str(tmp_path), f"htq_bits_{recs[1]['meta']['job_id']}.npz"))["j0_t0.5_Z"])


def test_fetch_bit_order_vs_statevector():
    """Noiseless local sampling (SamplerV2 on Aer) at Ns=4: per-shot
    estimators from the fetched bits agree with Statevector expectation
    values of the logical circuit."""
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from htq_hw import sim as S

    card, lat, center, be, emb, specs, pubs, info, bundles = _small_pubs(ns=4, times=(0.5,), target="grid:4x5")
    names = ["j0_t0.5_Z", "j0_t0.5_XYA", "j0_m0.5_Z"]
    shots = {n: 4000 for n in names}
    meta_recs = CP.submit(AerSimulator(), pubs, info, shots, [names], lat, emb, "cz",
                          out_dir="/tmp/claude-1000/-home-hlamm-Desktop-QC-hadronic-tensor/4a179643-b4b8-42be-b43f-ffcd49ec9721/scratchpad/htq_tests")
    path = CP.fetch(meta_recs[0]["job"], meta_recs[0]["meta"],
                    "/tmp/claude-1000/-home-hlamm-Desktop-QC-hadronic-tensor/4a179643-b4b8-42be-b43f-ffcd49ec9721/scratchpad/htq_tests")
    z = np.load(path, allow_pickle=True)
    obs = S.probe_observables(lat, ("J0", "J1"), card["couplings"]["eta"])
    for name, setting in (("j0_t0.5_Z", "Z"), ("j0_t0.5_XYA", "XYA")):
        est = A.estimate_probes(z[name], lat, setting, card["couplings"]["eta"])
        # logical reference: prep + gadget + block (numeric), no readout
        logical = C.base_circuit(lat, card, "J0", center=center, accumulate="ladder")
        logical.compose(C.trotter_block(lat, 1, 0.5, n_wires=lat.n_wires), inplace=True)
        sv = Statevector(logical)
        if setting == "Z":
            ref = np.array([sv.expectation_value(obs[f"XB_{v}"]).real for v in range(lat.ns)])
            assert np.abs(est["xJ0"] - ref).max() < 0.05
            assert abs(est["xa"] - sv.expectation_value(obs["X"]).real) < 0.05
            assert np.abs(est["gauss"] - 1.0).max() < 1e-12          # Gauss law exact in the noiseless sample
        else:
            for b in range(lat.ns):
                k = est["term"][b]
                ref = sv.expectation_value(obs[f"XT{k}_{b}"]).real
                assert abs(est["xT"][b] - ref) < 0.06
