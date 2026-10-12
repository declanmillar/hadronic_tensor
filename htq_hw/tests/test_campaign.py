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
    # Ns=4 on heavy-hex has no structured embedding (girth 12); this test
    # exercises submit/fetch, so the transpiler layout is explicitly allowed
    emb = T.choose_embedding(be, ns, center, allow_transpiler=True)
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


def test_fetch_bit_order_vs_statevector(scratch):
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
                          out_dir=str(scratch))
    path = CP.fetch(meta_recs[0]["job"], meta_recs[0]["meta"],
                    str(scratch))
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


# ---- budget guard: what a 3-hour allocation is owed ------------------------

class _StubJob:
    def __init__(self, est=None, jid="stub-job"):
        self.usage_estimation = {"quantum_seconds": est} if est is not None else None
        self._id, self.cancelled = jid, False

    def job_id(self):
        return self._id

    def cancel(self):
        self.cancelled = True


class _StubService:
    def __init__(self, remaining):
        self._rem = remaining

    def usage(self):
        if self._rem is None:
            raise RuntimeError("platform did not answer")
        return {"usage_remaining_seconds": self._rem}


@pytest.mark.parametrize("est,rem,want", [
    (1000.0, 10800.0, CP.GUARD_OK),
    (10000.0, 10800.0, CP.GUARD_OVER),          # 93% of what is left, guard 85%
    (1000.0, None, CP.GUARD_UNVERIFIABLE),
])
def test_preflight_states(est, rem, want):
    state, info = CP.preflight_budget(_StubService(rem), est, guard=0.85, log=None)
    assert state == want and info["estimated_s"] == est and info["remaining_s"] == rem


@pytest.mark.parametrize("est,rem,want,cancels", [
    (100.0, 10800.0, CP.GUARD_OK, False),
    (10000.0, 10800.0, CP.GUARD_OVER, True),
    (None, 10800.0, CP.GUARD_UNVERIFIABLE, False),   # unreadable estimate is not "it fits"
    (100.0, None, CP.GUARD_UNVERIFIABLE, False),
])
def test_budget_guard_is_tri_state(est, rem, want, cancels):
    job = _StubJob(est)
    assert CP.budget_guard(job, _StubService(rem), 0.85, log=None, estimate_wait=0) == want
    assert job.cancelled is cancels


@pytest.fixture
def estimate_clock(monkeypatch):
    class _Clock:
        elapsed = 0.0
        strftime = staticmethod(CP.time.strftime)

        def __init__(self):
            self.sleeps = []

        def monotonic(self):
            return self.elapsed

        def sleep(self, seconds):
            assert seconds > 0
            self.sleeps.append(seconds)
            self.elapsed += seconds

    clock = _Clock()
    monkeypatch.setattr(CP, "time", clock)
    return clock


class _RefreshingService(_StubService):
    def __init__(self, remaining, estimates=()):
        super().__init__(remaining)
        self.estimates = list(estimates)
        self.job_requests = []
        self.handles = []

    def job(self, jid):
        self.job_requests.append(jid)
        est = self.estimates.pop(0) if self.estimates else None
        if isinstance(est, Exception):
            raise est
        handle = _StubJob(est=est, jid=jid)
        self.handles.append(handle)
        return handle


def test_budget_guard_refreshes_cached_missing_estimate(estimate_clock):
    original = _StubJob()
    original.usage_estimation = {"quantum_seconds": None}  # Runtime caches this dict
    service = _RefreshingService(10800.0, [None, 100.0])

    assert CP.budget_guard(original, service) == CP.GUARD_OK
    assert estimate_clock.sleeps == [5.0, 5.0]
    assert service.job_requests == [original.job_id()] * 2
    assert all(handle is not original for handle in service.handles)
    assert service.handles[0] is not service.handles[1]
    assert original.usage_estimation == {"quantum_seconds": None}
    assert original.cancelled is False


@pytest.mark.parametrize("est,rem,want", [
    (100.0, 10800.0, CP.GUARD_OK),
    (10000.0, 10800.0, CP.GUARD_OVER),
    (None, None, CP.GUARD_UNVERIFIABLE),
    (100.0, None, CP.GUARD_UNVERIFIABLE),
])
def test_budget_guard_does_not_wait_unnecessarily(est, rem, want, estimate_clock):
    service = _RefreshingService(rem)
    assert CP.budget_guard(_StubJob(est), service) == want
    assert estimate_clock.sleeps == [] and service.job_requests == []


@pytest.mark.parametrize("wait,sleeps", [
    (0.0, []),
    (7.0, [5.0, 2.0]),
    (60.0, [5.0] * 12),
])
def test_budget_guard_missing_estimate_has_bounded_wait(wait, sleeps, estimate_clock):
    job, service = _StubJob(), _RefreshingService(10800.0)
    assert CP.budget_guard(job, service, estimate_wait=wait) == CP.GUARD_UNVERIFIABLE
    assert estimate_clock.sleeps == sleeps and estimate_clock.elapsed == wait
    assert len(service.job_requests) == len(sleeps)
    assert job.cancelled is False  # strict submission owns cancellation on timeout


@pytest.mark.parametrize("estimates,want", [
    ([RuntimeError("estimate pending"), 100.0], CP.GUARD_OK),
    ([RuntimeError("estimate pending")] * 2, CP.GUARD_UNVERIFIABLE),
])
def test_budget_guard_refresh_errors_do_not_reset_deadline(estimates, want, estimate_clock):
    service = _RefreshingService(10800.0, estimates)
    assert CP.budget_guard(_StubJob(), service, estimate_wait=7.0) == want
    assert estimate_clock.sleeps == [5.0, 2.0]
    assert len(service.job_requests) == 2


def test_budget_guard_cancels_an_overbudget_refreshed_estimate(estimate_clock):
    job, service = _StubJob(), _RefreshingService(10800.0, [10000.0])
    assert CP.budget_guard(job, service) == CP.GUARD_OVER
    assert job.cancelled is True and estimate_clock.sleeps == [5.0]


def _submit_args(tmp_path, n_jobs=3):
    lat, emb = Lattice(4), T.choose_embedding(T.resolve_backend("grid:4x5"), 4, 1)
    names = [f"p{i}" for i in range(n_jobs)]
    pubs = {n: None for n in names}
    return dict(pubs=pubs, info={n: {"n2q": 1} for n in names}, shots={n: 100 for n in names},
                jobs=[[n] for n in names], lat=lat, emb=emb, basis="cz", out_dir=str(tmp_path))


def test_real_submission_refuses_an_unfittable_campaign_before_job_zero(tmp_path, monkeypatch):
    """112 jobs that each fit can still overrun together, so the whole campaign
    is priced before the first one is sent."""
    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    monkeypatch.setattr(CP, "_sampler", lambda *a, **k: pytest.fail("submitted despite the guard"))
    kw = _submit_args(tmp_path)
    with pytest.raises(CP.BudgetError, match="only 600 QPU s remain|remain"):
        CP.submit("dev", service=_StubService(600.0), est_seconds=10000.0, batch=False,
                  log=None, **kw)
    with pytest.raises(CP.BudgetError, match="could not be read"):
        CP.submit("dev", service=_StubService(None), est_seconds=10.0, batch=False, log=None, **kw)
    with pytest.raises(CP.BudgetError, match="est_seconds"):
        CP.submit("dev", service=_StubService(10800.0), batch=False, log=None, **kw)
    assert not list(tmp_path.glob("htq_job_*.json"))


def test_guard_stops_the_loop_and_still_records_the_cancelled_job(tmp_path, monkeypatch):
    """The failure this exists for: job 0 is cancelled over budget and the
    other 111 are submitted anyway, with no record of the id that was spent."""
    sent = []

    class _Sampler:
        def run(self, pubs):
            sent.append(pubs)
            return _StubJob(est=10000.0, jid=f"stub-{len(sent) - 1}")

    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    monkeypatch.setattr(CP, "_sampler", lambda *a, **k: _Sampler())
    recs = CP.submit("dev", service=_StubService(10800.0), est_seconds=10.0, batch=False,
                     log=None, **_submit_args(tmp_path))
    assert recs == [] and len(sent) == 1                      # stopped, did not send the rest
    metas = sorted(tmp_path.glob("htq_job_*.json"))
    assert len(metas) == 1
    m = json.load(open(metas[0]))
    assert m["cancelled"] is True and m["guard"] == CP.GUARD_OVER and m["job_id"] == "stub-0"


def test_strict_submission_continues_once_estimates_arrive(tmp_path, monkeypatch, estimate_clock):
    sent = []

    class _Sampler:
        def run(self, pubs):
            job = _StubJob(jid=f"pending-{len(sent)}")
            sent.append(job)
            return job

    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    monkeypatch.setattr(CP, "_sampler", lambda *a, **k: _Sampler())
    recs = CP.submit("dev", service=_RefreshingService(10800.0, [100.0] * 3),
                     est_seconds=10.0, batch=False,
                     **_submit_args(tmp_path))
    assert len(recs) == 3 and not any(job.cancelled for job in sent)
    assert [r["meta"]["job_index"] for r in recs] == [0, 1, 2]
    assert all(r["meta"]["guard"] == CP.GUARD_OK for r in recs)
    assert estimate_clock.sleeps == [5.0] * 3


@pytest.mark.parametrize("stop_at", [0, 1])
def test_strict_submission_timeout_records_cancelled_job(
        tmp_path, monkeypatch, estimate_clock, stop_at):
    sent, messages = [], []

    class _Sampler:
        def run(self, pubs):
            job = _StubJob(est=100.0 if len(sent) < stop_at else None,
                           jid=f"pending-{len(sent)}")
            sent.append(job)
            return job

    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    monkeypatch.setattr(CP, "_sampler", lambda *a, **k: _Sampler())
    recs = CP.submit("dev", service=_RefreshingService(10800.0), est_seconds=10.0,
                     batch=False, log=messages.append,
                     **_submit_args(tmp_path))
    assert len(recs) == stop_at and len(sent) == stop_at + 1
    assert estimate_clock.elapsed == 60.0 and sent[-1].cancelled is True
    meta = json.loads((tmp_path / f"htq_job_pending-{stop_at}.json").read_text())
    assert meta["job_index"] == stop_at
    assert meta["job_id"] == sent[-1].job_id()
    assert meta["cancelled"] is True and meta["guard"] == CP.GUARD_UNVERIFIABLE


def test_submission_persists_job_id_before_interruptible_wait(tmp_path, monkeypatch, estimate_clock):
    sent = []

    class _Sampler:
        def run(self, pubs):
            job = _StubJob(jid=f"pending-{len(sent)}")
            sent.append(job)
            return job

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    monkeypatch.setattr(CP, "_sampler", lambda *a, **k: _Sampler())
    monkeypatch.setattr(estimate_clock, "sleep", interrupt)
    with pytest.raises(KeyboardInterrupt):
        CP.submit("dev", service=_RefreshingService(10800.0), est_seconds=10.0,
                  batch=False, **_submit_args(tmp_path))
    assert len(sent) == 1 and sent[0].cancelled is False
    meta = json.loads((tmp_path / "htq_job_pending-0.json").read_text())
    assert meta["job_id"] == "pending-0" and meta["job_index"] == 0
    assert meta["guard"] == "pending" and "cancelled" not in meta


@pytest.mark.parametrize("est,guard", [
    (None, CP.GUARD_UNVERIFIABLE),
    (10000.0, CP.GUARD_OVER),
])
def test_submission_stops_with_uncertain_cancellation_without_retry_advice(
        tmp_path, monkeypatch, estimate_clock, est, guard):
    sent, messages = [], []

    class _UncancellableJob(_StubJob):
        def cancel(self):
            raise RuntimeError("job already completed")

    class _Sampler:
        def run(self, pubs):
            job = _UncancellableJob(est=est, jid=f"uncertain-{len(sent)}")
            sent.append(job)
            return job

    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    monkeypatch.setattr(CP, "_sampler", lambda *a, **k: _Sampler())
    with pytest.raises(CP.BudgetError, match="Check its status before retrying") as exc:
        CP.submit("dev", service=_RefreshingService(10800.0), est_seconds=10.0,
                  batch=False, estimate_wait=7.0, log=messages.append,
                  **_submit_args(tmp_path))
    assert len(sent) == 1 and sent[0].cancelled is False
    meta = json.loads((tmp_path / "htq_job_uncertain-0.json").read_text())
    assert meta["job_id"] == "uncertain-0" and meta["job_index"] == 0
    assert meta["guard"] == guard and meta["cancelled"] is None
    assert "job already completed" in meta["cancellation_error"]
    assert not any("--only-jobs" in message for message in messages)
    assert "--only-jobs" not in str(exc.value)


def test_nonstrict_submission_keeps_missing_estimates_without_waiting(
        tmp_path, monkeypatch, estimate_clock):
    sent = []

    class _Sampler:
        def run(self, pubs):
            job = _StubJob(jid=f"unverified-{len(sent)}")
            sent.append(job)
            return job

    monkeypatch.setattr(T, "is_real_backend", lambda be: True)
    monkeypatch.setattr(CP, "_sampler", lambda *a, **k: _Sampler())
    service = _RefreshingService(10800.0)
    recs = CP.submit("dev", service=service, est_seconds=10.0, batch=False,
                     strict_guard=False, **_submit_args(tmp_path))
    assert len(recs) == 3 and not any(job.cancelled for job in sent)
    assert all(r["meta"]["guard"] == CP.GUARD_UNVERIFIABLE for r in recs)
    assert estimate_clock.sleeps == [] and service.job_requests == []


def test_local_submission_needs_no_allocation_and_keeps_every_job(tmp_path, monkeypatch):
    class _Sampler:
        def run(self, pubs):
            return _StubJob(est=None, jid=f"loc-{id(pubs) % 97}")

    monkeypatch.setattr(CP, "_sampler", lambda *a, **k: _Sampler())
    recs = CP.submit("fake", log=None, **_submit_args(tmp_path))
    assert len(recs) == 3 and all(r["meta"]["guard"] == CP.GUARD_OK for r in recs)
    assert all(r["meta"]["cancelled"] is False for r in recs)


@pytest.mark.parametrize("wait_args,wait", [
    ([], 60.0),
    (["--estimate-wait", "0"], 0.0),
    (["--estimate-wait", "17.5"], 17.5),
])
def test_submit_cli_forwards_estimate_wait(monkeypatch, wait_args, wait):
    from htq_hw import __main__ as cli

    names = [f"p{i}" for i in range(3)]
    calls = []

    def submit(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(cli, "check_versions", lambda: [])
    monkeypatch.setattr(cli, "_setup", lambda *a, **k: (None, None, None, "fake", None, []))
    monkeypatch.setattr(CP, "build_pub_circuits", lambda *a, **k: ({}, {}, None))
    monkeypatch.setattr(CP, "group_jobs", lambda *a, **k: [[name] for name in names])
    monkeypatch.setattr(cli, "_plan", lambda *a, **k: ({"shots": {name: 100 for name in names}}, 1e-4))
    monkeypatch.setattr(T, "is_real_backend", lambda be: False)
    monkeypatch.setattr(T, "backend_label", lambda be: "fake")
    monkeypatch.setattr(CP, "submit", submit)

    cli.main(["submit", *wait_args])
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[4] == [["p0"], ["p1"], ["p2"]]
    assert kwargs["estimate_wait"] == wait and kwargs["strict_guard"] is True


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_submit_cli_rejects_invalid_estimate_wait_before_submission(monkeypatch, capsys, value):
    from htq_hw import __main__ as cli

    monkeypatch.setattr(cli, "cmd_submit", lambda args: pytest.fail("invalid wait reached submission"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["submit", "--estimate-wait", value])
    assert exc.value.code == 2
    assert "finite, non-negative" in capsys.readouterr().err
