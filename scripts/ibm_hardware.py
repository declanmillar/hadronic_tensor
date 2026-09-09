"""IBM Heron hardware path for the 101-qubit production point (paper Sec.
"Toward hardware").  Three tiers of increasing depth/risk, all reusing the
exact production circuit builders:

  tier 1  certify   : prep-only certificates -- 50 Gauss-law stabilizers
                      <G_n> (each weight-3, value 1 in the ideal state and a
                      direct local-fidelity witness), charge profile
                      <J0(v)> (packet bump on staggered background), <H>,
                      fermion parity, total charge.  100 qubits, no Trotter.
  tier 2  transport : packet charge transport <J0(v,t)> after 1-4 Trotter
                      steps (t = 0.5..2.0, dt = 0.5).  One-point functions
                      only -- no ancilla, no controlled gates.  Truth curves
                      exist in data/w_meson_ns50_k1.26_v3.npz (one_pt_wp).
  tier 3  hadamard  : W00 integrand C(t,x) at t = 0, 0.5, 1.0 for ALL 50
                      sites (x-multiplexed: every XB_v shares one basis, so
                      extra probes are QPU-free).  J0 insertion is a SINGLE
                      controlled-Z (J0 = ((-1)^v - Z_v)/2), so the overhead
                      over tier 2 is one ancilla + one CZ + basis rotation.
                      t=0 is a known-truth slice that calibrates the global
                      damping factor kappa (cf. scripts/hw_t3_dryrun.py).

Modes:
  audit [fez|torino]        offline: transpile all tiers against a fake
                            Heron target, report depth / 2q-gate counts /
                            pub sizes / suggested shots.  No credentials.
  submit <tier> [backend]   submit one tier via QiskitRuntimeService
                            (saved account or QISKIT_IBM_TOKEN env).
                            Writes data/hw/job_tier<k>_*.json.
  analyze <jobfile>         fetch results, save npz next to the job file,
                            print comparisons against MPS truth.

  PYTHONPATH=. .venv/bin/python scripts/ibm_hardware.py audit
"""

import json
import os
import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from htensor import Z2Lattice, stateprep, wavepacket
from htensor import hamiltonian as ham
from htensor import currents as cur
from htensor import trotter
from htensor.measure import split_current, hadamard_test_circuit

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
K0TAG = "k1.26"     # boosted packet -- same prep as the S(q) hardware run
DT = 0.5
T2_TIMES = [0.5, 1.0, 1.5, 2.0]
T3_TIMES = [0.0, 0.5, 1.0, 2.0]  # t=0: known-truth calibration slice
T3_PROBES = list(range(NS))  # all sites: probes beyond the first are free
# staged submission against the hard 600 s budget: each part is a complete,
# analyzable unit; physics+mirror pairs stay in ONE job (drift immunity).
# t=2.0 targets the peak's coherent bounce (truth 0.480/0.436/0.458 at
# t=0.5/1.0/2.0) -- three slices resolve non-monotonic dynamics.
T3_PARTS = {"a": ["t0.5", "m0.5"], "b": ["t1.0", "m1.0"], "c": ["t0.0"],
            "d": ["t2.0", "m2.0"]}
T3_PART = None          # set from the CLI: submit 3 <backend> <part> [shots]
T3_SHOTS = None         # per-part shots override (default SHOTS[3])
T3_GUARD = 0.85         # cancel if usage_estimation > GUARD * remaining
SHOTS = {1: 8192, 2: 8192, 3: 100_000}
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def build_prep(lat):
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                   n_layers=2, restarts=2)["thetas"]
    z = np.load(f"data/wp10reg_params_{K0TAG}_L3.npz", allow_pickle=True)
    params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                           int(z["L"]))
    prep = stateprep.vacuum_ansatz(lat, TH)
    prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)
    return prep


def with_ancilla(op, anc_pauli):
    return SparsePauliOp([anc_pauli + l for l in op.paulis.to_labels()],
                         op.coeffs)


def build_tier(tier, lat, prep):
    """-> list of (circ_name, circuit, [(obs_name, SparsePauliOp)])."""
    if tier == 1:
        obs = [(f"G{n}", ham.gauss_operator(lat, n)) for n in range(NS)]
        obs += [(f"J0_{v}", cur.charge_density(lat, v)) for v in range(NS)]
        obs += [("H", ham.build_hamiltonian(lat, M0, G2, ETA)),
                ("Pf", ham.fermion_parity(lat)),
                ("Q", ham.total_charge(lat))]
        return [("prep", prep, obs)]
    if tier == 2:
        out = []
        obs = [(f"J0_{v}", cur.charge_density(lat, v)) for v in range(NS)]
        for t in T2_TIMES:
            qc = prep.copy()
            qc.compose(trotter.trotter_circuit(lat, M0, G2, ETA, t,
                                               int(round(t / DT))),
                       inplace=True)
            out.append((f"t{t:.1f}", qc, obs))
        return out
    if tier == 3:
        raise SystemExit("tier 3 uses the segment path: t3_segments/"
                         "t3_transpile (mirror self-mitigation)")
    raise SystemExit(f"unknown tier {tier}")


# ---- tier 3: Hadamard test with depth-matched mirror self-mitigation ----
# Each t>0 slice gets a companion 'mirror' circuit: the SAME prep + gadget
# and a Trotter block with the same gate skeleton but total angle
# MIRROR_EPS -> net identity.  Its exact truth is the t=0 slice, so it
# measures the per-(t,x) damping kappa_v(t), beta_v(t) at matched depth.
# The Trotter segments transpile at optimization_level=1 (O3 would delete
# the mirror's gates); prep+gadget transpiles once at O3 and is shared
# verbatim by all five circuits.
MIRROR_EPS = 1e-8


def t3_obs(lat):
    obs = []
    for v in T3_PROBES:
        B = cur.charge_density(lat, v)
        obs += [(f"XB_{v}", with_ancilla(B, "X")),
                (f"YB_{v}", with_ancilla(B, "Y")),
                (f"B_{v}", with_ancilla(B, "I"))]
    obs += [("X", with_ancilla(SparsePauliOp("I" * lat.n_qubits), "X")),
            ("Y", with_ancilla(SparsePauliOp("I" * lat.n_qubits), "Y"))]
    return obs


def t3_segments(lat, prep):
    """-> (base, [(name, t, trotter_block|None)], insertion metadata).
    base = prep + Hadamard-test gadget (J0 insertion = one controlled-Z)."""
    from htensor.measure import controlled_pauli
    id_c, terms = split_current(cur.charge_density(lat, CENTER))
    (ins_ops, ins_coeff), = terms
    base = QuantumCircuit(lat.n_qubits + 1)
    base.compose(prep, range(lat.n_qubits), inplace=True)
    base.h(lat.n_qubits)
    controlled_pauli(base, lat.n_qubits, ins_ops)
    segs = [("t0.0", 0.0, None)]
    for t in T3_TIMES:
        if t == 0.0:
            continue
        n = int(round(t / DT))
        segs.append((f"t{t:.1f}", t,
                     trotter.trotter_circuit(lat, M0, G2, ETA, t, n)))
        segs.append((f"m{t:.1f}", t,
                     trotter.trotter_circuit(lat, M0, G2, ETA,
                                             MIRROR_EPS, n)))
    meta = {"id_c": float(np.real(id_c)),
            "ins_coeff": float(np.real(ins_coeff))}
    return base, segs, meta


def t3_transpile(base, segs, be):
    """Stitched ISA: base at O3, each Trotter segment at O1 starting from
    base's final layout.  -> [(name, t, isa_circuit, layout_for_obs)].
    Asserts every mirror shares its physics partner's 2q skeleton."""
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    pm3 = generate_preset_pass_manager(backend=be, optimization_level=3,
                                       seed_transpiler=7)
    isa_A = pm3.run(base)
    fl = isa_A.layout.final_index_layout()
    pm1 = generate_preset_pass_manager(backend=be, optimization_level=1,
                                       seed_transpiler=7, initial_layout=fl)
    out, skel = [], {}
    for name, t, blk in segs:
        if blk is None:
            out.append((name, t, isa_A, isa_A.layout))
            continue
        b102 = QuantumCircuit(base.num_qubits)
        b102.compose(blk, range(blk.num_qubits), inplace=True)
        isa_B = pm1.run(b102)
        skel[name] = [(inst.operation.name,
                       tuple(isa_B.find_bit(q).index for q in inst.qubits))
                      for inst in isa_B.data
                      if inst.operation.num_qubits == 2]
        out.append((name, t, isa_A.compose(isa_B), isa_B.layout))
    for mname in [k for k in skel if k.startswith("m")]:
        pname = "t" + mname[1:]
        a, b = skel[pname], skel[mname]
        if a != b:
            raise SystemExit(f"mirror skeleton mismatch {pname}/{mname}: "
                             f"{len(a)} vs {len(b)} 2q gates")
        log(f"{pname}: mirror skeleton matches physics "
            f"({len(a)} 2q gates)")
    return out


def fake_backend(name):
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeTorino
    return FakeFez() if name == "fez" else FakeTorino()


def isa_transpile(circs, backend):
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                      seed_transpiler=7)
    return [pm.run(c) for c in circs]


def audit(backend_name="fez"):
    be = fake_backend(backend_name)
    log(f"audit target: {be.name} ({be.num_qubits}q, "
        f"basis {be.operation_names})")
    lat = Z2Lattice(NS, pbc=True)
    prep = build_prep(lat)
    for tier in (1, 2):
        entries = build_tier(tier, lat, prep)
        circs = [c for _, c, _ in entries]
        isa = isa_transpile(circs, be)
        for (name, _, obs), tc in zip(entries, isa):
            two_q = sum(v for k, v in tc.count_ops().items()
                        if k in ("cz", "ecr", "cx"))
            log(f"tier {tier} [{name}]: depth {tc.depth()}, "
                f"2q gates {two_q}, 2q depth "
                f"{tc.depth(lambda i: i.operation.num_qubits == 2)}, "
                f"{len(obs)} observables, shots {SHOTS[tier]}")
    base, segs, _ = t3_segments(lat, prep)
    nobs = len(t3_obs(lat))
    for name, t, tc, _ in t3_transpile(base, segs, be):
        two_q = sum(v for k, v in tc.count_ops().items()
                    if k in ("cz", "ecr", "cx"))
        log(f"tier 3 [{name}]: depth {tc.depth()}, 2q gates {two_q}, "
            f"2q depth {tc.depth(lambda i: i.operation.num_qubits == 2)}, "
            f"{nobs} observables, shots {SHOTS[3]}")
    log("audit done")


def submit(tier, backend_name=None):
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
    service = QiskitRuntimeService()
    be = (service.backend(backend_name) if backend_name
          else service.least_busy(min_num_qubits=NS * 2 + 5,
                                  operational=True))
    log(f"backend: {be.name}")
    lat = Z2Lattice(NS, pbc=True)
    prep = build_prep(lat)
    if tier == 3:
        base, segs, meta3 = t3_segments(lat, prep)
        obs = t3_obs(lat)
        isa3 = t3_transpile(base, segs, be)
        if T3_PART:
            isa3 = [e for e in isa3 if e[0] in T3_PARTS[T3_PART]]
        pubs, circ_names, circ_meta = [], [], []
        for name, t, tc, layout in isa3:
            pubs.append((tc, [o.apply_layout(layout) for _, o in obs]))
            circ_names.append(name)
            circ_meta.append(dict(meta3, t=t,
                                  kind="mirror" if name.startswith("m")
                                  else "phys"))
            two_q = sum(v for k, v in tc.count_ops().items()
                        if k in ("cz", "ecr", "cx"))
            log(f"pub {name}: 2q gates {two_q}, depth {tc.depth()}")
        obs_names = [[on for on, _ in obs]] * len(pubs)
    else:
        entries = build_tier(tier, lat, prep)
        isa = isa_transpile([c for _, c, _ in entries], be)
        pubs = []
        for (name, _, obs), tc in zip(entries, isa):
            layout_obs = [o.apply_layout(tc.layout) for _, o in obs]
            pubs.append((tc, layout_obs))
        circ_names = [n for n, _, _ in entries]
        obs_names = [[on for on, _ in obs] for _, _, obs in entries]
        circ_meta = [c.metadata for _, c, _ in entries]
    est = EstimatorV2(mode=be)
    est.options.default_shots = (T3_SHOTS if tier == 3 and T3_SHOTS
                                 else SHOTS[tier])
    est.options.dynamical_decoupling.enable = True
    est.options.dynamical_decoupling.sequence_type = "XY4"
    est.options.twirling.enable_gates = True
    est.options.twirling.enable_measure = True
    est.options.resilience.measure_mitigation = True
    shots = int(est.options.default_shots)
    job = est.run(pubs)
    if tier == 3:
        # budget guard: the instance has a HARD 600 s allocation.  Cancel
        # before execution (free) if the platform's own estimate would eat
        # more than T3_GUARD of what remains.
        try:
            rem = service.usage()["usage_remaining_seconds"]
        except Exception:
            rem = None
        est_s = None
        try:
            ue = job.usage_estimation
            est_s = ue.get("quantum_seconds", None) if ue else None
        except Exception as e:
            log(f"usage_estimation unavailable ({type(e).__name__})")
        log(f"budget: estimated {est_s} s, remaining {rem} s")
        if est_s is not None and rem is not None and est_s > T3_GUARD * rem:
            job.cancel()
            log(f"CANCELLED {job.job_id()}: estimate {est_s} s exceeds "
                f"{T3_GUARD:.0%} of remaining {rem} s -- resubmit with "
                f"fewer shots (submit 3 <backend> <part> <shots>)")
            return
    os.makedirs("data/hw", exist_ok=True)
    meta = {"job_id": job.job_id(), "tier": tier, "backend": be.name,
            "shots": shots, "circ_names": circ_names,
            "obs_names": obs_names, "circ_meta": circ_meta}
    part = T3_PART or ""
    path = f"data/hw/job_tier{tier}{part}_{job.job_id()}.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
    log(f"submitted {job.job_id()} -> {path}")


def analyze_t3(jobfile, metas, results):
    """Assemble C(t,x) from the raw tier-3 estimator values, with mirror
    self-mitigation.  Assembly validated against the MPS pipeline in
    scripts/hw_t3_dryrun.py (t=0 row reproduces the truth grid to 2e-16):
      <X (x) B_v> = id_b <X_anc> + <X (x) c_b P_b>    (bare-anc subtraction)
      C = c_a (sx + i sy) + id_a (B - id_b) + id_b (A0 - id_a) + id_a id_b
    Each t>0 slice is calibrated by its depth-matched mirror circuit, whose
    exact truth is the t=0 slice (from data/hw_t3_ideal50_*.npz):
      kappa_v(t) = sx_mirror(t,v) / sx_ideal(0,v)    ancilla-sector damping
      beta_v(t)  = (B_mirror(t,v)-1/2)/(B_ideal(0,v)-1/2)  Z-sector damping
    The t=0 slice is its own mirror.

    metas/results are LISTS: multiple jobs covering the same circuits are
    inverse-variance combined per observable (ev,std) -- e.g. a shot top-up
    of the t=0.5 pair merges with the original 50k job as if run together."""
    from htensor.measure import split_current
    lat = Z2Lattice(NS, pbc=True)
    if isinstance(metas, dict):                      # back-compat: single job
        metas, results = [metas], [results]
    cm = metas[0]["circ_meta"]
    id_a = float(cm[0]["id_c"])
    c_a = float(cm[0]["ins_coeff"])
    id_b = np.array([split_current(cur.charge_density(lat, v))[0]
                     for v in T3_PROBES])
    jc = T3_PROBES.index(CENTER)

    # exact t=0 references (all-site noiseless grid, dt=0.5 pipeline)
    idl = np.load(f"data/hw_t3_ideal50_{K0TAG}.npz")
    it0 = int(np.argmin(np.abs(idl["times"])))
    sx_i0 = np.array([idl[f"XB_{v}"][it0] for v in T3_PROBES]) \
        - id_b * idl["X"][it0]
    b_i0 = np.array([idl[f"B_{v}"][it0] for v in T3_PROBES])
    times = sorted({float(m["t"]) for m in cm})

    def byname_of(meta, res):
        bn = {}
        for name, names, pub in zip(meta["circ_names"], meta["obs_names"],
                                    res):
            ev = np.atleast_1d(np.asarray(pub.data.evs, dtype=float))
            sd = np.atleast_1d(np.asarray(pub.data.stds, dtype=float))
            d = {n: (float(v), float(s)) for n, v, s in zip(names, ev, sd)}
            g = lambda tg, k: np.array([d[f"{tg}_{v}"][k] for v in T3_PROBES])
            r = {"XB": g("XB", 0), "YB": g("YB", 0), "B": g("B", 0),
                 "XBe": g("XB", 1), "YBe": g("YB", 1), "Be": g("B", 1),
                 "X": d["X"][0], "Y": d["Y"][0],
                 "Xe": d["X"][1], "Ye": d["Y"][1]}
            r["sx"] = r["XB"] - id_b * r["X"]
            r["sy"] = r["YB"] - id_b * r["Y"]
            r["sxe"] = np.sqrt(r["XBe"] ** 2 + (id_b * r["Xe"]) ** 2)
            r["sye"] = np.sqrt(r["YBe"] ** 2 + (id_b * r["Ye"]) ** 2)
            bn[name] = r
        return bn

    def calibrate(bn):
        """One job -> per-time (C_raw_anc, C_cal_anc, b_cal, C_err, kap, bet).
        Each job is calibrated by ITS OWN mirror, so a run's drift is
        absorbed by that run's mirror (physics and mirror ran in the same
        job on the same qubits -> the damping is common-mode and cancels in
        the sx/kappa ratio; verified stable even where raw values drift 4s)."""
        out = {}
        for t in times:
            p = bn[f"t{t:.1f}"]
            m = bn[f"m{t:.1f}"] if t > 0 else p
            kap = np.where(np.abs(sx_i0) > 0.02, m["sx"] / sx_i0, np.nan)
            kap = np.where(np.isnan(kap), np.nanmedian(kap), kap)
            bet = np.where(np.abs(b_i0 - 0.5) > 0.02,
                           (m["B"] - 0.5) / (b_i0 - 0.5), 1.0)
            b_cal = 0.5 + (p["B"] - 0.5) / bet
            anc_raw = c_a * (p["sx"] + 1j * p["sy"]) + id_a * (p["B"] - id_b)
            anc_cal = c_a * (p["sx"] + 1j * p["sy"]) / kap
            kerr = np.where(np.abs(sx_i0) > 0.02, m["sxe"] / np.abs(sx_i0),
                            np.median(m["sxe"]) / 0.02) / np.abs(kap)
            # kappa-uncertainty propagates through the FULL complex ancilla
            # signal c_a(sx+i sy)/kap, so both sx^2 and sy^2 enter
            cerr = np.sqrt((c_a / kap) ** 2 * (p["sxe"] ** 2 + p["sye"] ** 2)
                           + (c_a / kap) ** 2 * (p["sx"] ** 2 + p["sy"] ** 2)
                           * kerr ** 2
                           + (id_a * p["Be"] / np.abs(bet)) ** 2)
            out[t] = dict(anc_raw=anc_raw, anc_cal=anc_cal, b_cal=b_cal,
                          B=p["B"], cerr=cerr, kap=kap, bet=bet)
        return out

    per_job = [calibrate(byname_of(m, r)) for m, r in zip(metas, results)]

    # per-time, inverse-variance merge of the CALIBRATED slices across jobs,
    # with chi-square inflation so residual run-to-run drift (not captured by
    # each run's own mirror) enters the error bar honestly.
    C, C_cal, C_err, kaps, bets, b_cals = [], [], [], [], [], []
    for t in times:
        slabs = [pj[t] for pj in per_job]
        w = np.array([1.0 / np.maximum(s["cerr"], 1e-9) ** 2 for s in slabs])
        wsum = w.sum(0)
        ac = sum(w[i] * slabs[i]["anc_cal"] for i in range(len(slabs))) / wsum
        bc = sum(w[i] * slabs[i]["b_cal"] for i in range(len(slabs))) / wsum
        e = 1.0 / np.sqrt(wsum)
        if len(slabs) > 1:
            chi2 = sum(w[i] * np.abs(slabs[i]["anc_cal"] - ac) ** 2
                       for i in range(len(slabs))) / (len(slabs) - 1)
            e = e * np.sqrt(np.maximum(chi2, 1.0))     # inflate, never shrink
        C_cal.append((ac, bc))
        C_err.append(e)
        b_cals.append(bc)
        kaps.append(np.mean([s["kap"] for s in slabs], 0))
        bets.append(np.mean([s["bet"] for s in slabs], 0))
        # raw (uncalibrated) merged for reference display only
        wr = np.array([1.0 / np.maximum(s["cerr"], 1e-9) ** 2 for s in slabs])
        C.append(sum(wr[i] * slabs[i]["anc_raw"]
                     for i in range(len(slabs))) / wr.sum(0))

    A0 = b_i0[jc]              # part-a jobs carry no t=0 pub; use MPS ref
    A0_raw = b_i0[jc]
    Cc = []
    for i, t in enumerate(times):
        anc, b_cal = C_cal[i]
        Cc.append(anc + id_a * (b_cal - id_b)
                  + id_b * (A0 - id_a) + id_a * id_b)
        C[i] = C[i] + id_b * (A0_raw - id_a) + id_a * id_b
    C, C_cal, C_err = np.array(C), np.array(Cc), np.array(C_err)
    kaps, bets, b_cals = np.array(kaps), np.array(bets), np.array(b_cals)

    nshot_tot = sum(m["shots"] for m in metas)
    np.savez(jobfile.replace(".json", ".npz"),
             times=np.array(times), probes=np.array(T3_PROBES),
             C=C, C_cal=C_cal, C_err=C_err, kappa_v=kaps, beta_v=bets,
             b_cal=b_cals, id_a=id_a, c_a=c_a, id_b=id_b, tier=3,
             backend=metas[0]["backend"], nshot=nshot_tot,
             merged_jobs=[m["job_id"] for m in metas])
    tru = np.load(f"data/w_meson_ns50_{K0TAG}_v3.npz")
    for i, t in enumerate(times):
        ti = int(np.argmin(np.abs(tru["times"] - t)))
        ct = tru["corr_wp"][ti, T3_PROBES].real
        r = np.corrcoef(C_cal[i].real, ct)[0, 1]
        log(f"t={t:3.1f}: kappa(center) {kaps[i][jc]:.3f}, "
            f"corr(C_cal, truth) = {r:.4f}, C_cal(center) = "
            f"{C_cal[i, jc].real:.4f} (truth {ct[jc]:.4f}, dt=0.5 "
            f"Trotter gap not deducted)")
    log(f"saved {jobfile.replace('.json', '.npz')}")


def t3check(backend_name="fez"):
    """Offline validation of the tier-3 submission path, no credentials:
    stitched transpilation + mirror-skeleton assert, then end-to-end MPS
    simulation of three ISA circuits (t0.0: base+observable mapping;
    m0.5: stitch+mirror, truth = t=0 slice; t0.5: physics stitch, truth =
    ideal dt=0.5 values) against data/hw_t3_ideal_*.npz."""
    from qiskit_aer import AerSimulator
    be = fake_backend(backend_name)
    lat = Z2Lattice(NS, pbc=True)
    prep = build_prep(lat)
    base, segs, _ = t3_segments(lat, prep)
    isa = {n: (t, tc, lay) for n, t, tc, lay in t3_transpile(base, segs, be)}
    for n, (t, tc, lay) in isa.items():
        two_q = sum(v for k, v in tc.count_ops().items()
                    if k in ("cz", "ecr", "cx"))
        log(f"[{n}] 2q gates {two_q}, depth {tc.depth()}")
    obs = dict(t3_obs(lat))
    idl = np.load(f"data/hw_t3_ideal_{K0TAG}.npz")
    irow = {0.0: 0, 0.5: 1, 1.0: 2}
    names = [f"{tag}_{v}" for v in idl["probes"]
             for tag in ("XB", "YB", "B")] + ["X", "Y"]
    sim = AerSimulator(method="matrix_product_state",
                       matrix_product_state_truncation_threshold=1e-8,
                       max_parallel_threads=4)
    for cname in ("t0.0", "m0.5", "t0.5"):
        t, tc, lay = isa[cname]
        row = irow[0.0] if cname.startswith("m") else irow[t]
        qc = tc.copy()
        for n in names:
            qc.save_expectation_value(obs[n].apply_layout(lay),
                                      list(range(tc.num_qubits)), label=n)
        d = sim.run(qc).result().data()
        diffs = {n: float(np.real(d[n])) - float(idl[n][row])
                 for n in names}
        worst = max(diffs, key=lambda n: abs(diffs[n]))
        log(f"[{cname}] vs ideal row t={idl['times'][row]:.1f}: "
            f"max |diff| = {abs(diffs[worst]):.2e} ({worst}); "
            f"XB_{CENTER}: isa {float(np.real(d[f'XB_{CENTER}'])):+.5f} "
            f"ideal {float(idl[f'XB_{CENTER}'][row]):+.5f}")
        # tolerance: the reference pipeline preps at bond cap 512, whose
        # residual bias on far-probe site observables is ~1.3e-3 (verified
        # cap-independent of trunc 1e-8 vs 1e-10); a genuine layout or
        # stitching bug would show at O(0.1).  5e-3 separates the two.
        bad = {n: v for n, v in diffs.items() if abs(v) > 5e-3}
        if bad:
            raise SystemExit(f"[{cname}] VALIDATION FAILED: {bad}")
    log("t3check PASSED: submission path reproduces the logical pipeline")


def analyze(jobfile):
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    # comma-separated jobfiles -> inverse-variance merge (tier 3 top-ups)
    files = jobfile.split(",")
    metas = [json.load(open(f)) for f in files]
    results = [service.job(m["job_id"]).result() for m in metas]
    if metas[0]["tier"] == 3:
        return analyze_t3(files[0], metas, results)
    meta, res = metas[0], results[0]
    out = {}
    for names, pub in zip(meta["obs_names"], res):
        ev = np.asarray(pub.data.evs, dtype=float)
        for n, v in zip(names, np.atleast_1d(ev)):
            out[n] = v
    np.savez(jobfile.replace(".json", ".npz"), **out,
             tier=meta["tier"], backend=meta["backend"])
    tier = meta["tier"]
    if tier == 1:
        g = np.array([out[f"G{n}"] for n in range(NS)])
        log(f"Gauss stabilizers: mean {g.mean():.4f}, min {g.min():.4f} "
            f"(ideal 1; mean is a local-fidelity witness)")
        log(f"<H> = {out['H']:.3f} (MPS truth -49.37 for the {K0TAG} prep), "
            f"Pf = {out['Pf']:.3f}, Q = {out['Q']:.3f}")
    log(f"saved {jobfile.replace('.json', '.npz')}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "audit":
        audit(sys.argv[2] if len(sys.argv) > 2 else "fez")
    elif mode == "t3check":
        t3check(sys.argv[2] if len(sys.argv) > 2 else "fez")
    elif mode == "submit":
        if len(sys.argv) > 4:
            if sys.argv[4] not in T3_PARTS:
                raise SystemExit(f"unknown part {sys.argv[4]!r}; "
                                 f"choose from {sorted(T3_PARTS)}")
            T3_PART = sys.argv[4]
        if len(sys.argv) > 5:
            T3_SHOTS = int(sys.argv[5])
        submit(int(sys.argv[2]),
               sys.argv[3] if len(sys.argv) > 3 else None)
    elif mode == "analyze":
        analyze(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {mode}")
