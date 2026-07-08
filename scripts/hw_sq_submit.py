"""Submission circuit + classical estimator for the static structure factor
S(q^1) on 101 qubits -- the prep-only, one-basis, all-momenta hardware
observable.  Built and validated here; NOT submitted.

Pipeline (exactly what a hardware run does):
  circuit  = prep (vacuum ansatz + wavepacket block); H on every link qubit
             (X basis for the Gauss stabilizers); measure all 101 qubits.
  estimator= per shot, z_v = 1-2 bit(site v); the momentum-projected
             magnetization rho(q) = sum_v e^{i q x_v} z_v is a complex scalar,
             and S(q) = [<|rho(q)|^2> - |<rho(q)>|^2]/N_x is its shot
             variance.  Gauss witnesses G_n from the same bitstrings post-
             select the physical sector.

Modes:
  audit                  transpile vs a fake Heron target; depth / 2q gates /
                         shots and QPU-seconds for an 11 s budget.
  simval <ntraj> <shots> noisy Pauli-trajectory sim WITH bitstring sampling
                         and readout error -> S(q), mitigation, recovery vs
                         the exact curve (the real end-to-end validation).
  submit / analyze       hardware hooks (defined, never auto-invoked).

  PYTHONPATH=. .venv/bin/python scripts/hw_sq_submit.py audit
"""

import sys
import time

import numpy as np
from qiskit import QuantumCircuit, transpile

from htensor import Z2Lattice, stateprep, wavepacket, backends
from htensor import currents as cur
from htensor.measure import split_current

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
K0TAG = "k1.26"
P2, P1 = 0.005, 3e-4
RO01, RO10 = 0.012, 0.028      # asymmetric readout (amplitude-damping bias)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def build_prep():
    lat = Z2Lattice(NS, pbc=True)
    TH = stateprep.optimize_vacuum(Z2Lattice(6, pbc=True), M0, G2, ETA,
                                   n_layers=2, restarts=2)["thetas"]
    z = np.load(f"data/wp10reg_params_{K0TAG}_L3.npz", allow_pickle=True)
    params = wavepacket.params_from_vector(z["vec"], list(z["offsets"]),
                                           int(z["L"]))
    prep = stateprep.vacuum_ansatz(lat, TH)
    prep.compose(wavepacket.block_circuit(lat, CENTER, params), inplace=True)
    return lat, prep


def measured_circuit(lat, prep):
    """prep + H on link qubits (X basis) + measure all."""
    qc = prep.copy()
    for n in range(lat.n_links):
        qc.h(lat.link_qubit(n))
    qc.measure_all()
    return qc


def accumulate(bitstrings, lat, QS, ro_seed=7):
    """bitstrings (Nshot,nq) -> additive accumulators so partial runs on
    different machines combine exactly: sum_rho(q), sum |rho(q)|^2, Gauss
    sums, shot count.  S(q) is a shot-variance, so only these sums (not the
    per-worker S) may be pooled."""
    rng = np.random.default_rng(ro_seed)
    b = bitstrings.copy()
    b = b ^ (((b == 0) & (rng.random(b.shape) < RO01)) |
             ((b == 1) & (rng.random(b.shape) < RO10)))
    b = b.astype(np.int8)                          # signed for +-1 arithmetic
    sites = np.array([lat.site_qubit(v) for v in range(lat.ns)])
    zv = 1 - 2 * b[:, sites]
    xv = (np.arange(lat.ns) // 2).astype(float)    # integer position; QS carries 2pi/nx
    rho = zv @ np.exp(1j * np.outer(QS, xv)).T    # (Nshot, nq)
    gval = np.ones((len(b), lat.ns))
    for n in range(lat.ns):
        gval[:, n] = ((-1) ** n * (1 - 2 * b[:, lat.site_qubit(n)])
                      * (1 - 2 * b[:, lat.link_qubit(n - 1)])
                      * (1 - 2 * b[:, lat.link_qubit(n)]))
    return dict(sum_rho=rho.sum(0), sum_rho2=(np.abs(rho) ** 2).sum(0),
                gsum=gval.sum(0), n=len(b))


def finalize(accs, lat, QS):
    """combine accumulators -> S(q), Gauss."""
    n = sum(a["n"] for a in accs)
    sr = sum(a["sum_rho"] for a in accs)
    sr2 = sum(a["sum_rho2"] for a in accs)
    gs = sum(a["gsum"] for a in accs)
    S = (sr2 / n - np.abs(sr / n) ** 2).real / lat.nx
    return S, gs / n, n


def noise_transform(circ, rng):
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        qs = [circ.find_bit(b).index for b in inst.qubits]
        out.append(inst.operation, qs)
        nn = inst.operation.num_qubits
        if nn == 2 and rng.random() < P2:
            for q in qs:
                p = rng.integers(0, 4)
                (out.x if p == 1 else out.y if p == 2 else out.z
                 if p == 3 else (lambda _: None))(q)
        elif nn == 1 and rng.random() < P1:
            p = rng.integers(1, 4)
            (out.x if p == 1 else out.y if p == 2 else out.z)(qs[0])
    return out


mode = sys.argv[1]
lat, prep = build_prep()
QS = 2 * np.pi * np.arange(1, lat.nx // 2 + 1) / lat.nx

if mode == "audit":
    from qiskit_ibm_runtime.fake_provider import FakeFez
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    be = FakeFez()
    qc = measured_circuit(lat, prep)
    pm = generate_preset_pass_manager(backend=be, optimization_level=3,
                                      seed_transpiler=7)
    tc = pm.run(qc)
    two_q = sum(v for k, v in tc.count_ops().items()
                if k in ("cz", "ecr", "cx"))
    log(f"target {be.name}: depth {tc.depth()}, 2q gates {two_q}, "
        f"2q depth {tc.depth(lambda i: i.operation.num_qubits == 2)}")
    for shots in (2e4, 4e4, 1e5):
        log(f"  {int(shots):>6} shots -> ~{shots*250e-6:.1f} s QPU "
            f"(single circuit, all {len(QS)} momenta)")

elif mode == "simval":
    # simval <ntraj> <shots> [worker_id n_workers]
    from qiskit_aer import AerSimulator
    ntraj, shots = int(sys.argv[2]), int(sys.argv[3])
    wid = int(sys.argv[4]) if len(sys.argv) > 5 else 0
    nw = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    per = shots // ntraj
    anc = min(split_current(cur.charge_density(lat, CENTER))[1][0][0])
    accs = []
    ss = np.random.SeedSequence(2024)
    seeds = ss.spawn(ntraj * nw)              # global list; worker takes its slice
    mine = seeds[wid * ntraj:(wid + 1) * ntraj]
    for t in range(ntraj):
        rng = np.random.default_rng(mine[t])
        mps, perm = backends.prepare_state_mps(
            lat, prep, anc, cap=256, trunc=1e-8,
            circuit_transform=lambda c: noise_transform(c, rng))
        n = lat.n_qubits + 1
        qc = QuantumCircuit(n, lat.n_qubits)
        qc.set_matrix_product_state(mps)
        for nn in range(lat.n_links):
            qc.h(perm[lat.link_qubit(nn)])
        for v in range(lat.n_qubits):
            qc.measure(perm[v], v)
        sim = AerSimulator(method="matrix_product_state",
                           matrix_product_state_truncation_threshold=1e-8)
        cnt = sim.run(qc, shots=per).result().get_counts()
        bits = np.vstack([np.tile(np.frombuffer(bs[::-1].encode(), np.uint8)
                                  - ord("0"), (c, 1))
                          for bs, c in cnt.items()])
        accs.append(accumulate(bits, lat, QS, ro_seed=wid * 1000 + t))
        if (t + 1) % 4 == 0:
            log(f"worker {wid}: traj {t+1}/{ntraj}")
    tot = {k: sum(a[k] for a in accs) for k in accs[0]}
    np.savez(f"data/hwsq_acc_w{wid}_{K0TAG}.npz", **tot)
    log(f"worker {wid} saved accumulators ({tot['n']} shots)")
    if nw == 1:
        import subprocess
        subprocess.run([sys.executable, __file__, "combine"],
                       env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"})

elif mode == "submit":
    # submit <backend> <shots>
    from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    import json
    import os
    bname = sys.argv[2] if len(sys.argv) > 2 else None
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
    service = QiskitRuntimeService()
    be = (service.backend(bname) if bname
          else service.least_busy(min_num_qubits=101, operational=True))
    log(f"backend {be.name}; {shots} shots (~{shots*265e-6:.1f}s QPU)")
    qc = measured_circuit(lat, prep)
    pm = generate_preset_pass_manager(backend=be, optimization_level=3,
                                      seed_transpiler=7)
    tc = pm.run(qc)
    two_q = sum(v for k, v in tc.count_ops().items()
                if k in ("cz", "ecr", "cx"))
    log(f"transpiled: depth {tc.depth()}, {two_q} 2q gates")
    sampler = SamplerV2(mode=be)
    sampler.options.default_shots = shots
    sampler.options.twirling.enable_measure = True
    sampler.options.twirling.enable_gates = False
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    job = sampler.run([tc])
    os.makedirs("data/hw", exist_ok=True)
    creg = tc.cregs[-1].name
    meta = {"job_id": job.job_id(), "backend": be.name, "shots": shots,
            "creg": creg}
    json.dump(meta, open(f"data/hw/sq_job_{job.job_id()}.json", "w"), indent=1)
    log(f"SUBMITTED {job.job_id()} (creg '{creg}') "
        f"-> data/hw/sq_job_{job.job_id()}.json")

elif mode == "analyze":
    from qiskit_ibm_runtime import QiskitRuntimeService
    import json
    meta = json.load(open(sys.argv[2]))
    res = QiskitRuntimeService().job(meta["job_id"]).result()
    ba = res[0].data[meta["creg"]]
    counts = ba.get_counts()
    bits = np.vstack([np.tile(np.frombuffer(bs[::-1].encode(), np.uint8)
                              - ord("0"), (c, 1))
                      for bs, c in counts.items()]).astype(np.uint8)
    acc = accumulate(bits, lat, QS)
    S, G, nshot = finalize([acc], lat, QS)
    Si = np.load(f"data/hwsf_ideal_ns{NS}_{K0TAG}.npz")["S"]
    A = np.vstack([Si, np.ones_like(Si)]).T
    (f, c), *_ = np.linalg.lstsq(A, S, rcond=None)
    Smit = (S - c) / f
    res_ = (Smit - Si) / Si
    log(f"HARDWARE {meta['backend']}: {nshot} shots, Gauss mean {G.mean():.3f}")
    log(f"noise fit f={f:.3f} c={c:.3f}; recovered "
        f"{int(np.sum(np.abs(res_) < 0.15))}/{len(QS)} <15%, "
        f"{int(np.sum(np.abs(res_) < 0.20))}/{len(QS)} <20%")
    np.savez(f"data/hwsq_HARDWARE_{K0TAG}.npz", q=QS, S_raw=S, S_mit=Smit,
             S_ideal=Si, G=G, f=f, c=c, nshot=nshot, backend=meta["backend"])
    for i, q in enumerate(QS):
        log(f"  q={q:.2f}: exact {Si[i]:.3f} raw {S[i]:.3f} "
            f"mit {Smit[i]:.3f} ({res_[i]:+.1%})")

elif mode == "combine":
    import glob
    files = sorted(glob.glob(f"data/hwsq_acc_w*_{K0TAG}.npz"))
    accs = [dict(np.load(f)) for f in files]
    S, G, nshot = finalize(accs, lat, QS)
    Si = np.load(f"data/hwsf_ideal_ns{NS}_{K0TAG}.npz")["S"]
    A = np.vstack([Si, np.ones_like(Si)]).T
    (f, c), *_ = np.linalg.lstsq(A, S, rcond=None)
    Smit = (S - c) / f
    res = (Smit - Si) / Si
    log(f"combined {len(files)} workers, {nshot} shots; Gauss mean {G.mean():.3f}")
    log(f"noise fit f={f:.3f} c={c:.3f}; "
        f"recovered {int(np.sum(np.abs(res) < 0.15))}/{len(QS)} "
        f"<15%, {int(np.sum(np.abs(res) < 0.20))}/{len(QS)} <20%")
    np.savez(f"data/hwsq_simval_{K0TAG}.npz", q=QS, S_raw=S, S_mit=Smit,
             S_ideal=Si, G=G, f=f, c=c, nshot=nshot)
    for i, q in enumerate(QS):
        log(f"  q={q:.2f}: exact {Si[i]:.3f} raw {S[i]:.3f} "
            f"mit {Smit[i]:.3f} ({res[i]:+.1%})")
