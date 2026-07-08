"""Self-consistency calibration for the S(q^1) hardware measurement.

Sweeps a single noise-scale s (multiplying gate AND readout error together)
through the noisy Pauli-trajectory MPS sim, recording at each s BOTH
  - the Gauss-witness mean  G(s)  (physical-sector stabilizer average), and
  - the structure factor    S(q;s) with its floor-subtract/rescale recovery.

The hardware run measures G_hw from the SAME bitstrings, which picks s* via
G(s*) = G_hw; the test is whether S(q; s*) then also matches S_hw(q).  One
scalar noise level reproducing both the stabilizer witnesses and the
structure factor == the depolarizing+readout model is self-consistent.

Standalone by design: does NOT import scripts/hw_sq_submit.py so the live
hardware analyze path is never perturbed.  The estimator here is byte-for-byte
the validated one (xv = v//2 integer position; QS carries 2pi/nx).

  PYTHONPATH=. .venv/bin/python scripts/hw_sq_calib.py <ntraj> <per_shots>
"""

import sys
import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from htensor import Z2Lattice, stateprep, wavepacket, backends
from htensor import currents as cur
from htensor.measure import split_current

M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
K0TAG = "k1.26"
P2_0, P1_0 = 0.005, 3e-4            # nominal gate depolarizing (s=1)
RO01_0, RO10_0 = 0.012, 0.028      # nominal asymmetric readout (s=1)
SCALES = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
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


def noise_transform(circ, rng, p2, p1):
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        qs = [circ.find_bit(b).index for b in inst.qubits]
        out.append(inst.operation, qs)
        nn = inst.operation.num_qubits
        if nn == 2 and rng.random() < p2:
            for q in qs:
                p = rng.integers(0, 4)
                (out.x if p == 1 else out.y if p == 2 else out.z
                 if p == 3 else (lambda _: None))(q)
        elif nn == 1 and rng.random() < p1:
            p = rng.integers(1, 4)
            (out.x if p == 1 else out.y if p == 2 else out.z)(qs[0])
    return out


def accumulate(bits, lat, QS, ro01, ro10, ro_seed=7):
    rng = np.random.default_rng(ro_seed)
    b = bits.copy()
    b = b ^ (((b == 0) & (rng.random(b.shape) < ro01)) |
             ((b == 1) & (rng.random(b.shape) < ro10)))
    b = b.astype(np.int8)
    sites = np.array([lat.site_qubit(v) for v in range(lat.ns)])
    zv = 1 - 2 * b[:, sites]
    xv = (np.arange(lat.ns) // 2).astype(float)
    rho = zv @ np.exp(1j * np.outer(QS, xv)).T
    gval = np.ones((len(b), lat.ns))
    for n in range(lat.ns):
        gval[:, n] = ((-1) ** n * (1 - 2 * b[:, lat.site_qubit(n)])
                      * (1 - 2 * b[:, lat.link_qubit(n - 1)])
                      * (1 - 2 * b[:, lat.link_qubit(n)]))
    return dict(sum_rho=rho.sum(0), sum_rho2=(np.abs(rho) ** 2).sum(0),
                gsum=gval.sum(0), n=len(b))


def finalize(accs, lat, QS):
    n = sum(a["n"] for a in accs)
    sr = sum(a["sum_rho"] for a in accs)
    sr2 = sum(a["sum_rho2"] for a in accs)
    gs = sum(a["gsum"] for a in accs)
    S = (sr2 / n - np.abs(sr / n) ** 2).real / lat.nx
    return S, gs / n, n


def run_scale(s, lat, prep, QS, anc, ntraj, per):
    p2, p1 = P2_0 * s, P1_0 * s
    ro01, ro10 = min(RO01_0 * s, 0.4), min(RO10_0 * s, 0.4)
    seeds = np.random.SeedSequence(int(1000 * s) + 7).spawn(ntraj)
    accs = []
    for t in range(ntraj):
        rng = np.random.default_rng(seeds[t])
        mps, perm = backends.prepare_state_mps(
            lat, prep, anc, cap=256, trunc=1e-8,
            circuit_transform=lambda c: noise_transform(c, rng, p2, p1))
        nq = lat.n_qubits + 1
        qc = QuantumCircuit(nq, lat.n_qubits)
        qc.set_matrix_product_state(mps)
        for nn in range(lat.n_links):
            qc.h(perm[lat.link_qubit(nn)])
        for v in range(lat.n_qubits):
            qc.measure(perm[v], v)
        sim = AerSimulator(method="matrix_product_state",
                           matrix_product_state_truncation_threshold=1e-8)
        cnt = sim.run(qc, shots=per).result().get_counts()
        bstr = np.vstack([np.tile(np.frombuffer(bs[::-1].encode(), np.uint8)
                                  - ord("0"), (c, 1))
                          for bs, c in cnt.items()])
        accs.append(accumulate(bstr, lat, QS, ro01, ro10,
                               ro_seed=int(1000 * s) + t))
        log(f"  s={s:.1f} traj {t+1}/{ntraj}")
    S, G, nshot = finalize(accs, lat, QS)
    return S, G, nshot


def main():
    # hw_sq_calib.py <ntraj> <per> [scale_idx | "combine"]
    ntraj = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    Si = np.load(f"data/hwsf_ideal_ns{NS}_{K0TAG}.npz")["S"]
    A = np.vstack([Si, np.ones_like(Si)]).T

    if len(sys.argv) > 3 and sys.argv[3] == "combine":
        import glob
        lat = Z2Lattice(NS, pbc=True)
        QS = 2 * np.pi * np.arange(1, lat.nx // 2 + 1) / lat.nx
        rows = [dict(np.load(f)) for f in
                sorted(glob.glob(f"data/hwsq_calib_s*_{K0TAG}.npz"),
                       key=lambda p: float(dict(np.load(p))["s"]))]
        np.savez(f"data/hwsq_calib_{K0TAG}.npz",
                 scales=np.array([r["s"] for r in rows]), q=QS, S_ideal=Si,
                 gauss=np.array([r["gauss"] for r in rows]),
                 S_raw=np.array([r["S_raw"] for r in rows]),
                 S_mit=np.array([r["S_mit"] for r in rows]),
                 fcs=np.array([r["fc"] for r in rows]),
                 rec=np.array([r["rec"] for r in rows]))
        for r in rows:
            log(f"s={float(r['s']):.1f}: Gauss {float(r['gauss']):.3f}, "
                f"recovered {int(r['rec'])}/{len(QS)}")
        log(f"combined {len(rows)} scales -> data/hwsq_calib_{K0TAG}.npz")
        return

    lat, prep = build_prep()
    QS = 2 * np.pi * np.arange(1, lat.nx // 2 + 1) / lat.nx
    anc = min(split_current(cur.charge_density(lat, CENTER))[1][0][0])
    idxs = [int(sys.argv[3])] if len(sys.argv) > 3 else range(len(SCALES))
    for idx in idxs:
        s = SCALES[idx]
        S, G, nshot = run_scale(s, lat, prep, QS, anc, ntraj, per)
        (f, c), *_ = np.linalg.lstsq(A, S, rcond=None)
        Smit = (S - c) / f
        res = (Smit - Si) / Si
        rec = int(np.sum(np.abs(res) < 0.15))
        np.savez(f"data/hwsq_calib_s{idx}_{K0TAG}.npz", s=s, gauss=G.mean(),
                 G=G, S_raw=S, S_mit=Smit, fc=[f, c], rec=rec, nshot=nshot)
        log(f"s={s:.1f}: Gauss {G.mean():.3f}, f={f:.3f} c={c:.3f}, "
            f"recovered {rec}/{len(QS)}  ({nshot} shots) -> saved")


if __name__ == "__main__":
    main()
