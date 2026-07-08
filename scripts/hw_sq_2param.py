"""Two-parameter noise model for the S(q^1) self-consistency.

The single-scalar model failed because it forced one knob to fit two error
sectors: the Gauss witness G_n = <(-1)^n Z_n X_{n-1} X_n> probes the LINK
qubits (extra H layer, X basis), while S(q^1) is a SITE-only observable.
Here we separate them:
  - gate depolarizing scale g  -> sets the S(q) contrast (pin to hardware f),
  - link-readout scale L        -> sets the Gauss witness (pin to G_hw).
Because S(q) uses only site bits, it is *exactly* independent of L, and the
model is self-consistent by construction if (i) nominal-gate g~1 reproduces
the measured S(q), and (ii) some physical L reproduces the witness.  Readout
is classical post-processing, so bits are generated once per g and L is
scanned for free.  g<=1.3 stays out of the high-noise MPS segfault regime.

  bits <g_idx>   generate + save raw bits at GATES[g_idx]
  analyze        fit g via S(q), scan L via witness, locate (g*, L*)
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
P2_0, P1_0 = 0.005, 3e-4
RO01_0, RO10_0 = 0.012, 0.028
GATES = [0.7, 1.0, 1.3]
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


def gen_bits(g, lat, prep, anc, ntraj, per, base_seed):
    p2, p1 = P2_0 * g, P1_0 * g
    seeds = np.random.SeedSequence(base_seed).spawn(ntraj)
    allb = []
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
        b = np.vstack([np.tile(np.frombuffer(bs[::-1].encode(), np.uint8)
                               - ord("0"), (c, 1)) for bs, c in cnt.items()])
        allb.append(b.astype(np.uint8))
        log(f"  g={g} traj {t+1}/{ntraj}")
    return np.vstack(allb)


def apply_ro_split(bits, lat, link_scale, seed):
    """Site qubits get nominal readout; link qubits get link_scale x nominal."""
    rng = np.random.default_rng(seed)
    ro01 = np.full(bits.shape[1], RO01_0)
    ro10 = np.full(bits.shape[1], RO10_0)
    for n in range(lat.n_links):
        ro01[lat.link_qubit(n)] = min(RO01_0 * link_scale, 0.5)
        ro10[lat.link_qubit(n)] = min(RO10_0 * link_scale, 0.5)
    b = bits.astype(np.int8)
    flip = (((bits == 0) & (rng.random(bits.shape) < ro01)) |
            ((bits == 1) & (rng.random(bits.shape) < ro10)))
    return (b ^ flip.astype(np.int8))


def sq_and_witness(bsig, lat, QS):
    sites = np.array([lat.site_qubit(v) for v in range(lat.ns)])
    zv = 1 - 2 * bsig[:, sites]
    xv = (np.arange(lat.ns) // 2).astype(float)
    rho = zv @ np.exp(1j * np.outer(QS, xv)).T
    S = ((np.abs(rho) ** 2).mean(0) - np.abs(rho.mean(0)) ** 2).real / lat.nx
    gval = np.ones((len(bsig), lat.ns))
    for n in range(lat.ns):
        gval[:, n] = ((-1) ** n * (1 - 2 * bsig[:, lat.site_qubit(n)])
                      * (1 - 2 * bsig[:, lat.link_qubit(n - 1)])
                      * (1 - 2 * bsig[:, lat.link_qubit(n)]))
    return S, gval.mean()


mode = sys.argv[1]

if mode == "bits":
    gi = int(sys.argv[2])
    ntraj = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    per = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
    lat, prep = build_prep()
    anc = min(split_current(cur.charge_density(lat, CENTER))[1][0][0])
    bits = gen_bits(GATES[gi], lat, prep, anc, ntraj, per, 7000 + gi)
    np.savez_compressed(f"data/2p_bits_g{gi}_{K0TAG}.npz", bits=bits,
                        g=GATES[gi])
    log(f"saved g-idx {gi} (g={GATES[gi]}, {len(bits)} shots)")

elif mode == "analyze":
    import glob
    lat = Z2Lattice(NS, pbc=True)
    QS = 2 * np.pi * np.arange(1, lat.nx // 2 + 1) / lat.nx
    Si = np.load(f"data/hwsf_ideal_ns{NS}_{K0TAG}.npz")["S"]
    A = np.vstack([Si, np.ones_like(Si)]).T
    hw = np.load(f"data/hwsq_HARDWARE_{K0TAG}.npz")
    f_hw, G_hw = float(hw["f"]), float(hw["G"].mean())
    Ls = np.linspace(1.0, 12.0, 45)
    rows = []
    for fpath in sorted(glob.glob(f"data/2p_bits_g*_{K0TAG}.npz")):
        d = np.load(fpath)
        bits, g = d["bits"], float(d["g"])
        # S(q): site-only, independent of L (use nominal readout, L=1)
        bsig = apply_ro_split(bits, lat, 1.0, seed=11)
        S, _ = sq_and_witness(bsig, lat, QS)
        (f, c), *_ = np.linalg.lstsq(A, S, rcond=None)
        Smit = (S - c) / f
        corr = np.corrcoef(S, Si)[0, 1]
        # witness vs link-readout scale L
        wl = np.array([sq_and_witness(apply_ro_split(bits, lat, L, seed=13),
                                      lat, QS)[1] for L in Ls])
        Lstar = float(np.interp(G_hw, wl[::-1], Ls[::-1]))
        rows.append(dict(g=g, f=f, c=c, corr=corr, S=S, Smit=Smit,
                         Ls=Ls, wl=wl, Lstar=Lstar))
        log(f"g={g}: S(q) f={f:.3f} c={c:.3f} corr={corr:.3f}; "
            f"witness=G_hw({G_hw:.3f}) at L*={Lstar:.2f}")
    # g* = sampled gate scale whose S(q) best reproduces the hardware curve.
    # (f(g) is non-monotonic and shot-noisy, so select by shape fidelity --
    # correlation with the exact curve -- not by interpolating f.)
    gg = np.array([r["g"] for r in rows])
    ff = np.array([r["f"] for r in rows])
    r0 = max(rows, key=lambda r: r["corr"])
    gstar = r0["g"]
    np.savez(f"data/hwsq_2param_{K0TAG}.npz",
             gates=gg, fvals=ff, corrs=np.array([r["corr"] for r in rows]),
             Ls=Ls, wls=np.array([r["wl"] for r in rows]),
             Lstars=np.array([r["Lstar"] for r in rows]),
             gstar=gstar, Lstar=r0["Lstar"], f_hw=f_hw, G_hw=G_hw,
             S_at_gstar=r0["S"], Smit_at_gstar=r0["Smit"], S_ideal=Si, q=QS)
    log(f"HARDWARE f={f_hw:.3f} G={G_hw:.3f} -> g*={gstar:.2f}, "
        f"L*={r0['Lstar']:.2f} (link readout {r0['Lstar']:.1f}x site)")
