"""Factorization test at CGK-A: do the hadronic tensor's two-meson
spectral residues factorize into an infinite-volume amplitude times a
phase-shift-determined kinematic factor?  (task 17)

Answers Henry's question -- can W (convolved with kinematics) replace the
Luescher route for scattering?  The Lellouch-Luescher relation says a
finite-volume matrix element factorizes as
    |<n|O|0>|^2  =  |F(E_n)|^2 / rho(E_n),
    rho(E) = (1/2pi) d(pL + 2 delta)/dE   (density of two-meson states),
so |F(E)|^2 = |<n|O|0>|^2 * rho(E_n) must be VOLUME-INDEPENDENT if the
current's two-meson coupling is governed by the same delta(E) that
Luescher extracts from the spectrum.  Collapse across volumes => the
hadronic tensor and the phase shift carry the same scattering content
(factorization holds, but still needs delta from the finite-volume
quantization -- it does not bypass Luescher).

O is the P=0, parity-even meson-pair interpolator (translation-summed hop
bilinear).  delta(E) is extracted from the CGK-A two-meson spectrum by the
1+1d Luescher condition, exactly as in scripts/phase_shifts_v2.py.

  PYTHONPATH=. .venv/bin/python scripts/factorization.py
"""

import os
import sys as _s

import numpy as np
import scipy.sparse.linalg as spla

# Plot-only fast path: reuse the saved analysis npz unless --recompute.
_CACHE = "data/factorization_cgkA.npz"
FAST = os.path.exists(_CACHE) and "--recompute" not in _s.argv
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from htensor import Z2Lattice
from htensor import hamiltonian as ham
from htensor.hamiltonian import hop_term
from htensor.gaugefixed import PhysicalBasis

M0, G2, ETA = 0.1, 0.4, 1.0            # CGK-A
VOLS = (8, 10, 12, 14, 16, 18, 20, 22, 24)

# single-meson dispersion E(k) from the ns=20 CGK-A band
d20 = np.load("data/deep_levels_cgkinA_ns20.npz")
g, ph = d20["gaps"], d20["phases"]
ks, es = [], []
for kk in np.unique(np.round(np.abs(ph), 6)):
    m = np.isclose(np.abs(ph), kk) & (g > 0.1) & (g < 1.5)
    if m.any():
        ks.append(kk); es.append(g[m].min())
E = CubicSpline(np.array(ks), np.array(es), bc_type=((1, 0.0), (1, 0.0)))
M = float(E(0)); THR = 2.2651
print(f"CGK-A: M = {M:.4f}, 2M = {2*M:.4f}, MM' threshold = {THR:.4f}")
p_of = lambda E2: brentq(lambda q: 2 * E(q) - E2, 1e-9, np.pi)


def delta(E2, L, n):
    p = p_of(E2)
    dl = (2 * np.pi * n - p * L) / 2
    return (dl + np.pi / 2) % np.pi - np.pi / 2, p


# collect (E, |F|^2) across volumes
if FAST and "dl" in np.load(_CACHE):
    _c = np.load(_CACHE)
    Es, F2, nss, dls = _c["E"], _c["F2"], _c["ns"], _c["dl"]
    f2b = F2[Es < 1.9]
    print(f"[fast path] loaded {_CACHE}; --recompute to re-derive")
else:
    FAST = False
pts = []
for ns in ([] if FAST else VOLS):
    raw = f"data/fact_raw_ns{ns}.npz"
    if os.path.exists(raw):                # remote/cached eigsh (fact_raw_volume.py)
        _r = np.load(raw)
        dE, res = _r["dE"], _r["res"]
        print(f"  ns={ns}: loaded raw spectrum from {raw}")
    else:
        lat = Z2Lattice(ns, pbc=True)
        basis = PhysicalBasis(lat)
        sel = np.flatnonzero(basis.q == 0)
        H = basis.matrix(ham.build_hamiltonian(lat, M0, G2, ETA),
                         sub=sel).real
        O = basis.matrix(sum(hop_term(lat, b, ETA)
                             for b in range(lat.n_links)), sub=sel).real
        k = min(120, H.shape[0] - 2)
        w, v = spla.eigsh(H, k=k, which="SA")
        o = np.argsort(w); w, v = w[o], v[:, o]
        Ov = O @ v[:, 0]
        res = np.abs(v.conj().T @ Ov) ** 2
        dE = w - w[0]
        np.savez(raw, dE=dE, res=res, ns=ns, m0=M0, g2=G2, eta=ETA, k=k)
        print(f"  ns={ns}: saved raw spectrum -> {raw}")
    # two-meson P=0 even levels in the elastic window (match to spectrum)
    L = ns // 2
    free = [2 * float(E(2 * np.pi * nn / L)) for nn in range(L // 2 + 1)]
    cand = [(dE[i], res[i]) for i in range(len(dE))
            if 2 * M + 1e-3 < dE[i] < THR - 2e-3 and res[i] > 1e-6]
    # assign n by counting, compute delta and rho, |F|^2 = res * rho
    cand.sort()
    if ns <= 10:
        # tiny volumes: only the n=0 level is a clean MM scattering state;
        # the window above ~1.95 is occupied by tower/resonance-core states
        cand = cand[:1]
    for nn, (E2, r) in enumerate(cand):
        dl, p = delta(E2, L, nn)
        # rho = (1/2pi) d(pL+2delta)/dE ; dp/dE from dispersion, ddelta/dp
        dpdE = 1.0 / (2 * E(p, 1)) if E(p, 1) != 0 else np.nan
        # ddelta/dE numerically from neighboring volumes later; use L term +
        # local finite-diff of delta vs p across this volume's levels
        rho_L = L * dpdE / (2 * np.pi)
        pts.append((E2, r, p, dl, rho_L, ns))
        print(f"  ns={ns} n={nn}: E={E2:.4f} p={p:.4f} delta={dl:+.4f} "
              f"res={r:.4e}")

if not FAST:
    pts = np.array(pts)
    Es, res, ps, dls, rhoL, nss = pts.T
    # ddelta/dE from a spline through ALL elastic levels: below the MM'
    # threshold the single-channel delta(E) is one smooth function, so every
    # level constrains it (the earlier E<1.9 truncation zeroed the
    # phase-shift term of rho for the upper levels and distorted |F|^2
    # there).  Average duplicate energies before splining.
    order = np.argsort(Es)
    Eu, du = [], []
    # L=4 (ns=8) has M*L ~ 3.4: its Luscher delta carries O(20%) volume
    # corrections and is excluded from the delta(E) spline (its |F|^2 is
    # still evaluated, with the smooth-curve derivative).
    for e, dl, _ns in zip(Es[order], dls[order], nss[order]):
        if int(_ns) == 8:
            continue
        if Eu and abs(e - Eu[-1]) < 1e-6:
            du[-1] = 0.5 * (du[-1] + dl)
        else:
            Eu.append(e); du.append(dl)
    # unwrap the mod-pi fold (delta rises through +pi/2 near E ~ 2.1)
    du = np.unwrap(np.array(du), period=np.pi)
    dspl = CubicSpline(np.array(Eu), du) if len(Eu) > 3 else None
    ddeltadE = np.array([dspl(e, 1) if dspl is not None else 0.0
                         for e in Es])
    rho = rhoL + ddeltadE / np.pi
    # O is translation-summed (res ~ Ns^2 x local), so the volume-stable
    # object is the per-length spectral density: F2 = res * rho / L
    F2 = res * rho / (nss.astype(int) // 2)

    # collapse metric on the GROUND (n=0) branch, L >= 5
    nidx0 = np.zeros(len(Es), dtype=int)
    for _v in set(nss.astype(int)):
        _m = nss == _v
        nidx0[_m] = np.argsort(np.argsort(Es[_m]))
    branch = [(Es[i], F2[i], int(nss[i])) for i in range(len(Es))
              if nidx0[i] == 0 and int(nss[i]) >= 14]
    f2b = np.array([b[1] for b in branch])
    print("\nnear-threshold branch |F(E)|^2 = residue x rho:")
    for E2, f2, ns in sorted(branch):
        print(f"  E={E2:.4f}  |F|^2={f2:.4f}  (L={ns//2})")
    spread = (f2b.max() - f2b.min()) / f2b.mean()
    print(f"collapse: mean |F|^2 = {f2b.mean():.4f}, "
          f"spread (max-min)/mean = {spread:.1%} across L=7..12")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit as _cfit
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paper_style import OI, MARKERS

    ER_, G_ = 2.128, 0.277
    ee = np.linspace(1.72, 2.30, 300)
    sig = 0.10
    # volumes displayed: L >= 7, colored/marked in fig:delta's sequence
    # (first shown volume -> OI[0]/MARKERS[0], filled, exactly as fig 5)
    shown = [v for v in sorted(set(nss.astype(int))) if v // 2 >= 7]
    style = {v: (OI[i], MARKERS[i]) for i, v in enumerate(shown)}

    fig, (axd, ax) = plt.subplots(
        2, 1, sharex=True, figsize=(3.6, 4.3), constrained_layout=True,
        height_ratios=[1.0, 1.15])
    for a in (axd, ax):
        a.axvspan(ER_ - G_/2, ER_ + G_/2, color="0.93", zorder=0)
        a.axvline(ER_, ls=":", color="0.55", lw=0.9)

    # ---- top: elastic phase shift, unwrapped, with BW + linear fit
    msk = np.isin(nss.astype(int), shown)
    o = np.argsort(Es[msk])
    Eo = Es[msk][o]
    dlo = np.unwrap(dls[msk][o], period=np.pi)
    vo = nss[msk][o].astype(int)

    def dmodel(E, a, b, er, g):
        return a + b * (E - 2 * M) + np.pi / 2 + np.arctan(2 * (E - er) / g)

    pfit, _ = _cfit(dmodel, Eo, dlo, p0=[0.0, 0.5, ER_, G_])
    rms = np.sqrt(np.mean((dlo - dmodel(Eo, *pfit)) ** 2))
    print(f"BW+linear fit: E_R = {pfit[2]:.4f}, Gamma = {pfit[3]:.4f}, "
          f"rms = {rms:.3f} rad")
    axd.plot(ee, dmodel(ee, *pfit), "-", color="0.35", lw=1.2, zorder=2)
    for v in shown:
        c, mk = style[v]
        m = vo == v
        axd.plot(Eo[m], dlo[m], mk, color=c, ms=5, lw=0,
                 label=f"$N_x={v//2}$", zorder=6)
    axd.axhline(np.pi / 2, color="0.7", lw=0.6, ls=":")
    axd.text(1.735, np.pi / 2 + 0.07, r"$\pi/2$", fontsize=7, color="0.4")
    axd.set_ylabel(r"elastic phase shift $\delta(E)$")
    axd.legend(fontsize=6.2, ncol=3, loc="upper left", framealpha=0.9,
               columnspacing=0.7, handletextpad=0.35)

    # ---- bottom: per-length spectral density.  Unsmeared levels (markers +
    # thin interpolant), identically smeared N_x = 10-12 ED spectra (mean +
    # min/max envelope; parameter-free), and the real-time quench curves
    # (data/quench_SE_curves.npz; see stageB_quench_seq.py) at matched
    # sigma = 0.10 and at the ordered-limit-honest sigma = 0.05.
    curves = []
    for v in (20, 22, 24):
        dr = np.load(f"data/fact_raw_ns{v}.npz")
        el = dr["dE"] > 1.70               # elastic window: exclude the
        rr, de = dr["res"][el], dr["dE"][el]   # sub-threshold scalar pole
        curves.append(np.array([(rr * np.exp(-((de-e)**2)/(2*sig**2))).sum()
                      / (np.sqrt(2*np.pi)*sig) / (v//2) for e in ee]))
    c = np.array(curves)
    from scipy.interpolate import UnivariateSpline
    _o = np.argsort(Es[msk])
    _sp = UnivariateSpline(Es[msk][_o], F2[msk][_o],
                           s=msk.sum() * (0.05 * F2[msk].max())**2)
    ax.plot(ee, _sp(ee), "-", color="0.6", lw=0.8, zorder=2)
    ax.fill_between(ee, c.min(0), c.max(0), color="0.78", lw=0, zorder=1)
    ax.plot(ee, c.mean(0), "-", color="0.35", lw=1.2, zorder=3,
            label=r"ED, $\sigma_E{=}0.10$")
    try:
        q = np.load("data/quench_SE_curves.npz")
        S10 = np.interp(ee, q["E"], q["S10"])
        S05 = np.interp(ee, q["E"], q["S05"])
        ax.plot(ee, S10, "-", color=OI[0], lw=1.4, zorder=4,
                label=r"real time, $\sigma_E{=}0.10$")
        ax.plot(ee, S05, "--", color=OI[0], lw=1.4, zorder=5,
                label=r"real time, $\sigma_E{=}0.05$")
    except OSError:
        print("quench curves missing; bottom panel drawn without them")
    for v in shown:
        cc, mk = style[v]
        m = nss == v
        ax.plot(Es[m], F2[m], mk, color=cc, ms=5, lw=0, zorder=6)
    ax.text(ER_, 0.0018, r"$E_R$", fontsize=7.5, color="0.35", ha="center")
    ax.set_xlabel("$E$ (two-meson energy)")
    ax.set_ylabel(r"spectral density $\,|\langle n|O|0\rangle|^2\rho/N_x$")
    ax.set_xlim(1.72, 2.30)
    ax.set_ylim(0, 0.049)
    ax.legend(fontsize=6.2, loc="upper left", framealpha=0.9)
    fig.savefig("data/factorization_cgkA.pdf", dpi=200)
    print("wrote data/factorization_cgkA.pdf")
    mid = (ee > 1.78) & (ee < 2.24)
    rel = (c.max(0) - c.min(0)) / c.mean(0)
    print(f"smeared-curve volume spread: median {np.median(rel[mid]):.1%}, "
          f"max {rel[mid].max():.1%} on [1.78, 2.24]")
except Exception as e:
    print("plot skipped:", e)

if not FAST:
    nidx = np.zeros(len(Es), dtype=int)
    for _v in set(nss.astype(int)):
        _m = nss == _v
        nidx[_m] = np.argsort(np.argsort(Es[_m]))
    np.savez("data/factorization_cgkA.npz", E=Es, F2=F2, res=res, rho=rho,
             ns=nss, nidx=nidx, dl=dls, p=ps, M=M, thr=THR,
             branch_mean=f2b.mean(), branch_spread=spread)
    print("saved data/factorization_cgkA.npz")
