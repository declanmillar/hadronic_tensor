"""Two-meson wavepacket construction and return-probability R(t) in the
gauge-fixed physical basis (task 15: DHK postdiction).

DHK (arXiv:2505.20408) prepare two well-separated meson wavepackets
    Psi_i(k) = N exp(-i k mu_i) exp(-(k - kbar_i)^2 / (4 sigma_i^2))
built from single-meson creation operators, and measure the return
probability R(t) = |<Psi|U(t)|Psi>|^2 (their Fig. 10, N_P = 13).

We reconstruct the state exactly: the position-space meson creation
operator O_x (a gauge-invariant fermion bilinear on the bond at physical
site x) is smeared into a single-packet operator M[phi_i] = sum_x phi_i(x)
O_x with phi_i(x) = sum_k Psi_i(k) e^{ikx}; the two-packet state is
M[phi_1] M[phi_2] |Omega> on the interacting vacuum.  Everything lives in
the 2^Ns physical (Gauss-resolved) basis, so N_s = 26 (their N_P = 13) is
a ~10^7-state exact Krylov evolution rather than a 2^52 statevector.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .lattice import Z2Lattice
from . import hamiltonian as ham
from .currents import bond_current
from .hamiltonian import hop_term
from .gaugefixed import PhysicalBasis


def gauge_fixed_system(lat: Z2Lattice, m0, g2, eta, n_band: int = 40,
                       full_band: bool = True, ncv: int | None = None,
                       degeneracy_tol: float = 1e-5):
    """(-> dict) vacuum, single-meson band, and reduced H in the Q=0
    physical sector.  full_band=False resolves only the vacuum and mass gap
    (k=4 eigsh) -- enough for R(t) at large volume where the full band
    eigensolve is the bottleneck.  degeneracy_tol: energy-cluster tolerance
    for the T2 resolution; must exceed eigsh's splitting of degenerate +-k
    pairs (~1e-6..1e-5) or the pair is silently dropped from the band."""
    basis = PhysicalBasis(lat)
    sel = np.flatnonzero(basis.q == 0)
    # build H directly on the Q=0 subspace (full-basis COO would exhaust
    # memory at large ns)
    H = basis.matrix(ham.build_hamiltonian(lat, m0, g2, eta), sub=sel).real
    k = min(n_band if full_band else 4, H.shape[0] - 2)
    w, v = spla.eigsh(H, k=k, which="SA", ncv=ncv)
    o = np.argsort(w)
    w, v = w[o], v[:, o]
    if not full_band:                          # skip the T2 build entirely
        return dict(basis=basis, sel=sel, H=H, T=None, vac=v[:, 0], evals=w,
                    evecs=v, band={}, band_states=[], M=float(w[1] - w[0]))
    T = basis.matrix_translation(sel) if hasattr(basis, "matrix_translation") \
        else basis.translation()[sel][:, sel]
    vac = v[:, 0]
    M = float(w[1] - w[0])                    # lightest meson (band-1 minimum)
    # single-meson band: states below the 2M two-meson threshold, with T2
    # momenta resolved cluster by cluster (eigvecs aligned to eigvals).
    band = {}
    band_states = []
    i = 1
    while i < len(w):
        j = i + 1
        while j < len(w) and w[j] - w[i] < degeneracy_tol:
            j += 1
        if w[i] - w[0] < 2 * M - 0.1:         # band-1 window
            blk = v[:, i:j]
            ev, U = np.linalg.eig(blk.conj().T @ (T @ blk))
            resolved = blk @ U
            for c in range(j - i):
                kk = round(float(np.angle(ev[c])), 4)
                st = resolved[:, c] / np.linalg.norm(resolved[:, c])
                # band[k] is the LOWEST level at each momentum (the single
                # meson); later levels in the window (band-2 / M* / two-
                # meson states at the same k) must not overwrite it.
                if kk not in band:
                    band[kk] = (w[i] - w[0], st)
                band_states.append(st)
        i = j
    return dict(basis=basis, sel=sel, H=H, T=T, vac=vac, evals=w, evecs=v,
                band=band, band_states=band_states, M=M)


def meson_operator(lat: Z2Lattice, basis: PhysicalBasis, sel, eta,
                   chi: float = 1.0):
    """Reduced position-space meson creation operators O_x on the Q=0
    physical basis: a hop + i*chi*current bilinear on the even bond at
    physical site x (chi tunes chirality so a complex envelope selects a
    definite-momentum branch).  Returns list over physical sites."""
    ops = []
    for x in range(lat.nx):
        bond = 2 * x
        O = (hop_term(lat, bond, eta) + 1j * chi * bond_current(lat, bond, eta))
        ops.append(basis.matrix(O, sub=sel))
    return ops


def packet_operator(meson_ops, kbar, sigma, mu, nx, staggered=False):
    """M[phi] = sum_x phi(x) O_x with phi(x) = sum_k Psi(k) e^{ikx},
    Psi(k) = exp(-i k x_mu) exp(-(k-kbar)^2/4 sigma^2), k = 2 pi j / nx.

    mu is the DHK center in STAGGERED-site units; the meson operators live
    on physical sites, so the physical center is x_mu = mu/2 (mod nx).  (The
    earlier physical=staggered identification wrapped mu=19 -> site 6 on the
    13-site ring, collapsing the two packets onto each other.)

    staggered=True: the DHK Gaussian (kbar, sigma) is written in the
    STAGGERED quasi-momentum k of the 2*nx-site lattice DHK actually simulate.
    A meson is a composite on a 2-staggered-site physical cell, so its COM
    momentum on the physical ring is K = 2 k (site doubling).  Referring the
    envelope to the physical momentum K that indexes the meson dispersion then
    maps (kbar, sigma) -> (2 kbar, 2 sigma): the single DISCRETE (1 vs 2)
    site-doubling ambiguity intrinsic to staggered fermions, no free knob.
    The displacement phase is unchanged because exp(-i k mu) = exp(-i K mu/2)
    = exp(-i K x_mu) with x_mu = mu/2 already the physical center."""
    if staggered:
        kbar, sigma = 2.0 * kbar, 2.0 * sigma
    x_mu = (mu / 2.0) % nx
    ks = 2 * np.pi * np.arange(nx) / nx
    ks = np.where(ks > np.pi, ks - 2 * np.pi, ks)
    Psi = np.exp(-1j * ks * x_mu) * np.exp(-((ks - kbar) ** 2) / (4 * sigma ** 2))
    xs = np.arange(nx)
    phi = (np.exp(1j * np.outer(xs, ks)) @ Psi)
    M = meson_ops[0] * phi[0]
    for x in range(1, nx):
        M = M + meson_ops[x] * phi[x]
    return M


def two_meson_state(gf, eta, packets, chi: float = 1.0, staggered=False):
    """|Psi> = M[phi_1] M[phi_2] |Omega>, normalized. packets: list of
    (kbar, sigma, mu). staggered=True forwards the site-doubling convention
    (K = 2 k) to packet_operator."""
    lat_nx = gf["basis"].lat.nx
    mops = meson_operator(gf["basis"].lat, gf["basis"], gf["sel"], eta, chi)
    psi = gf["vac"]
    for kbar, sigma, mu in packets:
        M = packet_operator(mops, kbar, sigma, mu, lat_nx, staggered=staggered)
        psi = M @ psi
    return psi / np.linalg.norm(psi)


def return_probability(gf, psi, times):
    """R(t) = |<psi|e^{-iHt}|psi>|^2 via Krylov; times must be sorted, t0=0."""
    H = gf["H"]
    R = np.empty(len(times))
    amp = np.empty(len(times), dtype=complex)
    prev = 0.0
    state = psi.copy()
    for i, t in enumerate(times):
        if t > prev:
            state = spla.expm_multiply(-1j * H * (t - prev), state)
            prev = t
        a = np.vdot(psi, state)
        amp[i] = a
        R[i] = np.abs(a) ** 2
    return R, amp
