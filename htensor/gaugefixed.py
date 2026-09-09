"""Gauge-fixed (physical-sector) exact spectroscopy.

In the product basis {sites in the Z basis} x {links in the X basis} every
Gauss operator G_n = (-1)^n Z_n X_{n-1,n} X_{n,n+1} is diagonal, so the
G_n = +1 sector is spanned by product states whose link pattern is fixed by
the matter configuration and a single holonomy bit:

    e_n = s_n e_{n-1},   e_n = 1 - 2 x_n,   s_n = (-1)^n (1 - 2 z_n),

with ring closure prod_n s_n = 1 selecting the admissible matter parities.
Any gauge-invariant Pauli string maps this basis to itself up to a sign, so
reduced matrices are built mechanically from the SAME SparsePauliOp objects
used everywhere else -- no rederivation of seam/JW/staggering conventions.
Dimensions: 2^ns physical states, a few thousand after the Q = 0 cut
(ns = 14: 6864), versus 2^(2 ns) for the penalized full-space solver.

Validation: scripts/deep_levels_gf.py reproduces the stored full-space
deep_levels_ns{8,10,12}.npz gaps and T2 phases.
"""

import numpy as np
import scipy.sparse as sp

from .lattice import Z2Lattice
from . import hamiltonian as ham
from .currents import charge_density


def _terms(op):
    """SparsePauliOp -> [({qubit: 'X'|'Y'|'Z'}, complex coeff), ...]."""
    op = op.simplify()
    n = op.num_qubits
    out = []
    for label, c in zip(op.paulis.to_labels(), op.coeffs):
        ops = {n - 1 - i: ch for i, ch in enumerate(label) if ch != "I"}
        out.append((ops, complex(c)))
    return out


class PhysicalBasis:
    """Enumeration of the full G_n = +1 product basis.

    NOTE: the basis is NOT restricted to a charge sector -- individual Pauli
    strings (e.g. the XX half of a hop) change Q, and only their sum
    cancels; Q = q blocks are exact principal submatrices AFTER summing
    (self.q gives the diagonal charge of each basis state)."""

    def __init__(self, lat: Z2Lattice):
        assert lat.pbc, "gauge-fixed basis implemented for PBC"
        ns = lat.ns
        # compact dtypes: at ns=26 the (2^ns, ns) bit arrays are 14 GB each
        # at int64; int8 (values in {-1,0,1}) cuts that 8x.
        z = np.arange(1 << ns, dtype=np.int64)
        zbit = ((z[:, None] >> np.arange(ns)) & 1).astype(np.int8)
        stag = ((-1) ** np.arange(ns)).astype(np.int8)[None, :]
        s = (stag * (1 - 2 * zbit)).astype(np.int8)
        keep = s.prod(axis=1, dtype=np.int8) == 1          # ring closure
        z = z[keep]
        zbit = zbit[keep]
        s = s[keep]
        q = ((stag - (1 - 2 * zbit)) // 2).sum(axis=1, dtype=np.int16)
        # link bits for both holonomies: e_n = e_0 * prod_{m<=n, m>=1} s_m
        cum = np.cumprod(np.concatenate(
            [np.ones((len(z), 1), dtype=np.int8), s[:, 1:]], axis=1),
            axis=1, dtype=np.int8)
        self.lat, self.ns = lat, ns
        self.z = np.concatenate([z, z])                    # index = (z, e0)
        self.q = np.concatenate([q, q])
        self.h = np.concatenate([np.zeros(len(z), np.int8),
                                 np.ones(len(z), np.int8)])
        e0 = (1 - 2 * self.h[:, None]).astype(np.int8)
        self.ebit = (np.concatenate([cum, cum]) * e0).astype(np.int8)  # +-1
        self.xbit = ((1 - self.ebit) // 2).astype(np.int8)
        self.dim = len(self.z)
        # lookup (z, holonomy) -> row
        self.lut = -np.ones(1 << (ns + 1), dtype=np.int64)
        self.lut[(self.z << 1) | self.h] = np.arange(self.dim)

    def _zbit(self, n):
        return (self.z >> n) & 1

    def matrix(self, op, sub=None) -> sp.csr_matrix:
        """Reduced matrix of a gauge-invariant SparsePauliOp.

        sub: optional array of basis indices defining an invariant subspace
        (e.g. the Q=0 sector).  When given, the matrix is built directly on
        that subspace -- essential at large ns where the full 2^ns COO would
        exhaust memory.  The operator must preserve the subspace."""
        lat, ns = self.lat, self.ns
        site_of = {lat.site_qubit(n): n for n in range(ns)}
        link_of = {lat.link_qubit(n): n for n in range(lat.n_links)}
        terms = _terms(op)
        full = sub is None
        idx = slice(None) if full else sub
        nsub = self.dim if full else len(sub)
        zc = self.z if full else self.z[sub]
        xc0 = self.xbit[:, 0] if full else self.xbit[sub, 0]
        inv = None
        if not full:
            inv = np.full(self.dim, -1, dtype=np.int64)
            inv[sub] = np.arange(nsub)
        # Accumulate COO triplets, flushing to a running CSR whenever the
        # buffer grows large.  Each term emits up to nsub entries; over ~100
        # terms that would be billions at large ns, so a running sum keeps
        # peak memory ~ nnz(final) + one batch rather than terms*nsub.
        M = sp.csr_matrix((nsub, nsub), dtype=complex)
        data, cols, rows_acc = [], [], []
        buffered = [0]
        arange_nsub = np.arange(nsub)

        def flush():
            if not data:
                return M
            d = np.concatenate(data)
            cc = np.concatenate(cols)
            rr = np.concatenate(rows_acc)
            if inv is not None:
                keep = cc >= 0
                d, cc, rr = d[keep], cc[keep], rr[keep]
            m2 = sp.coo_matrix((d, (rr, cc)), shape=(nsub, nsub)).tocsr()
            data.clear(); cols.clear(); rows_acc.clear()
            buffered[0] = 0
            return M + m2

        for ops, c in terms:
            phase = np.full(nsub, c, dtype=complex)
            zmask, lflip = 0, np.zeros(nsub, dtype=np.int64)
            for q, p in ops.items():
                if q in site_of:
                    n = site_of[q]
                    zn = (zc >> n) & 1
                    if p == "Z":
                        phase = phase * (1 - 2 * zn)
                    elif p == "X":
                        zmask |= 1 << n
                    else:                                   # Y
                        phase = phase * 1j * (1 - 2 * zn)
                        zmask |= 1 << n
                else:
                    n = link_of[q]
                    en = self.ebit[idx, n]
                    if p == "X":
                        phase = phase * en
                    elif p == "Z":
                        lflip = lflip ^ (1 << n)
                    else:                                   # Y
                        phase = phase * 1j * (-en)
                        lflip = lflip ^ (1 << n)
            z2 = zc ^ zmask
            h2 = xc0 ^ (lflip & 1)
            col = self.lut[(z2 << 1) | h2]
            if inv is None:
                if np.any(col < 0):
                    raise ValueError("operator leaves the physical sector")
            else:
                # Individual Pauli strings of a subspace-preserving operator
                # can carry out-of-subspace matrix elements that cancel
                # against a sibling term (e.g. the Q=+-2 pieces of XX and YY
                # sum to zero).  Drop them: exact for Q-conserving ops.
                col = inv[col]
            data.append(phase)
            cols.append(col)
            rows_acc.append(arange_nsub)
            buffered[0] += nsub
            if full and buffered[0] < (1 << 62):
                continue
            if buffered[0] > 40_000_000:        # ~1 GB of complex COO
                M = flush()
        M = flush()
        return M

    def reflection(self, shift: int = 0, stag_phase: bool = False
                   ) -> sp.csr_matrix:
        """Spatial reflection R: site n -> (shift - n) mod ns (shift even =
        site-centered, odd = bond-centered), link m -> (shift - 1 - m) mod
        ns, as a Fock-space operator R c_n R^dag = eta_n c_{r(n)}
        (eta_n = (-1)^n if stag_phase).  The fermionic reordering sign is
        the inversion parity of the occupied sites' images.  Which variant
        commutes with H is convention -- select by commutator test
        (select_reflection)."""
        ns, dim = self.ns, self.dim
        r = [(shift - n) % ns for n in range(ns)]
        rl = [(shift - 1 - m) % ns for m in range(self.lat.n_links)]
        zbit = [(self.z >> n) & 1 for n in range(ns)]
        z2 = np.zeros(dim, dtype=np.int64)
        for m in range(ns):                      # z'_m = z_{r(m)}
            z2 |= zbit[r[m]] << m
        par = np.zeros(dim, dtype=np.int64)      # inversion parity
        for i in range(ns):
            for j in range(i + 1, ns):
                if r[i] > r[j]:                  # image order inverted
                    par ^= zbit[i] & zbit[j]
        phase = 1.0 - 2.0 * par
        if stag_phase:
            odd = np.zeros(dim, dtype=np.int64)
            for n in range(1, ns, 2):
                odd ^= zbit[n]
            phase = phase * (1.0 - 2.0 * odd)
        # links permute without fermionic signs; new holonomy = x'_0
        h2 = self.xbit[:, rl.index(0)]
        col = self.lut[(z2 << 1) | h2]
        if np.any(col < 0):
            raise ValueError("reflection left the physical sector")
        return sp.coo_matrix((phase, (col, np.arange(dim))),
                             shape=(dim, dim)).tocsr()

    def select_reflection(self, H: sp.csr_matrix) -> tuple:
        """Try (shift, stag_phase) variants; return (R, variant) for the one
        that commutes with H (and satisfies R^2 = 1, R T R^dag = T^dag)."""
        T = self.translation()
        for shift in (0, 1, 2):
            for stag in (False, True):
                R = self.reflection(shift, stag)
                if abs((H @ R - R @ H)).max() > 1e-9:
                    continue
                assert abs((R @ R) - sp.identity(self.dim)).max() < 1e-9
                assert abs(R @ T @ R.conj().T.tocsr()
                           - T.conj().T.tocsr()).max() < 1e-9
                return R, (shift, stag)
        raise ValueError("no reflection variant commutes with H")

    def translation(self) -> sp.csr_matrix:
        """Reduced T2 (one-spatial-site translation), matching
        spectroscopy.translate: optional string twist, then site n+2 -> n."""
        lat, ns = self.lat, self.ns
        phase = np.ones(self.dim, dtype=complex)
        if lat.ns % 4 == 0:
            tw = np.ones(self.dim, dtype=np.int64)
            for n in range(2, ns):
                tw = tw * (1 - 2 * self._zbit(n))
            phase = -phase * tw
        # rotate bits: new bit n = old bit (n+2) mod ns
        def rot(v, nbits):
            return ((v >> 2) | (v << (nbits - 2))) & ((1 << nbits) - 1)
        z2 = rot(self.z, ns)
        # new link n = old link n+2 -> new holonomy = old xbit[2]
        h2 = self.xbit[:, 2 % lat.n_links]
        col = self.lut[(z2 << 1) | h2]
        if np.any(col < 0):
            raise ValueError("translation left the sector (check ns mod 4)")
        # T2|j> = phase(j) |target(j)>: entry at (row=target, col=source)
        return sp.coo_matrix((phase, (col, np.arange(self.dim))),
                             shape=(self.dim, self.dim)).tocsr()


def deep_spectrum(lat: Z2Lattice, m0, g2, eta, k: int | None = None,
                  degeneracy_tol: float = 1e-5, refl: bool = False):
    """(gaps, T2 phases, energies[, R parities]) in the physical Q=0 sector.

    k=None -> dense eigh, ALL levels; else sparse eigsh lowest k.
    refl=True also returns the reflection parity for P = 0 levels
    (NaN for P != 0, where R maps k -> -k instead of labeling)."""
    basis = PhysicalBasis(lat)
    sel = np.flatnonzero(basis.q == 0)
    H = basis.matrix(ham.build_hamiltonian(lat, m0, g2, eta))
    H = H[sel][:, sel]
    assert np.abs(H.imag).max() < 1e-12
    if k is None or len(sel) <= 4096:
        w, v = np.linalg.eigh(H.real.toarray())
        if k is not None:
            w, v = w[:k], v[:, :k]
    else:
        w, v = sp.linalg.eigsh(H.real, k=k, which="SA")
        o = np.argsort(w)
        w, v = w[o], v[:, o]
    T = basis.translation()[sel][:, sel]
    R = None
    if refl:
        Rfull, variant = basis.select_reflection(
            basis.matrix(ham.build_hamiltonian(lat, m0, g2, eta)))
        R = Rfull[sel][:, sel]
    phases = np.empty(len(w))
    parities = np.full(len(w), np.nan)
    i = 0
    while i < len(w):
        j = i + 1
        while j < len(w) and w[j] - w[i] < degeneracy_tol:
            j += 1
        blk = v[:, i:j]
        tb = blk.conj().T @ (T @ blk)
        ev, U = np.linalg.eig(tb)
        order = np.argsort(np.angle(ev))
        ev, U = ev[order], U[:, order]
        phases[i:j] = np.angle(ev)
        if R is not None:
            resolved = blk @ U
            p0 = [m for m in range(j - i) if abs(np.angle(ev[m])) < 1e-4]
            if p0:
                # R preserves the P=0 subspace of the cluster; diagonalize
                # its block there (a P=0 pair can be R-mixed)
                sub = resolved[:, p0]
                q, _ = np.linalg.qr(sub)
                rb = q.conj().T @ (R @ q)
                pe = np.sort(np.real(np.linalg.eigvals(rb)))[::-1]
                for m, val in zip(p0, pe):
                    parities[i + m] = val
        i = j
    if R is not None and np.isfinite(parities[0]):
        # normalize to the vacuum: the raw Fock reordering sign on the
        # filled staggered sea is (-1)^(k'(k'-1)/2), k' = ns/2, which
        # alternates with volume (period 4 in L); physical parity is
        # relative to the vacuum
        parities = parities * parities[0]
    out = (w - w[0], phases, w)
    return out + ((parities,) if refl else ())
