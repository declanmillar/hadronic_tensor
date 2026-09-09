"""Exact spectroscopy: translation quantum numbers, the single-meson band,
and boosted Gaussian meson wavepackets (small-volume ground truth).

The wavepacket construction is phase-convention-safe: a localized
interpolating operator (the vector bond current, i.e. exactly the current
that appears in W^{mu nu}) is applied to the vacuum with a boosted Gaussian
envelope, then projected onto the single-meson band with sum_s |s><s| --
independent of the arbitrary phases eigsh attaches to each |s>.
"""

import numpy as np

from .lattice import Z2Lattice
from . import hamiltonian as ham
from . import currents as cur
from . import exact
from .pauli import pauli_term


# ------------------------------------------------------------- translations
def _qubit_shift(psi: np.ndarray, n_qubits: int, shift: int = 4) -> np.ndarray:
    """Cyclic qubit relabeling: content of qubit (q + shift) moves to qubit q."""
    tensor = psi.reshape((2,) * n_qubits)
    perm_axis = [n_qubits - 1 - ((n_qubits - 1 - ax + shift) % n_qubits)
                 for ax in range(n_qubits)]
    return np.transpose(tensor, axes=perm_axis).reshape(-1)


def translate(psi: np.ndarray, lat: Z2Lattice) -> np.ndarray:
    """One-spatial-site translation T2, exact on the physical (Gauss) sector.

    In the JW representation the seam string is pinned to the qubit origin,
    so the naive qubit shift is the translation symmetry only for
    ns = 2 (mod 4), where the physical-sector seam sign is +1 (periodic
    wrap).  For ns = 0 (mod 4) (antiperiodic-like wrap, seam sign -1) the
    qubit shift must be preceded by the compensating string twist
    V = prod_{n=2}^{ns-1} Z_n, which relocates the wrap sign along with the
    seam.  [T2, H] = 0 holds on physical states only -- this operator is NOT
    a symmetry of the full JW Hilbert space, a point the paper must state.
    """
    if lat.pbc and lat.ns % 4 == 0:
        mask = 0
        for n in range(2, lat.ns):
            mask |= 1 << lat.site_qubit(n)
        signs = 1.0 - 2.0 * (np.bitwise_count(
            np.arange(psi.size, dtype=np.uint64) & np.uint64(mask)) % 2)
        # overall -1 normalizes the twist so the vacuum carries k = 0
        # (V on the strong-coupling vacuum = (-1)^(ns/2 - 1) = -1 here)
        psi = -psi * signs
    return _qubit_shift(psi, lat.n_qubits, 4)


def _t2_phases(states: list[np.ndarray], energies: np.ndarray, lat: Z2Lattice,
               degeneracy_tol: float = 1e-5):
    """Resolve T2 (one-spatial-site translation) eigenphases, diagonalizing
    T2 inside each degenerate energy cluster.

    degeneracy_tol must exceed eigsh's numerical splitting of a degenerate
    +-k pair (typically ~1e-6 .. 1e-5 in absolute energy): a pair split
    wider than the tolerance lands in two 1-dim clusters whose real
    eigenvectors have no definite T2 phase, and the momentum is lost."""
    resolved_states, phases = [], []
    i = 0
    while i < len(states):
        j = i + 1
        while j < len(states) and abs(energies[j] - energies[i]) < degeneracy_tol:
            j += 1
        block = states[i:j]
        tblock = [translate(s, lat) for s in block]
        m = np.array([[np.vdot(a, tb) for tb in tblock] for a in block])
        vals, vecs = np.linalg.eig(m)
        for col, lam in enumerate(vals):
            s_new = sum(vecs[k, col] * block[k] for k in range(len(block)))
            s_new = s_new / np.linalg.norm(s_new)
            resolved_states.append(s_new)
            phases.append(np.angle(lam))
        i = j
    return resolved_states, np.array(phases)


# ------------------------------------------------------------- meson band
def meson_band(lat: Z2Lattice, m0, g2, eta, n_states: int | None = None,
               matrix_free: bool = False, ncv: int | None = None,
               degeneracy_tol: float = 1e-5) -> dict:
    """Vacuum + lowest physical Q=0 states resolved by momentum.

    Returns {vacuum, e0, k, energy, states}: for each momentum k = 2 pi j / Nx
    the lowest excited state and energy gap; the single-meson band when the
    lowest excitation per momentum is one meson (true at strong-ish coupling).
    matrix_free/ncv: see exact.lowest_physical_states (use above ~22 qubits);
    with a small n_states only the low-|k| part of the band is resolved.
    degeneracy_tol: energy-cluster tolerance for the T2 resolution (see
    _t2_phases); a warning is issued for every grid momentum that receives
    no band state.
    """
    import warnings

    nx = lat.nx
    if n_states is None:
        n_states = 2 * nx + 4
    energies, vecs = exact.lowest_physical_states(lat, m0, g2, eta, k=n_states,
                                                  matrix_free=matrix_free,
                                                  ncv=ncv)
    states = [vecs[:, i] for i in range(vecs.shape[1])]
    resolved, phases = _t2_phases(states, energies, lat,
                                  degeneracy_tol=degeneracy_tol)
    # recompute energies for resolved combinations (unchanged within clusters)
    H_op = ham.build_hamiltonian(lat, m0, g2, eta)
    if matrix_free:
        e_res = np.array([np.real(np.vdot(s, exact.apply_pauli_sum(H_op, s)))
                          for s in resolved])
    else:
        H = exact.to_sparse(H_op)
        e_res = np.array([np.real(np.vdot(s, H @ s)) for s in resolved])
    order = np.argsort(e_res)
    resolved = [resolved[i] for i in order]
    e_res, phases = e_res[order], phases[order]

    vacuum, e0 = resolved[0], e_res[0]
    k_grid = 2 * np.pi * np.arange(nx) / nx  # in (-pi, pi] after wrap
    k_grid = np.where(k_grid > np.pi + 1e-9, k_grid - 2 * np.pi, k_grid)
    band_k, band_e, band_states = [], [], []
    for k in sorted(set(np.round(k_grid, 12))):
        for s, e, ph in zip(resolved[1:], e_res[1:], phases[1:]):
            # convention: T2|k> = e^{+ik}|k>, aligned so a wavepacket built
            # with envelope e^{+i k0 x} has arg<T2> = +k0 (verified at
            # ns = 6 and 8); absolute direction vs +x is anchored against
            # the group velocity in the analysis layer
            if abs((np.angle(np.exp(1j * (k - ph))))) < 1e-4:
                band_k.append(k)
                band_e.append(e - e0)
                band_states.append(s)
                break
        else:
            warnings.warn(
                f"meson_band: no state with T2 phase {k:+.4f} among the "
                f"{len(resolved) - 1} excited levels (ns={lat.ns}); raise "
                f"n_states, or degeneracy_tol if a +-k pair was split",
                RuntimeWarning, stacklevel=2)
    return {"vacuum": vacuum, "e0": e0, "k": np.array(band_k),
            "energy": np.array(band_e), "states": band_states,
            "all_energies": e_res, "all_phases": phases}


# ------------------------------------------------------------- wavepacket
def _interp_op(lat: Z2Lattice, bond: int, kind: str):
    if kind == "cur":
        return cur.bond_current(lat, bond, eta=1.0)
    if kind == "hop":
        from . import hamiltonian as _h
        return _h.hop_term(lat, bond, eta=1.0)
    raise ValueError(kind)


def _packet_vector(lat: Z2Lattice, vac: np.ndarray, kind: str, parity: int,
                   k0: float, sigma_x: float, x0: int) -> np.ndarray:
    """Sum_x f(x) e^{i k0 (x - x0)} O_{kind}(bond 2x + parity) |vac>."""
    nx = lat.nx
    out = np.zeros_like(vac)
    for x in range(nx):
        pos = x + (0.25 if parity == 0 else 0.75)  # bond midpoint, spatial units
        dd = pos - (x0 + 0.25)
        d = (dd + nx / 2) % nx - nx / 2
        f = np.exp(-d**2 / (4 * sigma_x**2)) * np.exp(1j * k0 * d)
        out = out + f * exact.apply_pauli_sum(_interp_op(lat, 2 * x + parity, kind), vac)
    return out


def optimize_interpolator(lat: Z2Lattice, band: dict, k0: float = 0.0,
                          sigma_x: float = 1.0, x0: int | None = None) -> dict:
    """Best local interpolator mix over {cur, hop} x {even, odd bonds}:
    maximize the meson-band fraction of the packet-smeared operator acting on
    the vacuum (generalized Rayleigh quotient, 4x4).  Classical, cheap, and
    the coefficients are volume-independent by locality."""
    import scipy.linalg

    if x0 is None:
        x0 = lat.nx // 2
    basis = [("cur", 0), ("cur", 1), ("hop", 0), ("hop", 1)]
    vac = band["vacuum"]
    us = [_packet_vector(lat, vac, k, p, k0, sigma_x, x0) for k, p in basis]
    pus = []
    for u in us:
        pu = np.zeros_like(u)
        for s in band["states"]:
            pu = pu + s * np.vdot(s, u)
        pus.append(pu)
    A = np.array([[np.vdot(pi, pj) for pj in pus] for pi in pus])
    B = np.array([[np.vdot(ui, uj) for uj in us] for ui in us])
    B = B + 1e-12 * np.eye(len(basis))
    vals, vecs = scipy.linalg.eigh(A, B)
    c = vecs[:, -1]
    return {"mix": dict(zip(basis, c)), "band_fraction": float(vals[-1].real),
            "basis": basis}


def meson_wavepacket(lat: Z2Lattice, band: dict, k0: float = 0.0,
                     sigma_x: float = 1.0, x0: int | None = None,
                     mix: dict | None = None):
    """Boosted Gaussian meson wavepacket, band-projected (see module docstring).

    k0 in units of inverse spatial sites (2 pi j / Nx), sigma_x / x0 in
    spatial sites; x0 defaults to mid-ring.  mix: {(kind, parity): coeff}
    interpolator mix (default: the bare even-bond vector current); use
    optimize_interpolator for a high-band-fraction target.
    Returns (state, band_fraction_of_raw).
    """
    if x0 is None:
        x0 = lat.nx // 2
    if mix is None:
        mix = {("cur", 0): 1.0}
    vac = band["vacuum"]
    raw = np.zeros_like(vac)
    for (kind, parity), c in mix.items():
        raw = raw + c * _packet_vector(lat, vac, kind, parity, k0, sigma_x, x0)
    proj = np.zeros_like(raw)
    for s in band["states"]:
        proj = proj + s * np.vdot(s, raw)
    norm = np.linalg.norm(proj)
    if norm < 1e-12:
        raise RuntimeError("interpolator has no overlap with the meson band")
    return proj / norm, float(norm / np.linalg.norm(raw))


def electric_energy_profile(lat: Z2Lattice, psi: np.ndarray, vacuum: np.ndarray,
                            g2: float = 1.0) -> np.ndarray:
    """Vacuum-subtracted electric (link) energy per spatial site.  The gauge
    term is +(g2/2) sigma^x, the interacting vacuum favors <sigma^x> < 0,
    and the meson's flux excitation raises the local electric energy, so the
    profile is POSITIVE and peaked at the packet center."""
    out = np.empty(lat.nx)
    for x in range(lat.nx):
        val = 0.0
        for b in (2 * x, 2 * x + 1):
            op = exact.to_sparse(
                pauli_term(lat.n_qubits, {lat.link_qubit(b): "X"}, g2 / 2))
            val += np.real(np.vdot(psi, op @ psi) - np.vdot(vacuum, op @ vacuum))
        out[x] = val  # excitation energy above vacuum
    return out
