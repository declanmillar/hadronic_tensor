"""Shared helpers for the validated two-band ridge model (see
scripts/ridge_model_v2.py for the validation record)."""

import numpy as np

from htensor import Z2Lattice, exact
from htensor import currents as cur


def two_band_space(lat: Z2Lattice, m0, g2, eta, k_states: int = 14):
    e_all, v_all = exact.lowest_physical_states(lat, m0, g2, eta, k=k_states,
                                                matrix_free=True)
    gaps = e_all - e_all[0]
    keep = [i for i in range(k_states) if 2.0 < gaps[i] < 3.5]
    return ([v_all[:, i] for i in keep], e_all[keep],
            v_all[:, 0], float(e_all[0]))


def model_correlator(lat: Z2Lattice, sts, Ek, f, vc, times, m0, g2, eta):
    """Connected two-band-model correlator on the (t, v) grid, plus the
    local ME tensor."""
    Mv = np.array([[[np.vdot(sp, exact.apply_pauli_sum(
        cur.charge_density(lat, v), s)) for s in sts] for sp in sts]
        for v in range(lat.ns)])
    G = np.empty((len(times), lat.ns), complex)
    one0 = np.conj(f) @ Mv[vc] @ f
    for i, t in enumerate(times):
        a = np.exp(-1j * Ek * t) * f
        for v in range(lat.ns):
            two = np.conj(a) @ (Mv[v] @ (np.exp(-1j * Ek * t) * (Mv[vc] @ f)))
            G[i, v] = two - (np.conj(a) @ Mv[v] @ a) * one0
    return G, Mv


from htensor.analysis import onesided_ft  # noqa: E402  (moved 2026-09-02)
