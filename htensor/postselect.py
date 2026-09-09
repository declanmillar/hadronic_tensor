"""Gauge-sector post-selection, ranked by distance from the observable.

The Z2 Gauss law gives 50 commuting stabilizers on the Ns = 50 ring,

    G_n = (-1)^n Z_n X_{L,n-1} X_{L,n},        G_n |physical> = +|physical>,

all of which are measured for free in the Z readout setting (matter in Z,
links in X).  A noisy shot that leaves the physical subspace shows up as
G_n = -1 somewhere, so the shots can be filtered on the syndrome.

Demanding all 50 checks is useless in practice: with 30k shots on
ibm_kingston the all-checks acceptance is far below the shot floor.  The
checks are therefore **ranked by distance from the observable** and only the
top-ranked ones are imposed.  For a correlator anchored at site ``center``,
the rank of check n is |n - center| on the ring, and ``window`` sets how many
ranks are kept: window = 2 imposes the five checks n = center-2 .. center+2.
This is the trade the ranking exists to make -- the errors that matter for
<J0(x) J0(center)> are the ones near its support, and each extra rank costs
shots.

Truth-independent: the criterion is the stabilizer eigenvalue, never tuned
against the classical answer.  It is also **equal-time only**.  A t > 0
circuit's syndrome is measured after the evolution, so selecting on it
post-selects the final state rather than projecting the whole trajectory;
the t = 0 slice carries no q^0 structure, so this gain does not propagate
into the assembled tensor.

A wrong logical-to-column mapping shows up here first: on the released
kingston sample the acceptance is 0.388 at window 1, and 0.072 with the
columns permuted (scripts/check_raw_pubs.py).

Canonical implementation of the logic that appears inline in
scripts/hw_w00_coarse.py:29-70 and scripts/hw_cloud_figure.py:36-61;
`tests/test_postselect.py` asserts the three agree bit for bit.
"""

import numpy as np


def gauss_column(bits, lat, n: int) -> np.ndarray:
    """Per-shot value of the Gauss stabilizer G_n, as +-1.

    ``bits`` is (shots, n_qubits) uint8 from the Z readout setting: matter
    qubits measured in Z, link qubits in X (a Hadamard before readout), so
    the link bits already carry the X eigenvalue.
    """
    b = np.asarray(bits)
    return ((-1) ** n
            * (1 - 2 * b[:, lat.site_qubit(n)].astype(np.int8))
            * (1 - 2 * b[:, lat.link_qubit(n - 1)].astype(np.int8))
            * (1 - 2 * b[:, lat.link_qubit(n)].astype(np.int8)))


def check_ranking(lat, center: int, window: int) -> list[int]:
    """The checks imposed at a given window, nearest the observable first.
    Ring distance, so the seam is handled like any other neighbour."""
    ns = lat.ns
    return [(center + d) % ns for r in range(window + 1)
            for d in ((0,) if r == 0 else (-r, r))]


def keep_mask(bits, lat, center: int, window: int = 2) -> np.ndarray:
    """Boolean mask of the shots that pass every imposed check.
    ``window = 0`` imposes nothing and keeps everything (the raw sample)."""
    sel = np.ones(len(bits), bool)
    if window <= 0:
        return sel
    for n in check_ranking(lat, center, window):
        sel &= gauss_column(bits, lat, n) > 0
    return sel


def acceptance(bits, lat, center: int, windows=range(0, 8)) -> dict:
    """Keep fraction versus window: the cost side of the trade.
    -> {window: (kept, fraction)}."""
    return {w: (int(m.sum()), float(m.mean()))
            for w in windows for m in [keep_mask(bits, lat, center, w)]}


def connected_j0_correlator(bits, lat, center: int):
    """<J0(x) J0(center)>_conn and its shot error from Z-setting bits, with
    J0(v) = ((-1)^v - Z_v)/2.  Connected means the one-point product is
    subtracted using the same sample, so post-selection biases cancel in the
    subtraction the way they do in the analysis."""
    b = np.asarray(bits)
    sq = np.array([lat.site_qubit(v) for v in range(lat.ns)])
    sgn = np.array([(-1) ** v for v in range(lat.ns)])
    j = (sgn[None, :] - (1 - 2 * b[:, sq].astype(np.int8))) / 2
    one = j.mean(0)
    g = (j * j[:, [center]]).mean(0) - one * one[center]
    err = (j * j[:, [center]] - one[None, :] * one[center]).std(0) / np.sqrt(len(b))
    return g, err


def cloud_amplitude(bits, lat, center: int, exact, window: int = 2,
                    r_min: int = 1, r_max: int = 6):
    """Matched-filter amplitude of the measured screening cloud against an
    exact (MPS) reference: 1.0 means the device reproduces it fully.

    Fit region |x - center| in [r_min, r_max] on the ring, excluding the
    self term at x = center, which is trivially reproduced and would swamp
    the fit.  -> (amplitude, error, kept shots).
    """
    m = keep_mask(bits, lat, center, window)
    g, err = connected_j0_correlator(bits[m], lat, center)
    x = np.arange(lat.ns) - center
    x = np.minimum(np.abs(x), lat.ns - np.abs(x))
    reg = (x >= r_min) & (x <= r_max)
    den = float((np.asarray(exact)[reg] ** 2).sum())
    amp = float((np.asarray(exact)[reg] * g[reg]).sum()) / den
    amp_err = float(np.sqrt(((np.asarray(exact)[reg] * err[reg]) ** 2).sum())) / den
    return amp, amp_err, int(m.sum())
