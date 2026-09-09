"""The canonical gauge post-selection reproduces the inline versions in the
figure scripts, bit for bit, on the released ibm_kingston sample."""

import os

import numpy as np
import pytest

from htensor import Z2Lattice, postselect as PS

NS, VC = 50, 24
BITS = "data/hw/sq_bits_kingston.npz"
pytestmark = pytest.mark.skipif(not os.path.exists(BITS), reason="kingston bits not present")


@pytest.fixture(scope="module")
def sample():
    return np.load(BITS)["bits"].astype(np.int8), Z2Lattice(NS, pbc=True)


def _inline_gcol(b, lat, n):
    """Verbatim from scripts/hw_w00_coarse.py:60 and hw_cloud_figure.py:36."""
    return ((-1) ** n * (1 - 2 * b[:, lat.site_qubit(n)])
            * (1 - 2 * b[:, lat.link_qubit(n - 1)])
            * (1 - 2 * b[:, lat.link_qubit(n)]))


def test_matches_the_inline_selection(sample):
    bits, lat = sample
    for win in (0, 1, 2, 3):
        sel = np.ones(len(bits), bool)
        for n in range(VC - win, VC + win + 1):
            if win > 0:
                sel &= _inline_gcol(bits, lat, n) > 0
        assert np.array_equal(sel, PS.keep_mask(bits, lat, VC, win)), win


def test_ranking_is_by_ring_distance_from_the_observable(sample):
    _, lat = sample
    assert PS.check_ranking(lat, VC, 0) == [VC]
    assert PS.check_ranking(lat, VC, 2) == [VC, VC - 1, VC + 1, VC - 2, VC + 2]
    assert PS.check_ranking(lat, 0, 1) == [0, NS - 1, 1]          # wraps the seam
    for w in (1, 3):
        assert len(PS.check_ranking(lat, VC, w)) == 2 * w + 1


def test_acceptance_falls_and_the_cloud_amplitude_rises(sample):
    """The trade the ranking exists to make, on the released sample."""
    bits, lat = sample
    tru = np.load("data/w_meson_ns50_k1.26_v3.npz")
    one = tru["one_pt_wp"][0, :NS].real
    exact = tru["corr_wp"][0, :NS].real - one * one[VC]
    acc = PS.acceptance(bits, lat, VC, windows=(0, 1, 2, 3))
    assert acc[0][1] == 1.0
    assert acc[0][1] > acc[1][1] > acc[2][1] > acc[3][1]
    amps = [PS.cloud_amplitude(bits, lat, VC, exact, window=w)[0] for w in (0, 1, 2, 3)]
    assert amps[0] == pytest.approx(0.253, abs=0.01)     # raw device, ~25% of exact
    assert amps[2] == pytest.approx(0.597, abs=0.01)     # window 2, the published setting
    assert amps[0] < amps[1] < amps[2] < amps[3]


def test_connected_correlator_matches_the_inline_estimator(sample):
    bits, lat = sample
    sq = np.array([lat.site_qubit(v) for v in range(NS)])
    sgn = np.array([(-1) ** v for v in range(NS)])
    z = 1 - 2 * bits[:, sq]
    j = (sgn[None, :] - z) / 2
    o = j.mean(0)
    g_inline = (j * j[:, [VC]]).mean(0) - o * o[VC]
    g, _ = PS.connected_j0_correlator(bits, lat, VC)
    assert np.allclose(g, g_inline, atol=1e-12)
