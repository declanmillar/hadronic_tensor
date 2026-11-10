"""What the measured W^{mu nu}(q0, q1) map can and cannot say about q0.

The device measures the REAL part of the connected correlator, so the
integrand fed to the transform is real.  For real G the one-sided transform
obeys, exactly,

    W(-q0, q1) = W(q0, -q1),        W_odd(-q0, q1) = -W_odd(q0, q1),

because conjugating e^{i(q0 t - q1 x)} and taking 2 Re is the same operation
as flipping both arguments.  So the q0 < 0 half of the map is the q0 > 0
half with q1 reflected: weight at negative q0 is the image of the
positive-q0 response, not leakage into it, and it is not an error of any
kind.  The boost asymmetry's wedge inherits this.  Its bins below q0 = 0
are the sign-flipped mirror of bins above, in the template exactly as in
the data, so they are redundant rather than wrong and A does not move when
they are dropped.

Two further facts the q0 axis has to be read with:

  period    a transform over a time grid of spacing dt is periodic in q0
            with period 2 pi / dt, 12.566 at the published dt = 0.5, so any
            response above the Nyquist energy pi / dt = 6.283 folds into
            the measured window.  Only a finer SAMPLING dt moves that edge;
            tapers and longer t_max do not.
  support   the ordered-product transform has weight only at q0 = E_n -
            E_psi >= -dE (scripts/packet_excitation.py).  dE is 2.86 (prod)
            and 3.11 (relA) against a published grid starting at q0 = -1,
            so no part of the published window is kinematically forbidden
            and the map cannot be null-tested inside it.

  PYTHONPATH=. python scripts/w_map_symmetry.py <reduction.npz> [...]

with the reduction npz written by scripts/hw_w_tensor.py.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from htensor import Z2Lattice, analysis, asymmetry as asym, tensor  # noqa: E402

WEDGES = (-1.0, -0.5, 0.0, 0.25)
EXCITATION = "data/packet_excitation_ns50.npz"


def mirror(grid):
    """Index map q -> -q, and the entries for which -q is on the grid."""
    idx = np.array([int(np.argmin(np.abs(grid + q))) for q in grid])
    on = np.abs(grid[idx] + grid) < 1e-9
    return idx, on


def excitation(d):
    """-> (dE, provenance) for the card this reduction came from."""
    if not os.path.exists(EXCITATION):
        return None, f"{EXCITATION} not built"
    z = np.load(EXCITATION, allow_pickle=True)
    tag = "relA" if "relA" in str(d["source_j0"]) else "prod"
    sel = np.where(z["tag"] == tag)[0]
    if not len(sel):
        return None, f"no {tag} row in {EXCITATION}"
    # the widest-packet row is the most conservative edge
    i = sel[int(np.argmax(z["de"][sel]))]
    return float(z["de"][i]), f"{tag} {z['label'][i]}"


def report(path):
    d = np.load(path, allow_pickle=True)
    q0, q1 = d["q0"], d["q1"]
    dt, dx, ns, center = (float(d["integration_dt"]), float(d["integration_dx"]),
                          int(d["ns"]), int(d["center"]))
    lat = Z2Lattice(ns, pbc=True)
    comps = [c for c in tensor.COMPONENTS if f"W_hw_{c}" in d.files]
    i0, on0 = mirror(q0)
    j1, on1 = mirror(q1)
    de, prov = excitation(d)

    print(f"\n=== {path}")
    print(f"  q0 in [{q0.min():+.2f}, {q0.max():+.2f}], dt = {dt:g}: period "
          f"2pi/dt = {2 * np.pi / dt:.3f}, Nyquist |q0| <= {np.pi / dt:.3f}")
    if de is None:
        print(f"  kinematic edge: unavailable ({prov})")
    else:
        inside = "inside" if -de >= q0.min() else "below"
        print(f"  kinematic edge q0 >= -dE = {-de:+.3f} ({prov}), {inside} the grid's "
              f"lower edge {q0.min():+.2f}")
    if not on1.all():
        print("  q1 grid is not mirror symmetric; the identity cannot be checked")
        return

    for c in comps:
        G = d[f"G_hw_{c}"]
        im = float(np.abs(np.imag(G)).max()) if np.iscomplexobj(G) else 0.0
        W, Wo = d[f"W_hw_{c}"], d[f"W_hw_odd_{c}"]
        peak = float(np.abs(W).max())
        r_full = float(np.abs(W[np.ix_(i0[on0], np.arange(len(q1)))]
                              - W[np.ix_(np.where(on0)[0], j1)]).max())
        r_odd = float(np.abs(Wo[np.ix_(i0[on0], np.arange(len(q1)))] + Wo[on0]).max())
        print(f"  {c}: max|Im G| = {im:.1e}, peak |W| = {peak:.3f}")
        print(f"      max|W(-q0,q1) - W(q0,-q1)|   = {r_full:.2e}  "
              f"({r_full / peak:.1e} of peak)")
        print(f"      max|Wodd(-q0,q1) + Wodd(q0,q1)| = {r_odd:.2e}  "
              f"({r_odd / max(np.abs(Wo).max(), 1e-300):.1e} of peak)")

        # periodicity, on an extended grid (the saved one is narrower than a period)
        x = tensor.x_grid(lat, c, center)
        times, mask = d[f"times_{c}"], d[f"mask_{c}"]
        sig_t, sig_x = float(d[f"sig_t_{c}"]), float(d["sig_x"])
        per = 2 * np.pi / dt
        qe = np.arange(q0.min(), q0.max() + 1e-9, float(np.diff(q0)[0]))
        ft = lambda g, qq, f=1.0: analysis.onesided_ft(
            times, x, np.where(mask, g, 0.0), qq, q1, f * sig_t, sig_x, dt, dx)
        res = float(np.abs(ft(d[f"G_hw_{c}"], qe)
                           - ft(d[f"G_hw_{c}"], qe + per)).max())
        print(f"      max|W(q0) - W(q0 + 2pi/dt)| = {res:.2e}  "
              f"({res / peak:.1e} of peak), over q0 in "
              f"[{qe.min():+.1f}, {qe.max():+.1f}]")

        # the wedge's sub-zero bins are redundant: A should not move
        if f"W_ref_{c}" not in d.files:
            continue
        Wr, G_err = d[f"W_ref_{c}"], d[f"G_err_{c}"]
        # hw_w_tensor --unmeasured-error ref: the masked sites carry the
        # matched reference |G| as an error; it enters A the same way as the
        # shot errors, through the filter's kernel, with no mask on its own
        # points (it lives exactly where the measured mask is False)
        G_unm = d[f"G_unmeasured_{c}"] if f"G_unmeasured_{c}" in d.files else None
        have_unm = G_unm is not None and np.any(G_unm > 0)
        ft_all = lambda g, qq: analysis.onesided_ft(times, x, g, qq, q1, sig_t, sig_x, dt, dx)
        print(f"      {'wedge q0':<16}{'bins':>5}{'A':>8}{'shot':>7}{'win':>7}"
              + (f"{'unmeas':>8}" if have_unm else "") + f"{'sigma':>7}")
        for lo_edge in WEDGES:
            wed = (lo_edge, asym.WEDGE[1])
            reg = asym.region_mask(q0, wed)
            if not reg.any():
                continue
            T = asym.odd(Wr)[reg]
            A = asym.amplitude(d[f"W_hw_{c}"], Wr, q0, wed)
            shot = asym.shot_error(G_err, T, lambda g: ft(g, q0)[reg], mask=mask)
            win = asym.window_systematic(None, T, lambda f: ft(d[f"G_hw_{c}"], q0, f)[reg])
            if have_unm:
                unm = asym.shot_error(G_unm, T, lambda g: ft_all(g, q0)[reg], mask=G_unm > 0)
                tot, sig = asym.significance(A, float(np.hypot(shot, unm)), win)
                print(f"      [{lo_edge:+.2f}, {wed[1]:.2f}]{'':4}{int(reg.sum()):5d}"
                      f"{A:8.3f}{shot:7.3f}{win:7.3f}{unm:8.3f}{sig:7.2f}")
            else:
                tot, sig = asym.significance(A, shot, win)
                print(f"      [{lo_edge:+.2f}, {wed[1]:.2f}]{'':4}{int(reg.sum()):5d}"
                      f"{A:8.3f}{shot:7.3f}{win:7.3f}{sig:7.2f}")


p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("npz", nargs="+", help="reduction npz from scripts/hw_w_tensor.py")
args = p.parse_args()
for path in args.npz:
    report(path)
