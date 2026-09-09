"""Reproduce the boost asymmetry from an assembled W^{00} file.

    PYTHONPATH=. python scripts/boost_asymmetry.py data/hw_w00_coarse_t3.npz

Reads `q0`, `q1`, `W_hw`, `W_ref` (and, if present, the stored `A_hat`,
`A_err`, `A_sys` from the run that produced the file) and prints the
matched-filter amplitude A with its two errors.  The definition and the
error model are documented in htensor/asymmetry.py and docs/METHODS.md;
the extraction that also propagates the shot error lives in
scripts/hw_w00_coarse.py, which needs the time-domain slices.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from htensor import asymmetry as A  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else "data/hw_w00_coarse_t3.npz"
lo, hi = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else A.WEDGE
z = np.load(path, allow_pickle=True)

a = A.amplitude(z["W_hw"], z["W_ref"], z["q0"], wedge=(lo, hi))
print(f"{path}")
print(f"  wedge q0 in [{lo:g}, {hi:g}], {int(A.region_mask(z['q0'], (lo, hi)).sum())} of "
      f"{len(z['q0'])} q0 rows x {len(z['q1'])} q1 columns")
print(f"  A = {a:.4f}   [A = 1 -> the device reproduces the reference asymmetry in full]")

if "A_err" in z.files:
    shot, sysw = float(z["A_err"]), float(z["A_sys"])
    tot, sig = A.significance(a, shot, sysw)
    print(f"  stored errors: +- {shot:.4f} (shot) +- {sysw:.4f} (window)"
          f"  -> {tot:.4f} total, {sig:.1f} sigma")
    if "A_hat" in z.files:
        d = abs(a - float(z["A_hat"]))
        print(f"  stored A_hat = {float(z['A_hat']):.4f}  (reproduced to {d:.2e})")
if "times" in z.files:
    print(f"  from {len(z['times'])} slices, t <= {float(np.max(z['times'])):g}, "
          f"sigma_t = {float(z['sig_t']):g}" if "sig_t" in z.files else "")
