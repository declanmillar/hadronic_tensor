"""Validate a raw per-shot bits file before it is added to the repository.

    PYTHONPATH=. python scripts/check_raw_pubs.py data/hw/raw/htq_bits_<id>.npz [--ns 50] [--center 24]

Reports shape, shots, the implied Ns, per-pub means, and the Gauss-law
acceptance.  The acceptance is the quickest test that the column convention
is right: logical wire i must be in column i (matter site v at wire 2v, link
(v, v+1) at 2v+1, ancilla last).  A correctly ordered Z-setting pub accepts
the near-insertion checks well above chance; a scrambled one sits at chance.
See data/hw/README.md for the format and docs/METHODS.md for the analysis.
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from htensor import Z2Lattice, postselect as PS  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--ns", type=int, default=None, help="default: inferred from the column count")
p.add_argument("--center", type=int, default=None, help="default: ns//2 - 1")
p.add_argument("--max-pubs", type=int, default=12)
a = p.parse_args()

z = np.load(a.path, allow_pickle=True)
names = [str(x) for x in z["pub_names"]] if "pub_names" in z.files else \
    [k for k in z.files if np.asarray(z[k]).ndim == 2]
print(f"{a.path}")
for k in ("job_id", "backend"):
    if k in z.files:
        print(f"  {k}: {z[k]}")
print(f"  {len(names)} pub(s)")

problems = []
arrs = {n: np.asarray(z[n]) for n in names if n in z.files}
if not arrs:
    raise SystemExit("no per-pub arrays found; expected one uint8 (shots, n_logical) array per pub name")

widths = {v.shape[1] for v in arrs.values()}
if len(widths) != 1:
    problems.append(f"pubs disagree on the qubit count: {sorted(widths)}")
w = sorted(widths)[0]
ns = a.ns or (w // 2)                       # 2Ns system qubits, ancilla may or may not be included
center = a.center if a.center is not None else ns // 2 - 1
print(f"  {w} columns -> Ns = {ns} ({2 * ns} system qubits"
      f"{', + ancilla' if w > 2 * ns else ''}), centre {center}")

lat = Z2Lattice(ns, pbc=True)
print(f"\n  {'pub':<34} {'shots':>7} {'mean bit':>9} {'Gauss acc w=1':>14} {'w=2':>7}")
for n in names[: a.max_pubs]:
    b = arrs[n]
    if b.dtype != np.uint8:
        problems.append(f"{n}: dtype {b.dtype}, expected uint8")
    if b.size and b.max() > 1:
        problems.append(f"{n}: values above 1 -- these are not bits")
    bi = b[:, : 2 * ns].astype(np.int8)
    try:
        a1 = float(PS.keep_mask(bi, lat, center, 1).mean())
        a2 = float(PS.keep_mask(bi, lat, center, 2).mean())
    except Exception as e:                  # a width that cannot host the lattice
        problems.append(f"{n}: Gauss check failed ({type(e).__name__}: {e})")
        a1 = a2 = float("nan")
    print(f"  {n[:34]:<34} {len(b):>7} {b.mean():>9.4f} {a1:>14.3f} {a2:>7.3f}")
if len(names) > a.max_pubs:
    print(f"  ... {len(names) - a.max_pubs} more")

print("\n  Reference (released kingston Z-setting sample, correct order): 0.388 (w=1), 0.214 (w=2);"
      "\n  the same array with permuted columns gives 0.072 and 0.016. A much lower acceptance means"
      "\n  the column order is probably wrong, or the pub was not read in Z (matter Z, links X).")
if problems:
    print("\nPROBLEMS:")
    for x in problems:
        print(f"  - {x}")
    raise SystemExit(1)
print("\nformat OK")
