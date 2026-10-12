"""W^{mu nu}(q0, q1) for all four components assembled from HARDWARE (or
rehearsal) slices, against the MPS reference windowed identically.

Generalizes scripts/hw_w00_coarse.py.  Inputs are per-slice npz files with
the tier-3 key contract (C_cal, C_err, kappa_v, b_cal, ...) plus
`component` in {00, 10, 01, 11} and `t`, as written by htq_hw.analyze /
losch_t3_pool-style poolers, found by the template --slices, e.g.
"data/hw/slice_{comp}_t{t:.1f}.npz".  For every slice:

    G_hw(x) = [C_cal(x) - <J^mu(x,t)> A0] - G_vac^conn(x)

with A0 = <J^nu(0)> and <J^mu(x,t)> from b_cal when present (00, 01 with
J0 probes) else from the MPS one-point functions (J1 probes: the term is
<= 1e-3), and G_vac^conn from the MPS production run (hybrid subtraction).
Sites with kappa < KAP_MIN are masked (MPS-filled in the sum rule).  The
charge-conservation sum rule sum_x G = 0 applies to J0-probe components
(00, 01) only.  One-sided windowed transform (analysis.onesided_ft), the
MPS reference restricted to the same times, window and mask.

  PYTHONPATH=. .venv/bin/python scripts/hw_w_tensor.py --j0 data/w_meson_ns50_k1.26_v3.npz \
      --j1 data/w_meson_j1_ns50_k1.26.npz --slices "data/hw/slice_{comp}_t{t:.1f}.npz" --tmax 6
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI  # noqa: E402
from htensor import analysis, tensor  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--j0", required=True, help="MPS production npz, J0 insertion")
p.add_argument("--j1", required=True, help="MPS production npz, J1 insertion")
p.add_argument("--slices", required=True, help="template with {comp} and {t}")
p.add_argument("--times", type=float, nargs="+", default=None, help="slice times (default 0..tmax step 0.5)")
p.add_argument("--tmax", type=float, default=6.0)
p.add_argument("--kap-min", type=float, default=0.05)
p.add_argument("--sig-xf", type=float, default=2.0, help="sigma_x = nx / sig_xf (hardware default 2)")
p.add_argument("--out", default=None)
p.add_argument("--label", default="hardware")
args = p.parse_args()

g = tensor.load_mps_components(args.j0, args.j1)
lat, C, k0 = g.lat, g.center, g.k0
NS = lat.ns
slice_times = np.array(args.times if args.times else np.arange(0.0, args.tmax + 1e-9, 0.5))
d0 = np.load(args.j0)
d1 = np.load(args.j1)
A0 = {"0": complex(d0["insert_1pt_wp"]), "1": complex(d1["insert_1pt_wp"])}
one_mps = {comp: g.one_pt_wp[comp] for comp in tensor.COMPONENTS}
vac_conn = {}
for comp, (src, block, _) in tensor._LAYOUT.items():
    d = d0 if src == "j0" else d1
    sl = slice(block * NS, (block + 1) * NS)
    vac_conn[comp] = analysis.subtract(d["corr_vac"][:, sl], None, d["one_pt_vac"][:, sl],
                                       complex(d["insert_1pt_vac"]))

q0 = np.arange(-1.0, 6.001, 0.04)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)
q1 = 2 * np.pi * ks / lat.nx
sig_x = lat.nx / args.sig_xf
DT, DX = 0.5, 0.5
log_msg = []
W_hw, W_ref, W_err, band, corr = {}, {}, {}, {}, {}


def onesided(times, x, G, msk, sig_t):
    return analysis.onesided_ft(times, x, np.where(msk, G, 0.0), q0, q1, sig_t, sig_x, DT, DX)


def werr(times, x, err, msk, sig_t):
    wt = np.exp(-times ** 2 / (2 * sig_t ** 2)).copy()
    wt[0] *= 0.5
    wx = np.exp(-x ** 2 / (2 * sig_x ** 2))[None, :] * msk
    return 2 * DT * DX * np.sqrt(((wt[:, None] * wx) ** 2 * np.where(msk, err, 0.0) ** 2).sum())


for comp in tensor.COMPONENTS:
    nu = comp[1]
    x = g.x[comp]
    rows = {}
    for t in slice_times:
        path = args.slices.format(comp=comp, t=t)
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        assert str(d["component"]) == comp, (path, d["component"])
        ti = int(np.argmin(np.abs(g.times - t)))
        Ccal = np.asarray(d["C_cal"])[0]
        Cerr = np.asarray(d["C_err"])[0].real
        kap = np.asarray(d["kappa_v"])[0].real if "kappa_v" in d.files else np.ones(NS)
        one = np.asarray(d["b_cal"])[0] if ("b_cal" in d.files and comp in ("00", "01")) else one_mps[comp][ti]
        G_hw = (Ccal - one * A0[nu]) - vac_conn[comp][ti]
        finite = np.isfinite(G_hw) & np.isfinite(Cerr) & np.isfinite(kap) & (Cerr >= 0)
        ok = (kap > args.kap_min) & finite
        if not ok.any():
            log_msg.append(f"{comp} t={t}: omitted; no finite calibrated probes above kappa threshold")
            continue
        rows[t] = (G_hw, Cerr, ok)
        log_msg.append(f"{comp} t={t}: {int((~ok).sum())} masked ({int((~finite).sum())} nonfinite/invalid), "
                       f"median err {np.median(Cerr[ok]):.4f}")
    if not rows:
        log_msg.append(f"{comp}: no slices found")
        continue
    times = np.array(sorted(rows))
    if not np.isclose(times[0], 0, rtol=0, atol=1e-9):
        log_msg.append(f"{comp}: omitted; the one-sided transform requires a valid t=0 row")
        continue
    if len(times) < 2:
        log_msg.append(f"{comp}: omitted; no valid nonzero-time row")
        continue
    Gh = np.array([rows[t][0] for t in times]).real
    Ge = np.array([rows[t][1] for t in times])
    mask = np.array([rows[t][2] for t in times])
    # Calibration can leave NaNs on rejected probes. Zero them explicitly;
    # multiplication by a zero mask would still contaminate the transform.
    Gh = np.where(mask, Gh, 0.0)
    Ge = np.where(mask, Ge, 0.0)
    ti_mps = [int(np.argmin(np.abs(g.times - t))) for t in times]
    Gr = g.g[comp][ti_mps].real
    if comp in ("00", "01"):          # charge conservation: sum_x <J0(x,t) J^nu(0)>_conn = 0
        for i, t in enumerate(times):
            m = mask[i]
            target = -Gr[i][~m].sum()
            viol = Gh[i][m].sum() - target
            w2 = Ge[i][m] ** 2
            w2 = w2 / w2.sum() if w2.sum() > 0 else np.full(m.sum(), 1.0 / m.sum())
            Gh[i][m] -= viol * w2
            log_msg.append(f"{comp} t={t}: sum-rule violation {viol:+.4f} projected out")
    tmax = times[-1] if len(times) > 1 else 1.0
    sig_t = max(0.45, tmax / 2)
    W_hw[comp] = onesided(times, x, Gh, mask, sig_t)
    W_ref[comp] = onesided(times, x, Gr, mask, sig_t)
    W_err[comp] = werr(times, x, Ge, mask, sig_t)
    band[comp] = np.ptp(np.stack([onesided(times, x, Gh, mask, f * sig_t) for f in (0.75, 1.0, 1.5)]), axis=0)
    corr[comp] = float(np.corrcoef(W_hw[comp].ravel(), W_ref[comp].ravel())[0, 1])
    log_msg.append(f"{comp}: t_max={tmax}, sigma_t={sig_t:.2f}, W_err={W_err[comp]:.3f}, "
                   f"max|W| hw {np.abs(W_hw[comp]).max():.3f} ref {np.abs(W_ref[comp]).max():.3f}, "
                   f"corr(hw, ref) = {corr[comp]:.3f}")

if not W_hw:
    raise SystemExit("\n".join(log_msg) + "\nNo usable components; no tensor plotted.")

# ---- Ward check across components on the assembled hardware tensor
if all(c in W_hw for c in tensor.COMPONENTS):
    Wd = {c: (W_hw[c], None) for c in tensor.COMPONENTS}
    Wr = {c: (W_ref[c], None) for c in tensor.COMPONENTS}
    log_msg.append(f"q-space Ward residual (same window): hw {tensor.ward_residual_q(Wd, q0, q1)}  "
                   f"ref {tensor.ward_residual_q(Wr, q0, q1)}")
print("\n".join(log_msg))

out = args.out or f"data/hw_w_tensor_t{args.tmax:.0f}"
np.savez(out + ".npz", q0=q0, q1=q1, k0=k0, times=slice_times,
         **{f"W_hw_{c}": W_hw[c] for c in W_hw}, **{f"W_ref_{c}": W_ref[c] for c in W_ref},
         **{f"W_err_{c}": W_err[c] for c in W_err}, **{f"band_{c}": band[c] for c in band},
         **{f"corr_{c}": corr[c] for c in corr})

# ---- figure: hw vs ref per component
comps = [c for c in tensor.COMPONENTS if c in W_hw]
fig, axes = plt.subplots(2, len(comps), figsize=(2.6 * len(comps) + 1, 5.4), constrained_layout=True,
                         squeeze=False)
for j, comp in enumerate(comps):
    vmax = max(np.abs(W_ref[comp]).max(), np.abs(W_hw[comp]).max())
    for i, (Wp, ttl) in enumerate([(W_hw[comp], args.label), (W_ref[comp], "MPS, same window")]):
        ax = axes[i, j]
        pm = ax.pcolormesh(q1, q0, Wp, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest",
                           rasterized=True)
        ax.grid(False)
        if i == 1:
            ax.set_xlabel(r"$q^1$")
        if j == 0:
            ax.set_ylabel(r"$q^0$")
        ax.set_title(rf"$W^{{{comp}}}$ {ttl}" + (f" (corr {corr[comp]:.2f})" if i == 0 else ""), fontsize=8.5)
    fig.colorbar(pm, ax=axes[:, j], shrink=0.8, pad=0.01)
fig.suptitle(rf"four-component tensor, $k_0={k0:.2f}$, $t\leq{args.tmax:.0f}$", fontsize=10)
fig.savefig(out + ".pdf", dpi=200)
print(f"wrote {out}.{{npz,pdf}}")
