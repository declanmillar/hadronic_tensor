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
Where the measured b_cal is the thing subtracted, its error is re-weighted
from id_a to (id_a - A0) (htensor.analysis.g_error): G sees the probe
one-point only through that difference, so a saved C_err used unchanged
over-weights the wing-anchor and 1/beta terms by id_a/(id_a - A0), 22x at
the production point.  --ideal supplies the wing target for slices written
before b_err was saved.  Sites with kappa < KAP_MIN or |beta| < BETA_MIN are
masked (MPS-filled in the sum rule); the beta floor rejects probes whose
one-point denominator has collapsed, which kappa alone does not catch.  The
charge-conservation sum rule sum_x G = 0 applies to J0-probe components
(00, 01) only.  One-sided windowed transform (analysis.onesided_ft), the
MPS reference restricted to the same times, window and mask.

--parity additionally plots even/odd parts under q1 -> -q1, with matched
hardware/MPS heatmaps and momentum cuts. Bands combine a bound propagated
from saved C_err with the envelope of separately projected window scans.
C_err includes sampling and calibration terms (mirror/wing where present).
Propagation uses a diagonal, independent-site approximation; full spatial
and shared-mirror covariance is unavailable and is not included.
The reference uses its saved fine evolution step; no Trotter correction is
applied. Vacuum subtraction remains hybrid (MPS), not the measured vacuum.

--run-dir selects a hardware campaign; outputs go in analysis/<UTC timestamp>/.
An explicit --out selects a prefix; existing outputs are never overwritten.

  PYTHONPATH=. .venv/bin/python scripts/hw_w_tensor.py --j0 data/w_meson_ns50_k1.26_v3.npz \
      --j1 data/w_meson_j1_ns50_k1.26.npz --slices "data/hw/slice_{comp}_t{t:.1f}.npz" --tmax 6 --run-dir data/hw/runs/phoenix/2026-10-06
"""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI, OI_DIV, dev  # noqa: E402
from htensor import analysis, asymmetry, tensor  # noqa: E402
from htq_hw.campaign import QuarterTime  # noqa: E402
from run_outputs import (check_output_prefix, analysis_directory,  # noqa: E402
                          add_analysis_arguments, in_place)

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--j0", required=True, help="MPS production npz, J0 insertion")
p.add_argument("--j1", required=True, help="MPS production npz, J1 insertion")
p.add_argument("--slices", required=True, help="template with {comp} and {t}")
p.add_argument("--times", type=float, nargs="+", default=None, help="slice times (default 0..tmax step 0.5)")
p.add_argument("--tmax", type=float, default=6.0)
p.add_argument("--kap-min", type=float, default=0.05)
p.add_argument("--im", default=None, action=argparse.BooleanOptionalAction,
               help="keep the imaginary part of the connected correlator, when the slices "
                    "carry it (C_err_im), on BOTH the hardware and the MPS reference.  "
                    "The real part alone forces W(-q0,q1) = W(q0,-q1), so the q0 < 0 half "
                    "of the map is a reflection; the imaginary part is what gives the "
                    "transform its one-sided support.  Default: on when available")
p.add_argument("--drop-sites", type=int, nargs="+", default=(), metavar="SITE",
               help="logical sites masked in every slice (e.g. the two dead matter qubits, "
                    "sites 16 and 48 on the 2026-10-06 phoenix embedding)")
p.add_argument("--beta-min", type=float, default=0.01,
               help="mask probes with |beta| below this (default 0.01)")
p.add_argument("--sum-rule-min-coverage", type=float, default=0.0, metavar="FRAC",
               help="apply the charge-conservation projection on 00 and 01 only in slices that "
                    "retain at least this fraction of sites (default 0: always, the legacy "
                    "behaviour).  The projection's target is the reference's sum over the "
                    "masked sites, so under heavy masking it writes the reference into the "
                    "hardware map; 0.9 confines it to slices that are nearly all hardware")
p.add_argument("--unmeasured-error", default="none", choices=("none", "ref"),
               help="'ref': count every masked site as unmeasured, with an error equal to the "
                    "matched reference |G| there, added in quadrature to W_err (and so to the "
                    "plotted bands); 'none' (default, legacy): a masked site carries no error, "
                    "as if it were known to vanish")
p.add_argument("--window", default="gauss", choices=("gauss", "kaiser", "dpss", "rect"),
               help="time taper (default gauss, the published choice); --sig-t-p is its width knob")
p.add_argument("--sig-t-p", type=float, default=None,
               help="width knob for --window: sigma_t for gauss (default t_max/2), "
                    "beta for kaiser, NW for dpss, ignored for rect")
p.add_argument("--ideal", default=None,
               help="ideal-grid template with {family}, for slices with no saved b_err")
p.add_argument("--wing-surrogate", default=None,
               help="wing surrogate npz, for times beyond the ideal grid")
p.add_argument("--sig-xf", type=float, default=2.0, help="sigma_x = nx / sig_xf (hardware default 2)")
add_analysis_arguments(p, output_help="explicit output prefix")
p.add_argument("--label", default="hardware")
p.add_argument("--parity", action="store_true", help="also plot even/odd parts under q1 reflection and their cuts")
p.add_argument("--q0-range", nargs=2, type=float, default=(-1.0, 5.0), metavar=("MIN", "MAX"),
               help="display range only (default: -1 5)")
p.add_argument("--cut-q1", nargs="+", type=float, default=(0.8, 1.26, 2.3), metavar="Q1",
               help="momentum cuts, snapped to the nearest grid value")
args = p.parse_args()
if not (np.isfinite(args.tmax) and args.tmax >= 0 and np.isfinite(args.sig_xf) and args.sig_xf > 0):
    p.error("--tmax must be nonnegative and --sig-xf positive")
if not np.isfinite(args.kap_min) or not (np.isfinite(args.beta_min) and args.beta_min >= 0):
    p.error("--kap-min must be finite and --beta-min finite and nonnegative")
if not (np.isfinite(args.sum_rule_min_coverage) and 0 <= args.sum_rule_min_coverage <= 1):
    p.error("--sum-rule-min-coverage must be a fraction in [0, 1]")
if not (np.isfinite(args.q0_range).all() and args.q0_range[0] < args.q0_range[1]):
    p.error("--q0-range requires finite MIN < MAX")
out = (check_output_prefix(args.out if args.out else
                           analysis_directory(args.run_dir) / f"hw_w_tensor_t{args.tmax:.0f}")
       if (args.out or args.run_dir) else in_place(f"data/hw_w_tensor_t{args.tmax:.0f}"))

g = tensor.load_mps_components(args.j0, args.j1)
lat, C, k0 = g.lat, g.center, g.k0
NS = lat.ns
slice_times = np.array(args.times if args.times else np.arange(0.0, args.tmax + 1e-9, 0.5))
if not (np.isfinite(slice_times).all() and (slice_times >= 0).all()
        and (np.diff(slice_times) > 0).all()):
    p.error("--times must be finite, nonnegative and strictly increasing")
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
if args.q0_range[0] < q0[0] - 1e-12 or args.q0_range[1] > q0[-1] + 1e-12:
    p.error(f"--q0-range must lie within [{q0[0]:g}, {q0[-1]:g}]")
if not (np.isfinite(args.cut_q1).all() and min(args.cut_q1) >= q1[0] and max(args.cut_q1) <= q1[-1]):
    p.error(f"--cut-q1 must lie within [{q1[0]:g}, {q1[-1]:g}]")
cut_indices = [int(np.argmin(np.abs(q1 - q))) for q in args.cut_q1]
if len(set(cut_indices)) != len(cut_indices):
    p.error("--cut-q1 must select distinct momentum grid values")
sig_x = lat.nx / args.sig_xf
DT, DX = 0.5, 0.5
window_factors = (0.75, 1.0, 1.5)
log_msg = ["Re-only hardware and MPS; hybrid MPS vacuum subtraction; no Trotter correction.",
           f"MPS dt_target={g.meta.get('dt_target', 'unrecorded')}; hardware dt read from slices."]
W_hw, W_ref, W_err, band, corr = {}, {}, {}, {}, {}
W_unm = {}            # error assigned to the masked (unmeasured) sites, 0 unless --unmeasured-error
reweighted = []
window_hw, window_ref, details = {}, {}, {}
source_paths, source_dt = {}, {}


_IDEALS, _SURR = {}, None
if args.ideal:
    import htq_hw.analyze as hqa                                    # noqa: E402
    _SURR = np.load(args.wing_surrogate) if args.wing_surrogate else None
    for fam in ("j0", "j1p1"):
        try:
            _IDEALS[fam] = hqa.IdealGrid(args.ideal.format(family=fam), expect_ns=NS)
        except (FileNotFoundError, OSError):
            pass
    if not _IDEALS:
        p.error(f"--ideal {args.ideal!r} matched no grid for j0 or j1p1")


def bcal_error(d, comp, t):
    """-> (b_err, provenance).  Saved when the reduction wrote it, else the
    wing scatter recomputed from the saved b_cal (a lower bound on b_err, so
    a conservative G error), else None and C_err is used unchanged."""
    if "b_err" in d.files:
        return np.asarray(d["b_err"])[0].real, "saved"
    fam = "j0" if comp == "00" else "j1p1"
    if fam not in _IDEALS:
        return None, "unavailable"
    sN, src = hqa.b_err_floor(np.asarray(d["b_cal"])[0], t, _IDEALS[fam], C, _SURR)
    return (None, f"no wing target ({src})") if sN is None else (sN, f"wing scatter from {src}")


def onesided(times, x, G, msk, sig_t):
    return analysis.onesided_ft(times, x, np.where(msk, G, 0.0), q0, q1, sig_t, sig_x, DT, DX,
                                window=args.window)


def werr(times, x, err, msk, sig_t):
    wt = analysis.time_window(times, sig_t, args.window).copy()
    wt[0] *= 0.5
    wx = np.exp(-x ** 2 / (2 * sig_x ** 2))[None, :] * msk
    return 2 * DT * DX * np.sqrt(((wt[:, None] * wx) ** 2 * np.where(msk, err, 0.0) ** 2).sum())


for comp in tensor.COMPONENTS:
    nu = comp[1]
    x = g.x[comp]
    rows = {}
    paths, steps = [], []
    for t in slice_times:
        path = args.slices.format(comp=comp, t=QuarterTime(t))
        if not os.path.exists(path):
            continue
        d = np.load(path, allow_pickle=True)
        assert str(d["component"]) == comp, (path, d["component"])
        saved_t = float(d["t"]) if "t" in d.files else float(d["times"][0])
        if not np.isclose(saved_t, t, rtol=0, atol=1e-9):
            raise ValueError(f"{path}: stored time {saved_t} does not match requested time {t}")
        ti = int(np.argmin(np.abs(g.times - t)))
        if not np.isclose(g.times[ti], t, rtol=0, atol=1e-9):
            raise ValueError(f"{path}: no MPS reference at t={t}; nearest is {g.times[ti]}")
        Ccal = np.asarray(d["C_cal"])[0]
        Cerr = np.asarray(d["C_err"])[0].real
        Cerr_im = np.asarray(d["C_err_im"])[0].real if "C_err_im" in d.files else None
        kap = np.asarray(d["kappa_v"])[0].real if "kappa_v" in d.files else np.ones(NS)
        bet = np.asarray(d["beta_v"])[0].real if "beta_v" in d.files else np.ones(NS)
        used_bcal = "b_cal" in d.files and comp in ("00", "01")
        one = np.asarray(d["b_cal"])[0] if used_bcal else one_mps[comp][ti]
        G_hw = (Ccal - one * A0[nu]) - vac_conn[comp][ti]
        if used_bcal:
            b_err, how = bcal_error(d, comp, float(t))
            if b_err is not None:
                before = float(np.median(Cerr))
                Cerr = analysis.g_error(Cerr, b_err, float(d["id_a"]), A0[nu])
                reweighted.append(f"{comp} t={t:g}: b_cal error {how}, median C_err "
                                  f"{before:.4f} -> {np.median(Cerr):.4f}")
            else:
                reweighted.append(f"{comp} t={t:g}: b_cal error NOT re-weighted ({how})")
        if any(np.shape(a) != (NS,) for a in (G_hw, Cerr, kap, bet)):
            raise ValueError(f"{path}: expected {NS} probe values")
        finite = np.isfinite(G_hw) & np.isfinite(Cerr) & np.isfinite(kap) & (Cerr >= 0)
        ok = (kap > args.kap_min) & (np.abs(bet) > args.beta_min) & finite
        if args.drop_sites:
            ok[list(args.drop_sites)] = False
        if not np.isfinite(g.g[comp][ti]).all():
            raise ValueError(f"{path}: nonfinite MPS reference")
        if not ok.any():
            log_msg.append(f"{comp} t={t}: omitted; no finite calibrated probes above kappa threshold")
            continue
        rows[t] = (G_hw, Cerr, ok, kap, Cerr_im)
        paths.append(path)
        steps.append(float(d["dt"]) if "dt" in d.files else DT)
        log_msg.append(f"{comp} t={t}: {int((~ok).sum())} masked ({int((~finite).sum())} nonfinite/invalid, "
                       f"{int((np.abs(bet) <= args.beta_min).sum())} below the beta floor), "
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
    if not np.allclose(steps, steps[0], rtol=0, atol=1e-9):
        raise ValueError(f"{comp}: cannot pool different evolution steps in one tensor")
    have_im = [t for t in times if rows[t][4] is not None]
    use_im = bool(have_im) and args.im is not False
    if args.im and not have_im:
        log_msg.append(f"{comp}: --im requested but no slice carries C_err_im; real part only")
    if use_im and len(have_im) < len(times):
        raise ValueError(f"{comp}: slices at t={[t for t in times if t not in have_im]} carry no "
                         f"imaginary part while others do; reduce the tier uniformly or pass --no-im")
    Gh = np.array([rows[t][0] for t in times])
    # a contiguous copy rather than a .real view: the view's stride reaches
    # BLAS inside onesided_ft, which is a summation-order hazard even though
    # it happens to make no difference on this grid
    Gh = Gh if use_im else np.ascontiguousarray(Gh.real)
    Ge = np.array([rows[t][1] for t in times])
    if use_im:
        # W = 2[Re(K) Gr - Im(K) Gi], and |Re K|, |Im K| <= dt dx w, so the
        # published bound's convention carries over with the two in quadrature
        Ge = np.hypot(Ge, np.array([rows[t][4] for t in times]))
        log_msg.append(f"{comp}: imaginary part kept on {len(have_im)}/{len(times)} slices; "
                       f"W(-q0,q1) = W(q0,-q1) no longer holds by construction")
    mask = np.array([rows[t][2] for t in times])
    # Calibration can leave NaNs on rejected probes. Zero them explicitly;
    # multiplication by a zero mask would still contaminate the transform.
    Gh = np.where(mask, Gh, 0.0)
    Ge = np.where(mask, Ge, 0.0)
    ti_mps = [int(np.argmin(np.abs(g.times - t))) for t in times]
    Gr = (g.g[comp][ti_mps] if use_im
          else np.ascontiguousarray(g.g[comp][ti_mps].real))
    if comp in ("00", "01"):          # charge conservation: sum_x <J0(x,t) J^nu(0)>_conn = 0
        for i, t in enumerate(times):
            m = mask[i]
            if m.mean() < args.sum_rule_min_coverage:
                log_msg.append(f"{comp} t={t}: sum-rule projection skipped, {int(m.sum())}/{m.size} sites "
                               f"retained is below the coverage floor {args.sum_rule_min_coverage:g}")
                continue
            target = -Gr[i][~m].sum()
            viol = Gh[i][m].sum() - target
            w2 = Ge[i][m] ** 2
            w2 = w2 / w2.sum() if w2.sum() > 0 else np.full(m.sum(), 1.0 / m.sum())
            Gh[i][m] -= viol * w2
            log_msg.append(f"{comp} t={t}: sum-rule violation "
                            + (f"{viol.real:+.4f}{viol.imag:+.4f}j" if use_im else f"{viol:+.4f}")
                            + " projected out")
    tmax = times[-1] if len(times) > 1 else 1.0
    sig_t = (args.sig_t_p if args.sig_t_p is not None else max(0.45, tmax / 2))
    W_hw[comp] = onesided(times, x, Gh, mask, sig_t)
    W_ref[comp] = onesided(times, x, Gr, mask, sig_t)
    W_err[comp] = werr(times, x, Ge, mask, sig_t)
    # A masked site is not known to vanish; with --unmeasured-error ref it is
    # unknown to within the matched reference's magnitude there, and that
    # bound propagates through the same window weights as the measured errors.
    W_unm[comp] = (werr(times, x, np.abs(Gr), ~mask, sig_t)
                   if args.unmeasured_error == "ref" else 0.0)
    W_err[comp] = float(np.hypot(W_err[comp], W_unm[comp]))
    window_hw[comp] = np.stack([onesided(times, x, Gh, mask, f * sig_t) for f in window_factors])
    window_ref[comp] = np.stack([onesided(times, x, Gr, mask, f * sig_t) for f in window_factors])
    band[comp] = np.ptp(window_hw[comp], axis=0)
    corr[comp] = (float(np.corrcoef(W_hw[comp].ravel(), W_ref[comp].ravel())[0, 1])
                  if np.std(W_hw[comp]) > 0 and np.std(W_ref[comp]) > 0 else float("nan"))
    for value in (W_hw[comp], W_ref[comp], W_err[comp], window_hw[comp], window_ref[comp]):
        if not np.isfinite(value).all():
            raise ValueError(f"{comp}: nonfinite assembled output")
    details.update({f"times_{comp}": times, f"mask_{comp}": mask,
                    f"coverage_{comp}": mask.mean(axis=1), f"W_unmeasured_{comp}": W_unm[comp],
                    f"G_unmeasured_{comp}": (np.where(mask, 0.0, np.abs(Gr))
                                             if args.unmeasured_error == "ref" else np.zeros_like(Ge)),
                    f"kappa_{comp}": np.array([rows[t][3] for t in times]),
                    f"G_hw_{comp}": Gh, f"G_ref_{comp}": np.where(mask, Gr, 0.0),
                    f"G_err_{comp}": Ge, f"x_{comp}": x, f"sig_t_{comp}": sig_t,
                    f"projection_{comp}": ("complex connected correlator" if use_im
                                           else "Re connected correlator"),
                    f"hardware_dt_{comp}": steps[0], f"source_slices_{comp}": paths})
    source_paths[comp], source_dt[comp] = paths, steps[0]
    log_msg.append(f"{comp}: t_max={tmax}, sigma_t={sig_t:.2f}, W_err={W_err[comp]:.3f}"
                   + (f" (of which unmeasured sites {W_unm[comp]:.3f})" if W_unm[comp] else "")
                   + f", site coverage {mask.mean():.2f} (min over slices {mask.mean(axis=1).min():.2f}), "
                   f"max|W| hw {np.abs(W_hw[comp]).max():.3f} ref {np.abs(W_ref[comp]).max():.3f}, "
                   f"corr(hw, ref) = {corr[comp]:.3f}")

log_msg = ([f"kappa floor {args.kap_min:g}, |beta| floor {args.beta_min:g}, "
            f"time window {args.window}"
            + (f" (p = {args.sig_t_p:g})" if args.sig_t_p is not None else "")
            + (f", sites {list(args.drop_sites)} dropped" if args.drop_sites else "")
            + (f", sum rule only at coverage >= {args.sum_rule_min_coverage:g}"
               if args.sum_rule_min_coverage > 0 else "")
            + (", masked sites carry the reference |G| as error" if args.unmeasured_error == "ref" else "")]
           + reweighted + log_msg)
if not W_hw:
    raise SystemExit("\n".join(log_msg) + "\nNo usable components; no tensor plotted.")

# ---- Ward check across components on the assembled hardware tensor.  The
# exact form (tensor.ward_pair) needs every site, so masked points are filled
# from the reference for this check only; the quotable number is the mixed
# fit, reference J0 side against hardware J1 side (scripts/ward_scale.py).
for nu, (c0, c1) in {"0": ("00", "10"), "1": ("01", "11")}.items():
    if c0 not in W_hw or c1 not in W_hw:
        continue
    t0, t1 = details[f"times_{c0}"], details[f"times_{c1}"]
    if len(t0) != len(t1) or not np.allclose(t0, t1):
        log_msg.append(f"Ward nu={nu}: {c0} and {c1} span different times; skipped")
        continue
    gr = {c: g.g[c][[int(np.argmin(np.abs(g.times - t))) for t in t0]] for c in (c0, c1)}
    cplx = str(details[f"projection_{c0}"]).startswith("complex")
    gr = {c: (v if cplx else np.ascontiguousarray(v.real)) for c, v in gr.items()}
    gh = {c: np.where(details[f"mask_{c}"], details[f"G_hw_{c}"], gr[c]) for c in (c0, c1)}
    wedge = asymmetry.region_mask(q0)
    fits = {}
    for lab, A, B in (("ref/ref", gr[c0], gr[c1]), ("ref/hw", gr[c0], gh[c1]),
                      ("hw/hw", gh[c0], gh[c1])):
        a, b = tensor.ward_pair(t0, details[f"x_{c0}"], details[f"x_{c1}"], A, B, q0, q1,
                                details[f"sig_t_{c0}"], sig_x, DT, DX, "integral")
        r = tensor.ward_scale(a, b, wedge)
        fits[lab] = f"[{r['s_lo']:.3f}, {r['s_hi']:.3f}] resid {r['resid_1']:.3f}"
    log_msg.append(f"exact Ward (integral form, wedge, masked points filled from the reference) "
                   f"nu={nu} [s_lo, s_hi]: " + "; ".join(f"{k} {v}" for k, v in fits.items()))
command = shlex.join([sys.executable, *sys.argv])
projections = {}
if args.parity:
    if not np.allclose(q1, -q1[::-1], rtol=0, atol=1e-12):
        raise ValueError("Even/odd projection requires a symmetric q1 grid")
    for comp in W_hw:
        for kind, windows in (("hw", window_hw[comp]), ("ref", window_ref[comp])):
            for part in ("even", "odd"):
                samples = asymmetry.odd(windows) if part == "odd" else 0.5 * (windows + windows[..., ::-1])
                projections[f"W_{kind}_{part}_{comp}"] = samples[1]
                projections[f"window_{kind}_{part}_{comp}"] = samples
                projections[f"window_min_{kind}_{part}_{comp}"] = samples.min(axis=0)
                projections[f"window_max_{kind}_{part}_{comp}"] = samples.max(axis=0)
            even = projections[f"W_{kind}_even_{comp}"]
            odd = projections[f"W_{kind}_odd_{comp}"]
            np.testing.assert_allclose(even + odd, windows[1], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(even, even[:, ::-1], rtol=0, atol=1e-12)
            np.testing.assert_allclose(odd, -odd[:, ::-1], rtol=0, atol=1e-12)
log_msg.append("cuts at q1 = " + ", ".join(f"{q1[j]:.6f}" for j in cut_indices))
log_msg.append("command: PYTHONPATH=. " + command)
print("\n".join(log_msg))
np.savez(out + ".npz", q0=q0, q1=q1, k0=k0, times=slice_times, dropped_sites=np.array(args.drop_sites, int),
         ns=NS, center=C, sig_x=sig_x, integration_dt=DT, integration_dx=DX,
         source_j0=args.j0, source_j1=args.j1, label=args.label, command=command,
         cut_q1=q1[cut_indices], q0_plot_limits=args.q0_range, window_factors=window_factors,
         reference_projection="Re connected correlator", vacuum_subtraction="hybrid MPS",
         trotter_correction="none", **g.meta, **details, **projections,
         **{f"window_hw_{c}": window_hw[c] for c in W_hw},
         **{f"window_ref_{c}": window_ref[c] for c in W_ref},
         **{f"W_hw_{c}": W_hw[c] for c in W_hw}, **{f"W_ref_{c}": W_ref[c] for c in W_ref},
         **{f"W_err_{c}": W_err[c] for c in W_err}, **{f"band_{c}": band[c] for c in band},
         **{f"corr_{c}": corr[c] for c in corr})

# ---- figures: retain the full tensor and optionally add parity comparisons.
comps = [c for c in ("00", "01", "10", "11") if c in W_hw]
outputs = [out + ".npz"]
actual_tmax = max(float(details[f"times_{c}"][-1]) for c in comps)
hw_steps = ", ".join(f"{step:g}" for step in sorted(set(source_dt.values())))
display_label = re.sub(r"ibm_([A-Za-z0-9]+)", lambda match: dev(match.group(1)), args.label)
title = rf"{display_label}: Re-correlator tensor, $k_0={k0:.2f}$, $t\leq{actual_tmax:g}$"
method_note = (rf"Hybrid MPS vacuum subtraction; hardware $dt={hw_steps}$, "
               rf"MPS $dt={g.meta.get('dt_target', float('nan')):g}$; no Trotter correction")


def save_figure(fig, suffix):
    for extension in ("pdf", "png"):
        path = out + suffix + "." + extension
        fig.savefig(path, dpi=220)
        outputs.append(path)
    plt.close(fig)


for part in ([None, "even", "odd"] if args.parity else [None]):
    fig, axes = plt.subplots(2, len(comps), figsize=(2.8 * len(comps) + 0.6, 6.0),
                             layout="compressed", squeeze=False, sharex=True, sharey=True)
    for j, comp in enumerate(comps):
        values = [(W_hw[comp], display_label), (W_ref[comp], "MPS, matched window and mask")]
        if part:
            values = [(projections[f"W_hw_{part}_{comp}"], display_label),
                      (projections[f"W_ref_{part}_{comp}"], "MPS, matched window and mask")]
            # Keep the same component scale across even and odd figures.
            vmax = max(float(np.abs(projections[f"W_{kind}_{other}_{comp}"]).max())
                       for kind in ("hw", "ref") for other in ("even", "odd"))
        else:
            vmax = max(float(np.abs(value).max()) for value, _ in values)
        vmax = max(vmax, 1e-15)
        for i, (value, label) in enumerate(values):
            ax = axes[i, j]
            pm = ax.pcolormesh(q0, q1, value.T, cmap=OI_DIV, vmin=-vmax, vmax=vmax,
                               shading="nearest", rasterized=True)
            ax.grid(False)
            ax.set_aspect("equal")
            ax.set_xlim(*args.q0_range)
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "$0$", r"$\pi$"])
            if i == 1:
                ax.set_xlabel(r"$q^0$")
            if j == 0:
                ax.set_ylabel(r"$q^1$")
            symbol = rf"$W^{{{comp}}}$" if part is None else rf"$W^{{{comp}}}_{{\mathrm{{{part}}}}}$"
            ax.set_title(symbol + "\n" + label, fontsize=8)
        fig.colorbar(pm, ax=axes[:, j], shrink=0.85, pad=0.025, aspect=35)
    parity_note = ("\n" + rf"{part.capitalize()} under $q^1\to-q^1$" if part else "")
    fig.suptitle(title + parity_note + "\n" + method_note, fontsize=9)
    save_figure(fig, "_" + part if part else "")

parts = ("even", "odd") if args.parity else (None,)
fig, axes = plt.subplots(len(parts), len(comps), figsize=(2.8 * len(comps) + 0.6, 2.7 * len(parts) + 0.8),
                         constrained_layout=True, sharex=True, squeeze=False)
cut_colours = [OI[0], "0.45", OI[1], *OI[2:]]
for i, part in enumerate(parts):
    for j, comp in enumerate(comps):
        ax = axes[i, j]
        hw = W_hw[comp] if part is None else projections[f"W_hw_{part}_{comp}"]
        ref = W_ref[comp] if part is None else projections[f"W_ref_{part}_{comp}"]
        samples = window_hw[comp] if part is None else projections[f"window_hw_{part}_{comp}"]
        # The triangle inequality carries the same saved-error bound to each
        # parity projection, conditional on the diagonal site-error model.
        low, high = samples.min(axis=0) - W_err[comp], samples.max(axis=0) + W_err[comp]
        for n, k in enumerate(cut_indices):
            colour = cut_colours[n % len(cut_colours)]
            label = rf"$q^1={q1[k]:.2f}$"
            if np.isclose(q1[k], k0, rtol=0, atol=1e-8):
                label = rf"$q^1={q1[k]:.2f}=\bar k$"
            ax.fill_between(q0, low[:, k], high[:, k], color=colour, alpha=0.18, lw=0)
            ax.plot(q0, hw[:, k], color=colour, lw=1.2, label=label)
            ax.plot(q0, ref[:, k], color=colour, lw=0.9, ls="--")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlim(*args.q0_range)
        symbol = rf"$W^{{{comp}}}$" if part is None else rf"$W^{{{comp}}}_{{\mathrm{{{part}}}}}$"
        ax.set_title(symbol, fontsize=10)
        ax.set_ylabel(symbol)
        if i == len(parts) - 1:
            ax.set_xlabel(r"$q^0$")
        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc="upper right")
fig.suptitle(title + "\nSolid: hardware; dashed: matched MPS; bands: saved-error bound + window envelope"
             + "\n" + method_note, fontsize=9)
save_figure(fig, "_cuts")
manifest = dict(command="PYTHONPATH=. " + command, source_j0=args.j0, source_j1=args.j1,
                slices=source_paths, components=comps, hardware_dt=source_dt, mps_metadata=g.meta,
                actual_times={c: details[f"times_{c}"].tolist() for c in comps},
                retained_probes={c: details[f"mask_{c}"].sum(axis=1).tolist() for c in comps},
                mask_convention="True means retained; finite calibrated signal/error and kappa > threshold",
                kap_min=args.kap_min, reference_projection="real connected correlator",
                vacuum_subtraction="hybrid MPS", trotter_correction="none",
                uncertainty="saved-C_err bound plus window envelope; no MPS or Trotter uncertainty"
                            + ("; masked sites carry the matched reference |G| as error"
                               if args.unmeasured_error == "ref" else "; masked sites carry no error"),
                unmeasured_error=args.unmeasured_error,
                coverage={c: details[f"coverage_{c}"].tolist() for c in comps},
                error_propagation="diagonal (independent-site) approximation; C_err includes sampling and "
                                  "calibration terms; full spatial and shared-mirror covariance omitted",
                sum_rule="legacy error-weighted charge-conservation projection for 00 and 01"
                         + (f", only in slices retaining >= {args.sum_rule_min_coverage:g} of the sites"
                            if args.sum_rule_min_coverage > 0 else ""),
                sum_rule_min_coverage=args.sum_rule_min_coverage,
                q1_cuts=q1[cut_indices].tolist(), q0_plot_limits=args.q0_range, outputs=outputs)
Path(out + "_command.json").write_text(json.dumps(manifest, indent=2) + "\n")
Path(out + ".log").write_text("\n".join(log_msg) + "\n")
print(f"wrote {out}.{{npz,pdf,png,log}} and {out}_command.json")
