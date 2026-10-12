"""W00(q0,q1) assembled from HARDWARE data at coarse energy resolution.

Time slices of the connected, vacuum-subtracted integrand G(t,x):
  t=0   : equal-time <J0(x)J0(c)> from the existing S(q) kingston
          bitstrings (J0 is diagonal -> every pair is free, no ancilla,
          no kappa inversion; 30k shots)
  t=0.5 : tier-3 part a (mirror-calibrated Hadamard test, 50k shots)
  t=1.0 : tier-3 part b (150k shots)
Vacuum piece from the MPS production run (hybrid subtraction).  One-sided
Gaussian-windowed transform exactly as scripts/analyze_w_meson.py; the MPS
reference is windowed IDENTICALLY (same 3 times) so the comparison is
apples-to-apples.  delta-q0 ~ pi/t_max is coarse and labeled as such.

  PYTHONPATH=. .venv/bin/python scripts/hw_w00_coarse.py [tmax] [ps_win] [sig_xf]

Use --run-dir for analysis/<UTC timestamp>/ within a hardware campaign,
or --out for an explicit directory. Existing outputs are never overwritten.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from htensor import Z2Lattice, analysis
from run_outputs import check_output_prefix, output_directory, analysis_directory, add_analysis_arguments

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("tmax", nargs="?", type=float, default=1.0)
p.add_argument("ps_win", nargs="?", type=int, default=2)
p.add_argument("sig_xf", nargs="?", type=float, default=2.0)
add_analysis_arguments(p, output_help="explicit output directory")
args = p.parse_args()
TMAX, PS_WIN = args.tmax, args.ps_win
out = check_output_prefix((output_directory("kingston", args.out) if args.out else analysis_directory(args.run_dir))
                          / f"hw_w00_coarse_t{TMAX:.0f}")

NS, VC = 50, 24
K0TAG = "k1.26"
KAP_MIN = 0.05
# gauge-sector post-selection on the t=0 slice: keep only shots with
# G_n=+1 (physical) on the PS_WIN stabilizers each side of the insertion.
# Projects onto the gauge-invariant subspace where the screening cloud
# lives, un-damping the connected correlator ~2x -- truth-independent
# (never tuned to the MPS answer), at a shot cost.  PS_WIN=0 disables.
lat = Z2Lattice(NS, pbc=True)
tru = np.load(f"data/w_meson_ns50_{K0TAG}_v3.npz")

TIER3 = {0.5: "data/hw/job_tier3k_T05MERGED.npz",
         1.0: "data/hw/job_tier3i_T10RZZ.npz",
         1.5: "data/hw/job_tier3h_T15RZZ.npz",
         2.0: "data/hw/job_tier3f_T20RZZ.npz",
         3.0: "data/hw/job_tier3g_T30RZZ.npz"}

# ---- connected vacuum reference (MPS)
g_vac = analysis.subtract(tru["corr_vac"], None, tru["one_pt_vac"],
                          complex(tru["insert_1pt_vac"]))[:, :NS]
g_wp_mps = analysis.subtract(tru["corr_wp"], None, tru["one_pt_wp"],
                             complex(tru["insert_1pt_wp"]))[:, :NS]
G_mps = g_wp_mps - g_vac

# ---- t=0 slice from S(q) bitstrings, with gauge post-selection
bits = np.load("data/hw/sq_bits_kingston.npz")["bits"].astype(np.int8)
sq = np.array([lat.site_qubit(v) for v in range(NS)])
sgn = np.array([(-1) ** v for v in range(NS)])


def gauss_col(b, n):
    """Per-shot Gauss stabilizer G_n = (-1)^n Z_n X_{L,n-1} X_{L,n}; links
    were measured in the X basis (H before readout) so the link bits carry
    the X eigenvalue directly."""
    return ((-1) ** n * (1 - 2 * b[:, lat.site_qubit(n)])
            * (1 - 2 * b[:, lat.link_qubit(n - 1)])
            * (1 - 2 * b[:, lat.link_qubit(n)]))


sel = np.ones(len(bits), bool)
for n in range(VC - PS_WIN, VC + PS_WIN + 1):
    if PS_WIN > 0:
        sel &= gauss_col(bits, n) > 0
bsel = bits[sel]
z = 1 - 2 * bsel[:, sq]
j0 = (sgn[None, :] - z) / 2
Nsh = len(j0)
one0 = j0.mean(0)
C0 = (j0 * j0[:, [VC]]).mean(0)
g0 = C0 - one0 * one0[VC]
prod = j0 * j0[:, [VC]] - one0[None, :] * one0[VC]
g0_err = prod.std(0) / np.sqrt(Nsh)

rows = {0.0: (g0 - g_vac[0].real, g0_err, np.ones(NS, bool))}
log_msg = [f"t=0.0: S(q) bitstrings, gauge PS_WIN={PS_WIN} keeps "
           f"{Nsh}/{len(bits)} shots, median err {np.median(g0_err):.4f}"]

# ---- tier-3 slices
for t, path in sorted(TIER3.items()):
    if t > TMAX:
        continue
    try:
        d = np.load(path)
    except FileNotFoundError:
        continue
    cn = f"t{t:.1f}"
    C, Ce = d["C_cal"][0], d["C_err"][0]
    bet, kap = d["beta_v"][0], d["kappa_v"][0]
    # per-run-calibrated merge saves b_cal directly; older single-job npzs
    # reconstruct it from the raw physics B and beta
    b_cal = (d["b_cal"][0] if "b_cal" in d.files
             else 0.5 + (d[f"raw_{cn}_B"] - 0.5) / bet)
    A0 = float(tru["insert_1pt_wp"].real)
    ti = int(np.argmin(np.abs(tru["times"] - t)))
    G_hw = (C - b_cal * A0) - g_vac[ti]
    ok = kap > KAP_MIN                       # uninvertible sites -> masked
    rows[t] = (G_hw, Ce, ok)
    log_msg.append(f"t={t}: tier-3, {int((~ok).sum())} sites masked, "
                   f"median err {np.median(Ce[ok]):.4f}")

times = np.array(sorted(rows))
G_hw = np.array([rows[t][0] for t in times]).real
G_err = np.array([rows[t][1] for t in times])
mask = np.array([rows[t][2] for t in times])
ti_mps = [int(np.argmin(np.abs(tru["times"] - t))) for t in times]
G_ref = G_mps[ti_mps].real

# ---- charge-conservation sum rule: sum_x <J0(x,t) J0(c,0)>_conn =
# <Q J0>_c = 0 exactly (Q conserved, also by the Trotter circuit), and the
# vacuum-subtracted G inherits it.  Decoherence damps the connected
# screening cloud and breaks the rule; project it back, error-weighted
# (least squares: dG_x ~ sigma_x^2), so the correction lands on the noisy
# near-center sites.  On masked sites the constraint uses the MPS value.
for i, t in enumerate(times):
    m = mask[i]
    target = -G_ref[i][~m].sum()          # exact 0 minus masked-site truth
    viol = G_hw[i][m].sum() - target
    w2 = G_err[i][m] ** 2
    G_hw[i][m] -= viol * w2 / w2.sum()
    log_msg.append(f"t={t}: sum-rule violation {viol:+.4f} projected out")

# ---- one-sided windowed transform (analyze_w_meson conventions)
x0 = analysis.ring_fold((np.arange(NS) - VC) / 2, lat.nx)
q0 = np.arange(-1.0, 6.001, 0.04)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)
q1 = 2 * np.pi * ks / lat.nx
DT, DX = 0.5, 0.5
SIG_XF = args.sig_xf
sig_x = lat.nx / SIG_XF


def onesided(G, msk, sig_t):
    wt = np.exp(-times ** 2 / (2 * sig_t ** 2)).copy()
    wt[0] *= 0.5
    wx = np.exp(-x0 ** 2 / (2 * sig_x ** 2))[None, :] * msk
    et = np.exp(1j * np.outer(q0, times))
    ex = np.exp(-1j * np.outer(x0, q1))
    return 2 * np.real(DT * DX * (et @ ((wt[:, None] * wx) * G) @ ex))


def werr(err, msk, sig_t):
    wt = np.exp(-times ** 2 / (2 * sig_t ** 2)).copy()
    wt[0] *= 0.5
    wx = np.exp(-x0 ** 2 / (2 * sig_x ** 2))[None, :] * msk
    var = ((wt[:, None] * wx) ** 2 * err ** 2).sum()
    return 2 * DT * DX * np.sqrt(var)        # conservative (coherent) bound


SIG_T = max(0.45, TMAX / 2)
W_hw = onesided(G_hw, mask, SIG_T)
W_ref = onesided(G_ref, mask, SIG_T)         # same times, window AND mask
W_e = werr(G_err, mask, SIG_T)
band = np.ptp(np.stack([onesided(G_hw, mask, f * SIG_T)
                        for f in (0.75, 1.0, 1.5)]), axis=0)

# ---- odd-q1 (boost) asymmetry: matched filter against the windowed MPS
# template on the wedge q0 in [-1, 2.5].  The filter is linear in G_hw,
# so the shot error propagates exactly through the FT kernel.
odd = lambda W: 0.5 * (W - W[:, ::-1])
reg = (q0 >= -1) & (q0 <= 2.5)
T = odd(W_ref)[reg]
den = float((T * T).sum())
A_hat = float((T * odd(W_hw)[reg]).sum()) / den
avar = 0.0
for i in range(len(times)):
    for xx in range(NS):
        if not mask[i][xx]:
            continue
        basis = np.zeros_like(G_hw)
        basis[i, xx] = 1.0
        K = float((T * odd(onesided(basis, mask, SIG_T))[reg]).sum()) / den
        avar += (K * G_err[i][xx]) ** 2
A_sys = 0.5 * np.ptp(
    [float((T * odd(onesided(G_hw, mask, f * SIG_T))[reg]).sum()) / den
     for f in (0.75, 1.0, 1.5)])
print(f"odd-q1 asymmetry (matched filter vs MPS template): "
      f"A = {A_hat:.2f} +- {np.sqrt(avar):.2f} (shot) "
      f"+- {A_sys:.2f} (window)   [A=1 -> full template asymmetry]")

print("\n".join(log_msg))
print(f"t_max={TMAX}, sigma_t={SIG_T:.2f}, delta-q0 ~ pi/t_max = "
      f"{np.pi/TMAX:.2f}; shot err on W = {W_e:.3f}")
print(f"max W: hw {W_hw.max():.3f}, mps-same-window {W_ref.max():.3f}; "
      f"corr(hw, ref) = {np.corrcoef(W_hw.ravel(), W_ref.ravel())[0,1]:.3f}")

# ---- figure: heatmaps + cuts
fig = plt.figure(figsize=(9.0, 3.2), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
vmax = max(np.abs(W_ref).max(), np.abs(W_hw).max())
for i, (Wp, ttl) in enumerate([(W_hw, "ibm_kingston"),
                               (W_ref, "MPS, same window")]):
    ax = fig.add_subplot(gs[i])
    pm = ax.pcolormesh(q1, q0, Wp, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       shading="nearest")
    ax.set_xlabel(r"$q^1$")
    ax.set_title(ttl, fontsize=9)
    if i == 0:
        ax.set_ylabel(r"$q^0$")
fig.colorbar(pm, ax=fig.axes[:2], shrink=0.85, pad=0.01)
axc = fig.add_subplot(gs[2])
for j, col in zip((21, 17, 15), ("C1", "C2", "C0")):
    axc.fill_between(q0, W_hw[:, j] - W_e - band[:, j] / 2,
                     W_hw[:, j] + W_e + band[:, j] / 2,
                     color=col, alpha=0.20, lw=0)
    axc.plot(q0, W_hw[:, j], color=col, lw=1.3,
             label=rf"$q^1={q1[j]:.2f}$")
    axc.plot(q0, W_ref[:, j], color=col, lw=1.0, ls="--")
axc.axhline(0, color="0.6", lw=0.6)
axc.set_xlabel(r"$q^0$")
axc.set_title("cuts: hw (solid$\\pm$band) vs MPS (dashed)", fontsize=9)
axc.legend(fontsize=7)
fig.suptitle(rf"$W^{{00}}(q^0,q^1)$ from hardware, $t\leq{TMAX:.0f}$"
             rf"  ($\delta q^0\sim{np.pi/TMAX:.1f}$)", fontsize=10)
fig.savefig(out + ".pdf", dpi=200)
fig.savefig(out + ".png", dpi=200)
np.savez(out + ".npz", q0=q0, q1=q1, W_hw=W_hw, W_ref=W_ref,
         W_err=W_e, band=band, times=times, sig_t=SIG_T,
         A_hat=A_hat, A_err=np.sqrt(avar), A_sys=A_sys)
print(f"wrote {out}.{{pdf,png,npz}}")
