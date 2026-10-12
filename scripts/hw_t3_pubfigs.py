"""Publication figure candidates for the tier-3 hardware result, with the
gauge-post-selection cloud panel folded in.  Emits three PNGs/PDFs:

  hw_t3_combined  : ONE figure* -- slices (top) + cloud & tensor (bottom)
  hw_t3_figA      : TWO-figure option, part 1 -- integrand slices + cloud
  hw_t3_figB      : TWO-figure option, part 2 -- assembled W00 tensor

  PYTHONPATH=. .venv/bin/python scripts/hw_t3_pubfigs.py

Use --run-dir for analysis/<UTC timestamp>/ within a hardware campaign,
or --out for an explicit directory. Use --tensor to select a regenerated
coarse tensor. The default tensor is the archived Kingston result.
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from paper_style import *  # usetex + OI palette + dev()
import numpy as np

from htensor import Z2Lattice
from run_outputs import check_output_prefix, output_directory, analysis_directory, add_analysis_arguments

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--tensor", default="data/hw_w00_coarse_t3.npz",
               help="coarse Kingston tensor NPZ (default: archived result)")
add_analysis_arguments(p, output_help="explicit output directory")
args = p.parse_args()
out_dir = output_directory("kingston", args.out) if args.out else analysis_directory(args.run_dir)
for name in ("hw_t3_combined", "hw_t3_figA", "hw_t3_figB", "hw_cloud"):
    check_output_prefix(out_dir / name)

K0TAG = "k1.26"
NS, CENTER = 50, 24
lat = Z2Lattice(NS, pbc=True)
JOBS = [("data/hw/job_tier3k_T05MERGED.npz", 0.5, "277k"),
        ("data/hw/job_tier3i_T10RZZ.npz", 1.0, "20k"),
        ("data/hw/job_tier3d_BFIX.npz", 2.0, "120k")]
idl = np.load(f"data/hw_t3_ideal50_{K0TAG}.npz")
tru = np.load(f"data/w_meson_ns50_{K0TAG}_v3.npz")
w = np.load(args.tensor)
x = np.arange(NS) - CENTER
sq = np.array([lat.site_qubit(v) for v in range(NS)])
sgn = np.array([(-1) ** v for v in range(NS)])
bits = np.load("data/hw/sq_bits_kingston.npz")["bits"].astype(np.int8)


def draw_slice(a, path, T, sh, small=False):
    d = np.load(path)
    C_cal, C_err = d["C_cal"][0].real, d["C_err"][0]
    C_raw, kap = d["C"][0].real, d["kappa_v"][0]
    id_a, c_a, id_b = float(d["id_a"]), float(d["c_a"]), d["id_b"]
    ti = int(np.argmin(np.abs(tru["times"] - T)))
    row = int(np.argmin(np.abs(idl["times"] - T)))
    sxi = np.array([idl[f"XB_{v}"][row] for v in range(NS)]) \
        - id_b * idl["X"][row]
    syi = np.array([idl[f"YB_{v}"][row] for v in range(NS)]) \
        - id_b * idl["Y"][row]
    bi = np.array([idl[f"B_{v}"][row] for v in range(NS)])
    C_idl = (c_a * (sxi + 1j * syi) + id_a * (bi - id_b)
             + id_b * (idl["B_24"][0] - id_a) + id_a * id_b).real
    ms = 2.7 if small else 3.2
    a.plot(x, tru["corr_wp"][ti, :NS].real, "k-", lw=1.1, zorder=3)
    a.plot(x, C_idl, "--", color="#0072B2", lw=1.0, zorder=4)
    a.plot(x, C_raw, "x", color="0.55", ms=ms - 0.3, zorder=1)
    ok = kap > 0.05
    a.errorbar(x[ok], C_cal[ok], yerr=C_err[ok], fmt="o", color="#009E73",
               ms=ms, lw=0, elinewidth=0.8, capsize=1.3, zorder=5)
    a.plot(x[~ok], C_cal[~ok], "o", mfc="none", mec="#009E73", ms=ms,
           mew=0.7, zorder=5)
    a.text(0.03, 0.03, f"{int((~ok).sum())} masked", color="#009E73",
           fontsize=6, transform=a.transAxes)
    a.set_xlim(-25, 25)
    a.set_ylim(-0.17, 0.60)
    a.set_title(rf"$t={T}$ ({sh}, $\kappa_c={kap[CENTER]:.2f}$)", fontsize=8.5)
    a.set_xlabel(r"$x-x_{\rm packet}$", fontsize=8)


def slice_legend(a):
    a.plot([], [], "k-", lw=1.1, label="exact")
    a.plot([], [], "--", color="#0072B2", lw=1.0, label=r"ideal ($dt{=}0.5$)")
    a.plot([], [], "x", color="0.55", ms=2.8, label="raw")
    a.plot([], [], "o", color="#009E73", ms=3, label="mirror-cal.")
    a.legend(fontsize=6, loc="upper left", handlelength=1.3)


def draw_cloud(a, matched=False):
    one_t = tru["one_pt_wp"][0, :NS].real
    g_t = tru["corr_wp"][0, :NS].real - one_t * one_t[CENTER]

    def gcol(b, n):
        return ((-1) ** n * (1 - 2 * b[:, lat.site_qubit(n)])
                * (1 - 2 * b[:, lat.link_qubit(n - 1)])
                * (1 - 2 * b[:, lat.link_qubit(n)]))

    def conn(b):
        j = (sgn[None, :] - (1 - 2 * b[:, sq])) / 2
        o = j.mean(0)
        g = (j * j[:, [CENTER]]).mean(0) - o * o[CENTER]
        ge = (j * j[:, [CENTER]] - o[None, :] * o[CENTER]).std(0) \
            / np.sqrt(len(b))
        return g, ge, len(b)

    sel = np.ones(len(bits), bool)
    for n in range(CENTER - 2, CENTER + 3):
        sel &= gcol(bits, n) > 0
    g0, e0, n0 = conn(bits)
    g2, e2, n2 = conn(bits[sel])
    a.axhline(0, color="0.7", lw=0.6)
    # match the slice palette: black exact, grey x raw, red o mitigated
    a.plot(x, g_t, "k-", lw=1.3, zorder=3, label="exact (MPS)")
    a.plot(x, g0, "x", color="0.55", ms=5, zorder=2,
           label=rf"raw device ({n0//1000}k shots)")
    a.errorbar(x, g2, yerr=e2, fmt="o", color="#009E73", ms=4, lw=0,
               elinewidth=0.9, capsize=1.8, zorder=4,
               label=rf"gauge post-selected ({n2//1000}k kept)")
    a.set_xlim(-9, 9)
    a.set_ylim(-0.17, 0.30)
    fs = 9 if matched else 8
    a.set_xlabel(r"$x - x_{\rm insertion}$" if matched else r"$x-x_{\rm ins}$",
                 fontsize=fs)
    a.set_ylabel(r"$\langle J^0(x)\,J^0(x_0)\rangle_{\rm conn}$"
                 if matched else r"$\langle J^0 J^0\rangle_{\rm c}$",
                 fontsize=fs)
    a.legend(fontsize=7 if matched else 6, loc="lower right",
             handlelength=1.4)
    a.set_title(r"Connected screening cloud, $t=0$ (gauge post-selection)"
                if matched else "gauge post-selection",
                fontsize=9.5 if matched else 8.5)


def draw_heat(ax, W, ttl, vmax, ylab=True):
    pm = ax.pcolormesh(w["q0"], w["q1"], W.T, cmap=OI_DIV,
                       vmin=-vmax, vmax=vmax, shading="nearest")
    ax.set_xlabel(r"$q^0$")
    ax.set_title(ttl, fontsize=8.5)
    if ylab:
        ax.set_ylabel(r"$q^1$")
    else:
        ax.set_yticklabels([])
    return pm


def draw_cuts(ax):
    for j, col in zip((21, 17, 15), ("C1", "0.45", "#0072B2")):
        tot = w["W_err"] + w["band"][:, j] / 2
        ax.fill_between(w["q0"], w["W_hw"][:, j] - tot, w["W_hw"][:, j] + tot,
                        color=col, alpha=0.18, lw=0)
        lab = (rf"$q^1={w['q1'][j]:.2f} = \bar k$" if j == 17
               else rf"$q^1={w['q1'][j]:.2f}$")
        ax.plot(w["q0"], w["W_hw"][:, j], color=col, lw=1.2, label=lab)
        ax.plot(w["q0"], w["W_ref"][:, j], color=col, lw=0.9, ls="--")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$q^0$")
    ax.set_title("cuts (dashed: MPS)", fontsize=8.5)
    ax.legend(fontsize=6.2, loc="upper right")


VMAX = max(np.abs(w["W_ref"]).max(), np.abs(w["W_hw"]).max())

# ============ Candidate 1: single figure* (slices + tensor) ============
fig = plt.figure(figsize=(7.1, 5.6), constrained_layout=True)
gs = fig.add_gridspec(2, 12, height_ratios=[1, 1.12])
for i, (path, T, sh) in enumerate(JOBS):
    a = fig.add_subplot(gs[0, 4 * i:4 * i + 4])
    draw_slice(a, path, T, sh, small=True)
    if i == 0:
        a.set_ylabel(r"Re $C(t,x)$")
        slice_legend(a)
    else:
        a.set_yticklabels([])
ah1 = fig.add_subplot(gs[1, 0:4])
draw_heat(ah1, w["W_hw"], dev("kingston"), VMAX)
ah2 = fig.add_subplot(gs[1, 4:8])
pm = draw_heat(ah2, w["W_ref"], "MPS (same window)", VMAX, ylab=False)
fig.colorbar(pm, ax=[ah1, ah2], location="bottom", shrink=0.6,
             pad=0.14, aspect=34, label=r"$W^{00}$")
axc = fig.add_subplot(gs[1, 8:12])
draw_cuts(axc)
fig.suptitle(r"$W^{00}$ on 101 qubits: mirror-calibrated integrand slices "
             r"(top); assembled tensor, $t\leq2$ (bottom)", fontsize=9.5)
fig.savefig(out_dir / "hw_t3_combined.pdf", dpi=200)
fig.savefig(out_dir / "hw_t3_combined.png", dpi=200)

# ============ Candidate 2A: slices only (1x3) ============
figA, axA = plt.subplots(1, 3, figsize=(9.0, 3.2), constrained_layout=True,
                         sharey=True)
for i, (path, T, sh) in enumerate(JOBS):
    draw_slice(axA[i], path, T, sh)
    if i == 0:
        axA[i].set_ylabel(r"Re $C(t,x)$")
        slice_legend(axA[i])
figA.suptitle(r"$W^{00}$ integrand on 101 qubits (ibm_kingston): "
              r"mirror-calibrated Hadamard-test slices", fontsize=10)
figA.savefig(out_dir / "hw_t3_figA.pdf", dpi=200)
figA.savefig(out_dir / "hw_t3_figA.png", dpi=200)

# ============ standalone: gauge-PS cloud, format-matched ============
figC, axCl = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
draw_cloud(axCl, matched=True)
figC.savefig(out_dir / "hw_cloud.pdf", dpi=200)
figC.savefig(out_dir / "hw_cloud.png", dpi=200)

# ============ Candidate 2B: assembled tensor + odd projection ============
odd = lambda W: 0.5 * (W - W[:, ::-1])
Oh, Or = odd(w["W_hw"]), odd(w["W_ref"])
OVMAX = max(np.abs(Or).max(), np.abs(Oh).max())
figB = plt.figure(figsize=(7.05, 4.7))
gsB = figB.add_gridspec(2, 5,
                        width_ratios=[0.88, 0.88, 0.06, 0.58, 1.55],
                        hspace=0.05, wspace=0.14, left=0.065,
                        right=0.99, top=0.93, bottom=0.105)
b1 = figB.add_subplot(gsB[0, 0])
draw_heat(b1, w["W_hw"], dev("kingston"), VMAX)
b1.set_aspect("equal")
b2 = figB.add_subplot(gsB[0, 1])
pmB = draw_heat(b2, w["W_ref"], "MPS", VMAX, ylab=False)
b2.set_aspect("equal")
for a_ in (b1, b2):
    a_.set_xlabel("")
    a_.set_xticklabels([])
cax1 = figB.add_subplot(gsB[0, 2])
figB.colorbar(pmB, cax=cax1, label=r"$W^{00}$")
b3 = figB.add_subplot(gsB[1, 0])
pm3 = b3.pcolormesh(w["q0"], w["q1"], Oh.T, cmap=OI_DIV, vmin=-OVMAX,
                    vmax=OVMAX, shading="nearest")
b3.set_aspect("equal")
b3.set_xlabel(r"$q^0$"); b3.set_ylabel(r"$q^1$")
b3.text(0.04, 0.95, r"$A = 0.85 \pm 0.34$", transform=b3.transAxes,
        fontsize=7.5, va="top")
b4 = figB.add_subplot(gsB[1, 1])
pm4 = b4.pcolormesh(w["q0"], w["q1"], Or.T, cmap=OI_DIV, vmin=-OVMAX,
                    vmax=OVMAX, shading="nearest")
b4.set_aspect("equal")
b4.set_xlabel(r"$q^0$"); b4.set_yticklabels([])
cax2 = figB.add_subplot(gsB[1, 2])
figB.colorbar(pm4, cax=cax2, label=r"$W^{00}_{\rm odd}$")
bc = figB.add_subplot(gsB[:, 4])
draw_cuts(bc)
bc.set_ylabel(r"$W^{00}$")
# the equal-aspect heatmaps shrink inside their cells; match the cuts
# panel to their realized vertical extent
figB.canvas.draw()
for a_ in (b1, b2, b3, b4):
    a_.apply_aspect()
p1, p3, pc = b1.get_position(), b3.get_position(), bc.get_position()
bc.set_position([pc.x0, p3.y0, pc.width, p1.y1 - p3.y0])
figB.savefig(out_dir / "hw_t3_figB.pdf", dpi=200,
             bbox_inches="tight", pad_inches=0.03)
figB.savefig(out_dir / "hw_t3_figB.png", dpi=200,
             bbox_inches="tight", pad_inches=0.03)
print(f"wrote hw_t3_combined, hw_t3_figA, hw_t3_figB, hw_cloud (.pdf/.png) in {out_dir}")
