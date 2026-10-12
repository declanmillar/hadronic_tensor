"""All four components W^{mu nu}(q0, q1) from the production MPS grids, with
the lattice Ward / symmetry residual table and a paper-style figure.

  PYTHONPATH=. .venv/bin/python scripts/analyze_w_tensor.py \
      data/w_meson_ns50_k1.26_v3.npz data/w_meson_j1_ns50_k1.26.npz

Writes w_tensor_ns{ns}_k{k0:.2f}_W.npz and figures in a new directory under
data/mps/runs/<label>/<UTC timestamp>/. An optional positional output prefix
selects another location; existing outputs are never overwritten.
Conventions: htensor/tensor.py.

Pass --parity to plot the even and odd parts under q1 -> -q1 in the hardware
publication style, with q0 horizontal and one colour scale per component.
Saves PNG/PDF heatmaps and companion *_cuts figures. --q0-range changes only
the displayed energy range; --cut-q1 selects momentum cuts (snapped to the
grid). Cut bands show the envelope of the existing 0.75/1/1.5 window scan,
projected before forming the envelope. Arrays are saved in the output NPZ.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_style import OI, OI_DIV  # noqa: E402
from htensor import asymmetry, tensor  # noqa: E402
from run_outputs import check_output_prefix, output_directory  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("j0", help="MPS production grid with J0 insertion")
p.add_argument("j1", help="MPS production grid with J1 insertion")
p.add_argument("out_tag", nargs="?", help="output prefix (existing positional interface)")
p.add_argument("--parity", action="store_true", help="plot even/odd projections under q1 reflection")
p.add_argument("--label", default="", help="dataset label, e.g. prodR or relA")
p.add_argument("--q0-range", nargs=2, type=float, default=(-1.0, 5.0),
               metavar=("MIN", "MAX"), help="parity display range only (default: -1 5)")
p.add_argument("--cut-q1", nargs="+", type=float, default=(0.75, 1.26, 2.26),
               metavar="Q1", help="parity cuts, nearest grid momenta (default: 0.75 1.26 2.26)")
args = p.parse_args()
if not (np.isfinite(args.q0_range).all() and args.q0_range[0] < args.q0_range[1]):
    p.error("--q0-range requires finite MIN < MAX")
if not np.isfinite(args.cut_q1).all():
    p.error("--cut-q1 requires finite momenta")
path_j0, path_j1, out_tag = args.j0, args.j1, args.out_tag
if out_tag:
    out_tag = check_output_prefix(out_tag)

g = tensor.load_mps_components(path_j0, path_j1)
lat, k0 = g.lat, g.k0
tag = out_tag or check_output_prefix(output_directory(args.label or "tensor", kind="mps")
                                     / f"w_tensor_ns{lat.ns}_k{k0:.2f}")

q0 = np.arange(-1.0, 6.001, 0.04)
ks = np.arange(-(lat.nx // 2), lat.nx // 2 + 1)
q1 = 2 * np.pi * ks / lat.nx
W = tensor.assemble_W(g, q0, q1)

print(f"# {path_j0} + {path_j1}: ns={lat.ns} k0={k0:.4f} t_max={g.times[-1]}")
print("continuity, integral form (raw wp grids) :", tensor.continuity_residual_xt(g))
print("continuity, integral form (connected)    :", tensor.continuity_residual_xt(g, use_raw=False))
print("Ward, q-space q0 W^{0nu} vs qhat W^{1nu} :", tensor.ward_residual_q(W, q0, q1))
print("insertion-side Ward (eigenstate only)    :", tensor.insertion_ward_residual(W, q0, q1))
if abs(k0) < 1e-9:
    print("parity about insertion (rest; 00/10 approx):", tensor.parity_residuals(g))
for comp in tensor.COMPONENTS:
    w, s = W[comp]
    print(f"W^{comp}: max|W| = {np.abs(w).max():.4f}  window spread/max = "
          f"{s.max()/np.abs(w).max():.3f}  C(0,0) = {g.g[comp][0][np.argmin(np.abs(g.x[comp]))]:.4f}")

projections, window_bounds, plot_meta = {}, {}, {}
if args.parity:
    if args.q0_range[0] < q0[0] - 1e-12 or args.q0_range[1] > q0[-1] + 1e-12:
        p.error(f"--q0-range must lie within the transform grid [{q0[0]:g}, {q0[-1]:g}]")
    if not (np.all(np.diff(q1) > 0) and np.allclose(q1, -q1[::-1], rtol=0, atol=1e-12)):
        raise ValueError("Even/odd projection requires an increasing, reflection-symmetric q1 grid")
    if min(args.cut_q1) < q1[0] or max(args.cut_q1) > q1[-1]:
        p.error(f"--cut-q1 must lie within the momentum grid [{q1[0]:g}, {q1[-1]:g}]")
    cut_indices = [int(np.argmin(np.abs(q1 - q))) for q in args.cut_q1]
    if len(set(cut_indices)) != len(cut_indices):
        p.error("--cut-q1 values must select distinct grid momenta")
    for comp in ("00", "01", "10", "11"):
        w = W[comp][0]
        if w.shape != (len(q0), len(q1)) or not np.isfinite(w).all():
            raise ValueError(f"Invalid tensor array for component {comp}")
        odd = asymmetry.odd(w)
        even = 0.5 * (w + w[:, ::-1])
        np.testing.assert_allclose(even + odd, w, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(even, even[:, ::-1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(odd, -odd[:, ::-1], rtol=0, atol=1e-12)
        projections[f"even_{comp}"], projections[f"odd_{comp}"] = even, odd

    # A projection of the original spread is not a parity uncertainty:
    # project each window realization before taking its minimum/maximum.
    variants = [tensor.assemble_W(g, q0, q1, factors=(factor,))
                for factor in (0.75, 1.5)]
    for comp in ("00", "01", "10", "11"):
        for part in ("even", "odd"):
            samples = [projections[f"{part}_{comp}"]]
            for variant in variants:
                w = variant[comp][0]
                samples.append(asymmetry.odd(w) if part == "odd"
                               else 0.5 * (w + w[:, ::-1]))
            window_bounds[f"{part}_window_min_{comp}"] = np.min(samples, axis=0)
            window_bounds[f"{part}_window_max_{comp}"] = np.max(samples, axis=0)
    plot_meta = dict(q0_plot_limits=args.q0_range, cut_q1=q1[cut_indices],
                     window_factors=(0.75, 1.0, 1.5))
    print("cuts at q1 =", ", ".join(f"{q1[j]:.6f}" for j in cut_indices))

sigma_t, sigma_x = tensor.default_windows(g)
np.savez(tag + "_W.npz", q0=q0, q1=q1, k0=k0, ns=lat.ns, times=g.times,
         center=g.center, sigma_t=sigma_t, sigma_x=sigma_x,
         source_j0=path_j0, source_j1=path_j1, label=args.label,
         **g.meta, **{f"W_{c}": W[c][0] for c in W},
         **{f"spread_{c}": W[c][1] for c in W}, **projections,
         **window_bounds, **plot_meta)

if args.parity:
    # Reflect momentum for every component; mixed-current components can have
    # a nonzero odd part even for a rest packet. This is not a boost-only test.
    fig, axes = plt.subplots(2, 4, figsize=(11.8, 5.3), layout="compressed",
                             sharex=True, sharey=True)
    for j, comp in enumerate(("00", "01", "10", "11")):
        even, odd = projections[f"even_{comp}"], projections[f"odd_{comp}"]
        vmax = max(float(np.abs(even).max()), float(np.abs(odd).max()), 1e-15)
        for i, (part, w) in enumerate((("even", even), ("odd", odd))):
            ax = axes[i, j]
            pm = ax.pcolormesh(q0, q1, w.T, cmap=OI_DIV, shading="nearest",
                               vmin=-vmax, vmax=vmax, rasterized=True)
            ax.set_title(rf"$W^{{{comp}}}_{{\mathrm{{{part}}}}}$", fontsize=10)
            ax.grid(False)
            ax.set_aspect("equal")
            ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "$0$", r"$\pi$"])
            ax.set_ylim(-np.pi, np.pi)
            ax.set_xlim(*args.q0_range)
            if i == 1:
                ax.set_xlabel(r"$q^0$")
            if j == 0:
                ax.set_ylabel(r"$q^1$")
        fig.colorbar(pm, ax=axes[:, j], pad=0.025, shrink=0.85, aspect=35,
                     label=rf"$W^{{{comp}}}$")
    title_label = args.label + ": " if args.label else ""
    title = title_label + rf"MPS hadronic tensor, $N_s={lat.ns}$, $k_0={k0:.2f}$, $0\leq t\leq {g.times[-1]:g}$"
    fig.suptitle(
        title
        + "\n" + r"$W_{\mathrm{even/odd}}(q^0,q^1)=\frac{1}{2}[W(q^0,q^1)\pm W(q^0,-q^1)]$",
        fontsize=10)
    for suffix in ("pdf", "png"):
        fig.savefig(tag + "." + suffix, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(11.8, 5.8), constrained_layout=True,
                             sharex=True)
    # The hardware figure's low / packet / high momentum cut colours.
    cut_colours = [OI[0], "0.45", OI[1], *OI[2:]]
    for j, comp in enumerate(("00", "01", "10", "11")):
        for i, part in enumerate(("even", "odd")):
            ax = axes[i, j]
            w = projections[f"{part}_{comp}"]
            low = window_bounds[f"{part}_window_min_{comp}"]
            high = window_bounds[f"{part}_window_max_{comp}"]
            for n, k in enumerate(cut_indices):
                colour = cut_colours[n % len(cut_colours)]
                label = rf"$q^1={q1[k]:.2f}$"
                if np.isclose(q1[k], k0, rtol=0, atol=1e-8):
                    label = rf"$q^1={q1[k]:.2f}=\bar k$"
                ax.fill_between(q0, low[:, k], high[:, k], color=colour,
                                alpha=0.18, lw=0)
                ax.plot(q0, w[:, k], color=colour, lw=1.2, label=label)
            ax.axhline(0, color="0.6", lw=0.6)
            ax.set_title(rf"$W^{{{comp}}}_{{\mathrm{{{part}}}}}$", fontsize=10)
            ax.set_xlim(*args.q0_range)
            ax.set_ylabel(rf"$W^{{{comp}}}_{{\mathrm{{{part}}}}}$")
            if i == 1:
                ax.set_xlabel(r"$q^0$")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(title + "\nMomentum cuts; bands: window-scan envelope", fontsize=10)
    for suffix in ("pdf", "png"):
        fig.savefig(tag + "_cuts." + suffix, dpi=220)
    plt.close(fig)
    print("wrote", tag + "_W.npz", tag + ".pdf", tag + ".png")
    print("wrote", tag + "_cuts.pdf", tag + "_cuts.png")
    sys.exit(0)

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(11.2, 6.6), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
labels = {"00": r"$W^{00}$", "10": r"$W^{10}$", "01": r"$W^{01}$", "11": r"$W^{11}$"}
for i, comp in enumerate(tensor.COMPONENTS):
    ax = fig.add_subplot(gs[0, i])
    w = W[comp][0]
    if comp == "00":
        pm = ax.pcolormesh(q1, q0, w, cmap="Blues", shading="nearest", vmin=0,
                           vmax=np.percentile(w, 99.5), rasterized=True)
    else:
        v = np.percentile(np.abs(w), 99.5)
        pm = ax.pcolormesh(q1, q0, w, cmap="RdBu_r", shading="nearest", vmin=-v, vmax=v,
                           rasterized=True)
    ax.set_xlabel(r"$q^1$")
    if i == 0:
        ax.set_ylabel(r"$q^0$")
    ax.set_title(labels[comp] + rf"$(q^0,q^1)$, $k_0={k0:.2f}$", fontsize=10)
    ax.grid(False)
    fig.colorbar(pm, ax=ax, pad=0.02)

# bottom left: direct W^{11} vs Ward-converted (q0/qhat)^2 W^{00}
ax = fig.add_subplot(gs[1, 0:2])
conv = tensor.ward_converted_W11(W["00"][0], q0, q1)
for i, j in enumerate([lat.nx // 2 + 2, lat.nx // 2 + 5, lat.nx // 2 + 8]):
    ax.plot(q0, W["11"][0][:, j], color=OI[i], lw=1.8, label=rf"$q^1={q1[j]:.2f}$ direct")
    ax.plot(q0, conv[:, j], color=OI[i], lw=1.2, ls="--", label="Ward from $W^{00}$")
ax.axhline(0, color="0.4", lw=0.7)
ax.set_xlabel(r"$q^0$")
ax.set_ylabel(r"$W^{11}$")
ax.set_title(r"$W^{11}$: direct vs $(q^0/\hat q)^2 W^{00}$", fontsize=10)
ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)

# bottom right: the two mixed components, cuts
ax = fig.add_subplot(gs[1, 2:4])
for i, j in enumerate([lat.nx // 2 + 2, lat.nx // 2 + 5, lat.nx // 2 + 8]):
    ax.plot(q0, W["10"][0][:, j], color=OI[i], lw=1.8, label=rf"$W^{{10}}$, $q^1={q1[j]:.2f}$")
    ax.plot(q0, W["01"][0][:, j], color=OI[i], lw=1.2, ls=":", label=r"$W^{01}$")
ax.axhline(0, color="0.4", lw=0.7)
ax.set_xlabel(r"$q^0$")
ax.set_ylabel(r"$W^{10},\,W^{01}$")
ax.set_title("mixed components (cuts)", fontsize=10)
ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)
fig.savefig(tag + ".pdf", dpi=200)
print("wrote", tag + "_W.npz", tag + ".pdf")
