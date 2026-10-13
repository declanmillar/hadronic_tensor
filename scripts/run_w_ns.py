"""N_s-parametric MPS production of the two-current correlator grids for
the hadronic tensor (merges run_w_meson_ns50.py and run_w_j1_ns50.py, which
differed only in the inserted current).

Per run: one insertion (J0 at site CENTER or J1 at bond CENTER), probes J0
at all sites and/or J1 at all bonds, times 0..tmax, on the packet state and
(unless --vac-grid is given) on the vacuum for the pointwise subtraction.
The prepared state is built once and pickled (prep_common.stored_prep);
the npz is rewritten after every time slice. New grids go under
data/mps/runs/<tag>/<UTC timestamp>/. Use --resume --out <checkpoint.npz>
to continue an existing grid with the same time samples and parameters.

Output keys are those of the legacy files (times, k0, ns, center, corr_wp,
one_pt_wp, insert_1pt_wp, corr_vac, one_pt_vac, insert_1pt_vac, couplings,
dt_target, trunc) so htensor.tensor.load_mps_components reads them.

  PYTHONPATH=. .venv/bin/python scripts/run_w_ns.py --params data/wp10reg_params_k1.26_L3.npz \
      --insert j0 --probes both --tmax 8 --dt-out 0.5
  PYTHONPATH=. .venv/bin/python scripts/run_w_ns.py --state vac --insert j1  # vacuum grid only
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_common import add_state_args, resolve, build_prep, stored_prep, log  # noqa: E402
from htensor import backends  # noqa: E402
from htensor import currents as cur  # noqa: E402
from htensor.measure import split_current  # noqa: E402
from run_outputs import output_directory, output_file  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
add_state_args(p)
p.add_argument("--insert", choices=["j0", "j1"], default="j0")
p.add_argument("--probes", choices=["j0", "j1", "both"], default="both")
p.add_argument("--tmax", type=float, default=8.0)
p.add_argument("--dt-out", type=float, default=0.5)
p.add_argument("--dt-target", type=float, default=0.1)
p.add_argument("--trunc", type=float, default=1e-8)
p.add_argument("--chi-max", type=int, default=None, help="maximum evolution MPS bond dimension (default: uncapped)")
p.add_argument("--vac-grid", default=None, help="reuse corr_vac/one_pt_vac/insert_1pt_vac from this npz")
p.add_argument("--out", default=None, help="output NPZ (default: a new dated MPS run)")
p.add_argument("--resume", action="store_true", help="continue the existing NPZ named by --out")
args = resolve(p.parse_args())
if args.chi_max is not None and args.chi_max < 1:
    p.error("--chi-max must be positive")
out = output_file(args.out, resume=args.resume) if args.out or args.resume else None

lat, prep, info = build_prep(args)
C = args.center
times = np.arange(0.0, args.tmax + 1e-9, args.dt_out)
probes = []
if args.probes in ("j0", "both"):
    probes += [cur.charge_density(lat, v) for v in range(lat.ns)]
if args.probes in ("j1", "both"):
    probes += [cur.bond_current(lat, b, args.eta) for b in range(lat.ns)]
insert = cur.charge_density(lat, C) if args.insert == "j0" else cur.bond_current(lat, C, args.eta)
anc_site = min(split_current(insert)[1][0][0])
out = out or output_file(output_directory(args.tag, kind="mps")
                         / f"w_{args.tag}_ns{args.ns}_{args.insert}_{info['label']}.npz")
log(f"output: {out}")

# ---- which states to run
states = ["wp"] if args.state == "packet" else []
vac_from_file = None
if args.vac_grid:
    z = np.load(args.vac_grid)
    vac_from_file = {k: z[k] for k in ("corr_vac", "one_pt_vac", "insert_1pt_vac")}
    assert np.allclose(z["times"], times) and int(z["ns"]) == args.ns and int(z["center"]) == C
    # the vacuum two-current correlator depends on the INSERTED current and on
    # which probe blocks were measured, so a grid from another insertion or
    # probe set is not reusable (caught a real mis-launch on 2026-09-03)
    if "insert" in z.files and str(z["insert"]) != args.insert:
        raise SystemExit(f"--vac-grid {args.vac_grid} was computed for insert={str(z['insert'])}, "
                         f"not {args.insert}: the vacuum correlator differs")
    if "probes" in z.files and str(z["probes"]) != args.probes:
        raise SystemExit(f"--vac-grid {args.vac_grid} used probes={str(z['probes'])}, not {args.probes}")
    log(f"vacuum grid reused from {args.vac_grid}")
else:
    states.append("vac")

# ---- resume bookkeeping
done = {"wp": 0, "vac": 0}
store = {"wp": None, "vac": None}
if args.resume:
    z = np.load(out, allow_pickle=True)
    saved_chi = int(z["chi_max"]) if "chi_max" in z.files else -1
    if saved_chi != (args.chi_max if args.chi_max is not None else -1):
        raise ValueError(f"Resume checkpoint {out}: chi_max does not match this run")
    expected = dict(ns=args.ns, center=C, insert=args.insert, probes=args.probes,
                    m0=args.m0, g2=args.g2, eta=args.eta,
                    dt_target=args.dt_target, trunc=args.trunc, state=info["label"])
    for key, value in expected.items():
        if key not in z.files or np.asarray(z[key]).shape != () or z[key].item() != value:
            raise ValueError(f"Resume checkpoint {out}: {key} does not match this run")
    if not np.array_equal(z["times"], times):
        raise ValueError(f"Resume checkpoint {out}: time samples do not match this run")
    for s in states:
        if f"corr_{s}" in z.files:
            store[s] = {"corr": z[f"corr_{s}"], "one": z[f"one_pt_{s}"], "ins": complex(z[f"insert_1pt_{s}"])}
            done[s] = int(z[f"n_done_{s}"]) if f"n_done_{s}" in z.files else len(z["times"])
            if (store[s]["corr"].shape != (len(times), len(probes)) or
                    store[s]["one"].shape != (len(times), len(probes)) or
                    not 0 <= done[s] <= len(times)):
                raise ValueError(f"Resume checkpoint {out}: invalid {s} arrays or progress")
    log(f"resuming {out}: done {done}")


def save():
    d = dict(times=times, k0=(info.get("k0") if info.get("k0") is not None else np.nan), ns=args.ns,
             center=C, m0=args.m0, g2=args.g2, eta=args.eta, dt_target=args.dt_target, trunc=args.trunc,
             insert=args.insert, probes=args.probes, tag=args.tag, state=info["label"],
             chi_max=args.chi_max if args.chi_max is not None else -1,
             sigma_x=(info.get("sigma") if info.get("sigma") is not None else np.nan),
             wp_fidelity=(info.get("F") if info.get("F") is not None else np.nan))
    for s in ("wp", "vac"):
        if store[s] is not None:
            d[f"corr_{s}"], d[f"one_pt_{s}"], d[f"insert_1pt_{s}"] = store[s]["corr"], store[s]["one"], store[s]["ins"]
            d[f"n_done_{s}"] = done[s]
    if vac_from_file is not None:
        d.update(vac_from_file)
        d["n_done_vac"] = len(times)
    np.savez(out, **d)


for s in states:
    if done[s] >= len(times):
        continue
    if s == "wp":
        mps, perm, H = stored_prep(args, lat, prep, anc_site, info)
        stationary = False
    else:
        vac_args = argparse.Namespace(**{**vars(args), "state": "vac", "params": None, "prep_pkl": None})
        _, vprep, vinfo = build_prep(vac_args)
        mps, perm, H = stored_prep(vac_args, lat, vprep, anc_site, vinfo)
        stationary = True
    log(f"{s}: <H> = {H:.6f}")
    if store[s] is None:
        store[s] = {"corr": np.zeros((len(times), len(probes)), complex),
                    "one": np.zeros((len(times), len(probes)), complex), "ins": 0j}
    for i in range(done[s], len(times)):
        t = times[i]
        d = backends.hadamard_correlator_aer(
            lat, None, insert, probes, args.m0, args.g2, args.eta, [t], dt_target=args.dt_target,
            method="matrix_product_state", mps_trunc=args.trunc, mps_max_bond=args.chi_max,
            max_threads=args.threads,
            stationary_1pt=(stationary and t > 0), initial_mps=mps, initial_perm=perm)
        store[s]["corr"][i] = d.correlator[0]
        store[s]["one"][i] = d.probe_expect[0]
        if i == 0:
            store[s]["ins"] = complex(d.insert_expect)
        done[s] = i + 1
        save()
        log(f"{s} t={t:4.1f} done  C(t, x=0) = {d.correlator[0, C]:.5f}")
log(f"finished {out}")
