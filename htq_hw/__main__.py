"""Command line: ``python -m htq_hw <audit|embed|report> [options]``.

  audit     logical circuit statistics (no transpilation)
  embed     choose and validate an embedding on a target
  report    offline transpile table (prep, gadget, step, n-step blocks) per
            target x basis, with the physics/mirror skeleton assertion
  plan      campaign manifest + shot plan (per-slice table, minutes at 250 us / 4 ms)
  ideal     noiseless ideal grids per insertion family (hw_cal_grids key set)
  check     submission-path validation: ISA circuits mapped back to the logical
            register vs the ideal grids (Aer statevector / MPS)
  rehearse  sample the ISA pubs in Aer (optional Pauli-trajectory noise) -> bits
            -> analyze -> slice files
  submit    build + submit the campaign (local testing mode unless --real)
  fetch     job result -> bits npz;  analyze  bits -> slice npz files
  bundle    zip the package with cards, requirements, audit and report

Targets: fake:boston | fake:kingston | fake:fez | fake:nighthawk | fake:miami
| grid:RxC | heavyhex:d | <real IBM backend name>.  Nothing is ever
submitted; real names only resolve a Target for transpilation.
"""

import argparse
import json
import pathlib
import sys
import time

from . import CENTER, DEFAULT_CARD, DT, MIRROR_EPS, NS, __version__
from .campaign import DT_HALF_TIMES as CP_DT_HALF_TIMES
from . import circuits as C
from . import target as T
from .model import Lattice

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def _center_for(card, ns):
    return card["center"] if ns == card["ns"] else ns // 2 - 1


def cmd_audit(args):
    card = C.load_card(args.card)
    lat = Lattice(args.ns)
    center = _center_for(card, args.ns)
    print(f"htq_hw {__version__}  card {card['name']}  Ns={lat.ns}  qubits {lat.n_qubits}+1 "
          f"ancilla  center site {center} (qubit {lat.site_qubit(center)})  "
          f"couplings {card['couplings']}  vacuum {C.card_n_layers(card)} layers, link_ref "
          f"'{C.card_link_ref(card)}'  dt {DT}  mirror eps {MIRROR_EPS}")
    prep = C.prep_circuit(card, lat.ns, center)
    print(f"prep    : {dict(prep.count_ops())}  2q {T.count_2q(prep)}  depth {prep.depth()}")
    from qiskit.circuit import Parameter
    step = C.trotter_block(lat, 1, Parameter("t"), *C.card_couplings(card))
    print(f"step    : {dict(step.count_ops())}  2q {T.count_2q(step)}")
    for kind in ("J0", "J1a", "J1b"):
        for acc in ("ring", "ladder"):
            g = C.insertion_gadget(lat, kind, center, acc)
            print(f"gadget {kind:3} {acc:6}: {dict(g.count_ops())}  coeff {C.GADGET_COEFF[kind]}"
                  f"{' * eta' if kind != 'J0' else ''}")
    for basis in C.BASES:
        m = C.basis_map(lat, basis)
        print(f"basis {basis:3}: matter {[m[q] for q in lat.matter_qubits[:4]]}... "
              f"links {[m[q] for q in lat.link_qubits[:2]]}...")


def cmd_embed(args):
    card = C.load_card(args.card)
    center = _center_for(card, args.ns)
    for spec in args.targets:
        be = T.resolve_backend(spec)
        g = T.Graph.from_backend(be)
        print(f"{spec}: {be.num_qubits} qubits, {g.n_edges} edges, "
              f"grid {T.grid_coordinates(g)[:2] if T.grid_coordinates(g) else None}")
        emb = T.choose_embedding(be, args.ns, center, mode=args.mode, log=log)
        emb.validate(g)
        info = {k: v for k, v in emb.info.items() if k != "cycle"}
        print(f"  {emb.summary()}  info {info}")
        print(f"  layout: {emb.layout}")


def cmd_report(args):
    card = C.load_card(args.card)
    rows = T.report(args.targets, ns=args.ns, steps=tuple(args.steps), bases=tuple(args.basis),
                    mode=args.mode, gadgets=tuple(args.gadget), card=card,
                    cache_dir=args.cache, seed=args.seed, log=log)
    print()
    print(T.format_table(rows))
    for spec in args.targets:
        be = T.resolve_backend(spec)
        if T.is_real_backend(be):
            lat = Lattice(args.ns)
            emb = T.choose_embedding(be, args.ns, _center_for(card, args.ns), mode=args.mode)
            b = T.transpile_bundle(be, lat, card, emb, (max(args.steps),), args.basis[0], "J0", args.seed, args.cache)
            full = C.readout_layer(b.base.compose(b.blocks[max(args.steps)]["physics"]), lat,
                                   b.blocks[max(args.steps)]["layout"], "Z")
            pst = T.per_shot_time(be, full)
            print(f"{spec}: measured per-shot time (deepest pub, readout + reset + overhead) = "
                  f"{pst['total_s'] * 1e6:.0f} us -> plan re-pins with --rep-time {pst['total_s']:.6f}")
        elif getattr(be, "_htq_standin_for", None):
            print(f"{spec}: {be._htq_standin_for} not visible, counts above are for the stand-in {T.backend_label(be)}; "
                  f"per-shot time assumed {250e-6 * 1e6:.0f} us until measured")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1, default=str)
        print(f"rows written to {args.json}")


def _ideal_template(args):
    """Default to the grids as they are actually installed in the cards."""
    if args.ideal:
        return args.ideal
    return str(C.CARD_DIR / "{card}" / "ideal_{family}.npz").replace("{card}", args.card or "{card}")


def _setup(args, basis=None, require_real=False):
    from . import campaign as CP
    card = C.load_card(args.card)
    lat = Lattice(args.ns)
    center = _center_for(card, args.ns)
    fractional = (basis or args.basis) == "rzz"
    # refuse offline substitutions before any (expensive) transpiling happens
    be = T.resolve_backend(args.target, fractional=fractional, allow_standin=not require_real)
    if require_real:
        T.require_real_backend(be, args.target, fractional=fractional)
    if getattr(be, "_htq_standin_for", None):
        log(f"{be._htq_standin_for} not visible ({be._htq_standin_reason}); using {T.backend_label(be)}")
    gauss_sites = None
    if getattr(args, "preset", None) and "gauss-midcircuit" in args.preset:
        from .gauss import gauss_ancilla_sites
        gauss_sites = gauss_ancilla_sites(lat, center)
    emb = T.choose_embedding(be, args.ns, center, mode=args.mode, log=log, gauss_sites=gauss_sites)
    if getattr(args, "preset", None):
        specs = CP.compose_presets(args.preset)
        card = None                     # cards come from the specs
    else:
        specs = CP.manifest(times=args.times or CP.DEFAULT_TIMES, j1_mirrors=args.j1_mirrors,
                            dither=args.dither, im=args.im, dt_half=args.dt_half,
                            dt_half_times=tuple(args.dt_half_times))
    return card, lat, center, be, emb, specs


def _plan(args, specs, info, be, pubs):
    from . import campaign as CP
    n2q = {n: info[n]["n2q"] for n in pubs}
    rep = args.rep_time
    if T.is_real_backend(be):
        deepest = max(pubs, key=lambda n: info[n]["n2q"])
        pst = T.per_shot_time(be, pubs[deepest])
        print(f"measured per-shot time on {T.backend_label(be)} (deepest pub {deepest}): circuit "
              f"{pst['circuit_us']} us + reset {pst['reset_us']} us + overhead {pst['overhead_us']} us = {pst['total_s'] * 1e6:.0f} us; re-pinning the plan")
        rep = pst["total_s"]
    budget = args.budget if args.budget is not None else (180.0 if getattr(args, "preset", None) else 36.0)
    return CP.shots_plan(specs, n2q, budget, rep, mirror_floor=args.mirror_floor,
                         weighting=args.weighting, shots_per_pub=args.shots_per_pub), rep


def cmd_plan(args):
    from . import campaign as CP
    card, lat, center, be, emb, specs = _setup(args)
    print(CP.manifest_summary(specs))
    pubs, info, _ = CP.build_pub_circuits(be, lat, card, emb, specs, args.basis, cache_dir=args.cache, log=log)
    plan, rep = _plan(args, specs, info, be, pubs)
    n2q = {n: info[n]["n2q"] for n in pubs}
    print(CP.format_plan(plan))
    if getattr(args, "preset", None):
        print(CP.preset_table(specs, plan["shots"], rep))
    jobs = CP.group_jobs(specs, args.max_pubs)
    print(f"{len(jobs)} jobs: sizes {[len(j) for j in jobs]}")
    for j in jobs[:3]:
        print("  ", j)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"specs": CP.spec_dicts(specs), "shots": plan["shots"], "kappa": plan["kappa"],
                       "rows": plan["rows"], "jobs": jobs, "n2q": n2q}, f, indent=1)
        print(f"plan written to {args.json}")


def cmd_ideal(args):
    from . import sim as S
    card = C.load_card(args.card)
    center = _center_for(card, args.ns)
    times = [0.0] + list(args.times or __import__("htq_hw.campaign", fromlist=["x"]).DEFAULT_TIMES)
    fams = args.families
    paths = S.write_ideal_grids(card, args.ns, center, fams, times, _ideal_template(args), mirror=args.mirror,
                                cap=args.cap, trunc=args.trunc, threads=args.threads, cache_dir=args.cache, log=log)
    print(paths)


def cmd_check(args):
    from . import sim as S
    card, lat, center, be, emb, specs = _setup(args)
    times = [0.0] + list(args.times or (0.5, 1.0))
    t0 = time.time()
    try:
        res = S.check(be, lat, card, emb, _ideal_template(args), times, args.families, args.basis, args.tol,
                      args.cap, args.threads, args.cache, log=log)
    finally:
        print(f"\nper-family / per-readout worst |diff| ({time.time() - t0:.0f}s):")
        for tag, groups in S.LAST_GROUPS.items():
            print(f"  {tag:22} " + "  ".join(f"{g}: {v:.2e}" for g, (v, k) in groups.items()))
    print(f"worst |diff| {max(res.values()):.2e} over {len(res)} circuits")


def cmd_rehearse(args):
    from . import campaign as CP, sim as S
    card, lat, center, be, emb, specs = _setup(args)
    pubs, info, _ = CP.build_pub_circuits(be, lat, card, emb, specs, args.basis, cache_dir=args.cache, log=log)
    plan, _rep = _plan(args, specs, info, be, pubs)
    shots = {n: max(16, int(v * args.shots_scale)) for n, v in plan["shots"].items()}
    if args.shots:
        shots = {n: int(args.shots) for n in shots}
    noise = tuple(args.noise) if args.noise else None
    out = S.rehearse(be, lat, card, emb, specs, shots, _ideal_template(args), args.out, args.basis, noise,
                     args.seed, args.cap, args.threads, args.cache, n_traj=args.n_traj, log=log)
    print(f"bits {out['bits']}\nmeta {out['meta']}\n{len(out['slices'])} slices under {args.out}")


def cmd_submit(args):
    from . import campaign as CP
    card, lat, center, be, emb, specs = _setup(args, require_real=args.real)
    pubs, info, _ = CP.build_pub_circuits(be, lat, card, emb, specs, args.basis, cache_dir=args.cache, log=log)
    plan, _rep = _plan(args, specs, info, be, pubs)
    shots = {n: max(16, int(v * args.shots_scale)) for n, v in plan["shots"].items()}
    jobs = CP.group_jobs(specs, args.max_pubs)
    if args.only_jobs:
        jobs = [jobs[i] for i in args.only_jobs]
    service = None
    mode = be
    if args.real:
        if args.target.startswith(("fake:", "grid:", "heavyhex:")):
            raise SystemExit("--real needs a real backend name as --target")
        stamp = T.require_real_backend(be, args.target, fractional=(args.basis == "rzz"))
        if args.confirm != be.name:
            raise SystemExit(
                f"--real spends the allocation: confirm the device with --confirm {be.name} "
                f"(you passed {args.confirm!r})")
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        n_shots = sum(shots[n] for j in jobs for n in j)
        print(f"REAL submission to {stamp['name']} ({stamp['num_qubits']}q): {len(jobs)} jobs, "
              f"{n_shots} shots")
    elif T.is_real_backend(be):
        # SamplerV2(mode=<IBMBackend>) submits to hardware whatever we print
        raise SystemExit(
            f"--target {args.target} resolves to the LIVE device {be.name}; submitting without --real "
            f"would still run on hardware and spend the allocation. Use --real --confirm {be.name} to "
            f"submit deliberately, or --target fake:nighthawk to rehearse.")
    else:
        print(f"local testing mode on {T.backend_label(be)}: {len(jobs)} jobs")
    recs = CP.submit(mode, pubs, info, shots, jobs, lat, emb, args.basis, args.out, service, args.guard,
                     tag=args.tag, log=log)
    if not args.real and args.fetch:
        for r in recs:
            CP.fetch(r["job"], r["meta"], args.out, log=log)


def cmd_fetch(args):
    from . import campaign as CP
    for path in args.meta:
        with open(path) as f:
            meta = json.load(f)
        CP.fetch(meta["job_id"], meta, args.out, log=log)


def cmd_analyze(args):
    from . import analyze as A
    from . import campaign as CP
    if args.list_prefixes:
        print("prefixes present:", A.available_prefixes(args.bits) or "(none: unprefixed pubs only)")
        return
    card = C.load_card(args.card)
    center = _center_for(card, args.ns)
    prefix = args.prefix
    if prefix is None and args.preset:
        prefix = CP.name_prefix(args.preset, args.card)
    A.analyze(args.bits, _ideal_template(args), args.slices, args.ns, center, args.times, args.components,
              card["couplings"]["eta"], args.backend, log=log, prefix=prefix, card=args.card,
              wing_surrogate=args.wing_surrogate)


def cmd_bundle(args):
    import io
    import zipfile
    from contextlib import redirect_stdout
    root = pathlib.Path(__file__).parent
    out = pathlib.Path(args.out or f"htq_hw_{__version__}.zip")
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_audit(args)
    files = [p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in (".py", ".json", ".txt", ".md", ".npz")]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"htq_hw/{p.relative_to(root)}")
        z.writestr("htq_hw/run_audit.txt", buf.getvalue())
        if args.report_json and pathlib.Path(args.report_json).exists():
            rows = json.load(open(args.report_json))
            z.writestr("htq_hw/report_table.txt", T.format_table(rows))
        z.writestr("htq_hw/BUNDLE.txt", f"htq_hw {__version__} bundled {time.strftime('%Y-%m-%d %H:%M')}\n"
                   f"{len(files)} package files; run: pip install -r htq_hw/requirements.txt; "
                   f"python -m htq_hw report --targets fake:boston fake:nighthawk\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(files)} files)")


def _add_campaign_args(p):
    p.add_argument("--target", default="fake:boston")
    p.add_argument("--basis", default="cz", choices=["cz", "rzz"])
    p.add_argument("--mode", default="auto", choices=["auto", "ring", "ladder", "grid", "transpiler"])
    p.add_argument("--times", type=float, nargs="+", default=None)
    p.add_argument("--j1-mirrors", action="store_true")
    p.add_argument("--dither", action="store_true")
    p.add_argument("--im", action="store_true")
    p.add_argument("--dt-half", action="store_true",
                   help="Trotter-systematic control: repeat --dt-half-times with dt = 0.25 (mirrors included)")
    p.add_argument("--dt-half-times", type=float, nargs="+", default=list(CP_DT_HALF_TIMES))
    p.add_argument("--preset", nargs="+", default=None, choices=list(__import__("htq_hw.campaign", fromlist=["x"]).PRESET_NAMES),
                   help="compose campaign presets (each card's pubs carry a 'preset.card:' prefix)")
    p.add_argument("--budget", type=float, default=None, help="minutes at --rep-time (default 36, 180 with presets)")
    p.add_argument("--weighting", default=None, choices=["equal", "kappa"],
                   help="shots per pub: equal (default with presets) or kappa-weighted (default otherwise)")
    p.add_argument("--shots-per-pub", type=int, default=60000)
    p.add_argument("--rep-time", type=float, default=250e-6)
    p.add_argument("--mirror-floor", type=int, default=30000)
    p.add_argument("--max-pubs", type=int, default=8)
    p.add_argument("--ideal", default=None, help="ideal grid template with {family}")
    p.add_argument("--cache", default=str(pathlib.Path.home() / ".cache" / "htq_hw"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m htq_hw", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", default=DEFAULT_CARD)
    ap.add_argument("--ns", type=int, default=NS)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit").set_defaults(fn=cmd_audit)
    e = sub.add_parser("embed")
    e.add_argument("--targets", nargs="+", default=["fake:boston", "fake:nighthawk"])
    e.add_argument("--mode", default="auto", choices=["auto", "ring", "ladder", "grid", "transpiler"])
    e.set_defaults(fn=cmd_embed)
    r = sub.add_parser("report")
    r.add_argument("--targets", nargs="+", default=["fake:boston", "fake:nighthawk"])
    r.add_argument("--steps", nargs="+", type=int, default=[1, 12])
    r.add_argument("--basis", nargs="+", default=["cz", "rzz"], choices=["cz", "rzz"])
    r.add_argument("--gadget", nargs="+", default=["J0", "J1a"], choices=["J0", "J1a", "J1b"])
    r.add_argument("--mode", default="auto", choices=["auto", "ring", "ladder", "grid", "transpiler"])
    r.add_argument("--cache", default=str(pathlib.Path.home() / ".cache" / "htq_hw"),
                   help="qpy cache directory ('' disables)")
    r.add_argument("--seed", type=int, default=T.SEED)
    r.add_argument("--json", default=None, help="write the rows as JSON")
    r.set_defaults(fn=cmd_report)
    pl = sub.add_parser("plan")
    _add_campaign_args(pl)
    pl.add_argument("--json", default=None)
    pl.set_defaults(fn=cmd_plan)
    idl = sub.add_parser("ideal")
    idl.add_argument("--times", type=float, nargs="+", default=None)
    idl.add_argument("--families", nargs="+", default=["j0", "j1p1", "j1p2"])
    idl.add_argument("--ideal", default=None)
    idl.add_argument("--mirror", action="store_true")
    idl.add_argument("--cap", type=int, default=512)
    idl.add_argument("--trunc", type=float, default=1e-10)
    idl.add_argument("--threads", type=int, default=2)
    idl.add_argument("--cache", default=str(pathlib.Path.home() / ".cache" / "htq_hw"))
    idl.set_defaults(fn=cmd_ideal)
    ck = sub.add_parser("check")
    _add_campaign_args(ck)
    ck.add_argument("--families", nargs="+", default=["j0", "j1p1", "j1p2"])
    ck.add_argument("--tol", type=float, default=5e-3)
    ck.add_argument("--cap", type=int, default=512)
    ck.add_argument("--threads", type=int, default=2)
    ck.set_defaults(fn=cmd_check)
    rh = sub.add_parser("rehearse")
    _add_campaign_args(rh)
    rh.add_argument("--out", default="data/hw/rehearsal")
    rh.add_argument("--shots-scale", type=float, default=0.01)
    rh.add_argument("--shots", type=int, default=None, help="fixed shots per pub (overrides the plan)")
    rh.add_argument("--noise", type=float, nargs=2, metavar=("P2", "P1"), default=None)
    rh.add_argument("--seed", type=int, default=0)
    rh.add_argument("--n-traj", type=int, default=8, help="noise trajectories (batches) per pub")
    rh.add_argument("--cap", type=int, default=512)
    rh.add_argument("--threads", type=int, default=2)
    rh.set_defaults(fn=cmd_rehearse)
    sb = sub.add_parser("submit")
    _add_campaign_args(sb)
    sb.add_argument("--real", action="store_true", help="submit to the real backend named by --target")
    sb.add_argument("--confirm", default=None, metavar="BACKEND",
                    help="required with --real: the resolved backend name, typed out, so a real "
                         "submission cannot happen by accident")
    sb.add_argument("--fetch", action="store_true", help="local mode: fetch bits immediately")
    sb.add_argument("--shots-scale", type=float, default=1.0)
    sb.add_argument("--only-jobs", type=int, nargs="+", default=None)
    sb.add_argument("--guard", type=float, default=0.85)
    sb.add_argument("--tag", default="")
    sb.add_argument("--out", default="data/hw")
    sb.set_defaults(fn=cmd_submit)
    ft = sub.add_parser("fetch")
    ft.add_argument("meta", nargs="+", help="data/hw/htq_job_<id>.json files")
    ft.add_argument("--out", default="data/hw")
    ft.set_defaults(fn=cmd_fetch)
    an = sub.add_parser("analyze")
    an.add_argument("bits", nargs="+", help="data/hw/htq_bits_<id>.npz files (one per job)")
    an.add_argument("--ideal", default=None)
    an.add_argument("--slices", default="data/hw/slice_{comp}_t{t:.1f}.npz")
    an.add_argument("--times", type=float, nargs="+", default=None)
    an.add_argument("--components", nargs="+", default=["00", "10", "01", "11"])
    an.add_argument("--backend", default="")
    an.add_argument("--prefix", default=None,
                    help="'preset.card:' pub-name prefix selecting one card of a composed campaign")
    an.add_argument("--preset", default=None, help="with --card, derives --prefix")
    an.add_argument("--list-prefixes", action="store_true",
                    help="list the prefixes present in the bits files and exit")
    an.add_argument("--wing-surrogate", default=None, metavar="NPZ",
                    help="wing-anchor target for slices whose ideal grid stops short "
                         "(scripts/wing_surrogate.py build)")
    an.set_defaults(fn=cmd_analyze)
    bd = sub.add_parser("bundle")
    bd.add_argument("--out", default=None)
    bd.add_argument("--report-json", default=None)
    bd.set_defaults(fn=cmd_bundle)
    args = ap.parse_args(argv)
    if getattr(args, "cache", None) == "":
        args.cache = None
    if getattr(args, "weighting", None) is None and hasattr(args, "preset"):
        args.weighting = "equal" if args.preset else "kappa"
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
