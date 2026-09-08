"""Campaign manifest, shot plan, job grouping, submission and result fetch.

Default manifest: 12 slices t = 0.5 .. 6.0 (dt = 0.5) x 12 pubs
  physics  j0 x {Z, XYA, XYB}, j1p1 x {...}, j1p2 x {...}     (9)
  mirrors  j0 base + eps-angle block x {Z, XYA, XYB}          (3)
plus the t = 0 references once: j0 / j1p1 / j1p2 x 3 readouts (9 pubs).
Pub names ``{fam}_{t|m}{T:.1f}_{R}`` (``_Y`` suffix for ancilla-Y copies).

Options: j1_mirrors (j1p1 mirror x 3 per slice), dither (family j0d = J0
inserted at site CENTER+1), im (ancilla-Y copies of every pub), dt_half
(Trotter-systematic control: the SAME physical times ``dt_half_times``,
default t = 0.5 and 1.0, repeated with twice the steps, dt = 0.25, mirrors
included, pub names suffixed ``_dt0.25``; the analysis writes them to
separate ``slice_{comp}_t{T}_dt0.25.npz`` files for comparison at equal t).

Job pairing: every mirror shares a job with the physics pubs it calibrates
(same t, readout basis and ancilla basis); the t = 0 references go in one
job.  Submission uses SamplerV2 with gate twirling (off for fractional
gates), measurement twirling, XY4 dynamical decoupling and the budget guard
of scripts/ibm_hardware.py:287-306; the same code path runs in
qiskit-ibm-runtime local testing mode (SamplerV2(mode=FakeBoston())).
"""

import contextlib
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import CENTER, DT, ETA, MIRROR_EPS, __version__
from . import circuits as C
from . import target as T
from .model import Lattice

READOUTS = ("Z", "XYA", "XYB")
PHYSICS_FAMILIES = ("j0", "j1p1", "j1p2")
FAMILY_GADGET = {"j0": ("J0", 0), "j1p1": ("J1a", 0), "j1p2": ("J1b", 0), "j0d": ("J0", 1)}
DEFAULT_TIMES = tuple(round(0.5 * k, 2) for k in range(1, 13))
HALF_DT = 0.25
DT_HALF_TIMES = (0.5, 1.0)
# measured ibm_kingston ladder-run damping of the centre-site J0 signal
KINGSTON_KAPPA = {0.5: 0.65, 1.0: 0.65, 1.5: 0.60, 2.0: 0.60, 3.0: 0.52}
REP_TIME_HERON = 250e-6
REP_TIME_NIGHTHAWK = 4e-3


@dataclass(frozen=True)
class PubSpec:
    name: str
    family: str
    t: float
    mirror: bool
    readout: str
    anc_basis: str
    n_steps: int
    dt: float
    group: str
    card: str = ""
    preset: str = ""
    stretch: bool = False

    @property
    def prefix(self) -> str:
        return name_prefix(self.preset, self.card)


def name_prefix(preset: str, card: str) -> str:
    """Pub-name prefix ``preset.card:`` used when presets are composed
    (several cards in one campaign); empty for the plain single-card manifest."""
    return f"{preset}.{card}:" if preset else ""


def dt_suffix(dt: float) -> str:
    return "" if abs(dt - DT) < 1e-9 else f"_dt{dt:g}"


def pub_name(family: str, t: float, mirror: bool, readout: str, anc_basis: str = "X",
             dt: float = DT, prefix: str = "") -> str:
    return (prefix + f"{family}_{'m' if mirror else 't'}{t:.1f}_{readout}" + ("_Y" if anc_basis == "Y" else "")
            + dt_suffix(dt))


def split_prefix(name: str) -> tuple[str, str]:
    """('preset.card:' or '', bare name)."""
    i = name.rfind(":")
    return (name[:i + 1], name[i + 1:]) if i >= 0 else ("", name)


def parse_pub_name(name: str) -> dict:
    """{'family', 't', 'mirror', 'readout', 'anc_basis', 'dt', 'prefix'} from a pub name."""
    prefix, name = split_prefix(name)
    parts = name.split("_")
    out = {"family": parts[0], "t": float(parts[1][1:]), "mirror": parts[1][0] == "m",
           "readout": parts[2], "anc_basis": "X", "dt": DT, "prefix": prefix}
    for extra in parts[3:]:
        if extra == "Y":
            out["anc_basis"] = "Y"
        elif extra.startswith("dt"):
            out["dt"] = float(extra[2:])
    return out


def _spec(family, t, mirror, readout, anc, n_steps, dt, card="", preset="", stretch=False):
    prefix = name_prefix(preset, card)
    group = prefix + ("refs" if t == 0 else f"t{t:.2f}_{readout}_{anc}{dt_suffix(dt)}" + ("_stretch" if stretch else ""))
    return PubSpec(pub_name(family, t, mirror, readout, anc, dt, prefix), family, float(t), mirror, readout,
                   anc, n_steps, dt, group, card, preset, stretch)


def manifest(times=DEFAULT_TIMES, j1_mirrors: bool = False, dither: bool = False,
             im: bool = False, dt_half: bool = False, refs: bool = True,
             dt_half_times=DT_HALF_TIMES, families=PHYSICS_FAMILIES, readouts=READOUTS,
             mirror_readouts=None, dither_refs=None, stretch_times=(), card: str = "",
             preset: str = "") -> list[PubSpec]:
    """Ordered pub specs (see module docstring).  Default: 12 x 12 + 9 = 153.
    ``families``/``readouts`` restrict the physics pubs (e.g. j0 x Z only);
    ``stretch_times`` adds slices flagged stretch (their own groups, ordered
    last); ``dither_refs`` (default = ``dither``) controls j0d t = 0 refs."""
    fams = list(families) + (["j0d"] if dither else [])
    ref_fams = list(families) + (["j0d"] if (dither if dither_refs is None else dither_refs) else [])
    mirror_fams = ["j0"] + (["j1p1"] if j1_mirrors else [])
    mirror_readouts = list(readouts if mirror_readouts is None else mirror_readouts)
    ancs = ["X"] + (["Y"] if im else [])
    slices = [(float(t), int(round(t / DT)), DT, False) for t in times]
    if dt_half:
        slices += [(float(t), int(round(t / HALF_DT)), HALF_DT, False) for t in dt_half_times]
    slices.sort()
    slices += sorted((float(t), int(round(t / DT)), DT, True) for t in stretch_times)
    specs = []
    if refs:
        for anc in ancs:
            for fam in ref_fams:
                for r in readouts:
                    specs.append(_spec(fam, 0.0, False, r, anc, 0, DT, card, preset))
    for t, n, dt, stretch in slices:
        for anc in ancs:
            for fam in fams:
                for r in readouts:
                    specs.append(_spec(fam, t, False, r, anc, n, dt, card, preset, stretch))
            for fam in mirror_fams:
                for r in mirror_readouts:
                    specs.append(_spec(fam, t, True, r, anc, n, dt, card, preset, stretch))
    return specs


# ------------------------------------------------------------------ presets
WIDTH_CARDS = ("prod_k1.26_s0.75_ns50", "prod_k1.26_s1.00_ns50", "prod_k1.26_s1.50_ns50",
               "relA_k1.26_s0.75_ns50", "relA_k1.26_s1.00_ns50", "relA_k1.26_s1.50_ns50")
VAC_CARDS = {"prod": "prod_vac_ns50", "relA": "relA_vac_ns50"}
QPDF_MS = (1, 2, 3, 4, 5)          # z = +-2m, |z| <= 10
STRETCH_TIMES = (6.5, 7.0, 7.5, 8.0)
PRESET_NAMES = ("relA-core", "prod-bridge", "vac-w00", "qpdf-scan", "gauss-midcircuit")


def qpdf_specs(card: str, preset: str = "qpdf-scan", ms=QPDF_MS) -> list[PubSpec]:
    """Prep-only pubs: setting qXXm2 etc. = Pauli at the centre / at the far
    ends, Z in between (circuits.qpdf_basis_map); 4 kinds x len(ms) pubs."""
    prefix = name_prefix(preset, card)
    out = []
    for m in ms:
        for kind in ("XX", "YY", "XY", "YX"):
            setting = f"q{kind}m{m}"
            out.append(PubSpec(f"{prefix}qpdf_t0.0_{setting}", "qpdf", 0.0, False, setting, "X", 0, DT,
                               prefix + f"qpdf_m{m}", card, preset, False))
    return out


def preset_specs(name: str) -> list[PubSpec]:
    """The manifest of one preset (composable via compose_presets)."""
    t12 = tuple(round(0.5 * k, 2) for k in range(1, 13))
    if name == "relA-core":
        return manifest(times=t12, j1_mirrors=True, dither=True, dither_refs=False, dt_half=True,
                        dt_half_times=(0.5, 1.0), stretch_times=STRETCH_TIMES,
                        card="relA_k1.26_s0.75_ns50", preset=name)
    if name == "prod-bridge":
        return manifest(times=tuple(round(0.5 * k, 2) for k in range(1, 17)), families=("j0",), dither=True,
                        dither_refs=True, card="prod_k1.26_s0.75_ns50", preset=name)
    if name == "vac-w00":
        out = []
        for tag, card in VAC_CARDS.items():
            out += manifest(times=t12, families=("j0",), readouts=("Z",), card=card, preset=name)
        return out
    if name == "qpdf-scan":
        out = []
        for card in WIDTH_CARDS + tuple(VAC_CARDS.values()):
            out += qpdf_specs(card, name)
        return out
    if name == "gauss-midcircuit":
        from .gauss import gauss_specs
        return gauss_specs(name)
    raise ValueError(f"unknown preset {name!r}; known {PRESET_NAMES}")


def compose_presets(names) -> list[PubSpec]:
    """Concatenate presets; every stretch pub goes to the end."""
    specs = []
    for n in names:
        specs += preset_specs(n)
    return [s for s in specs if not s.stretch] + [s for s in specs if s.stretch]


def preset_table(specs, shots: dict | None = None, rep_time_s: float = REP_TIME_HERON) -> str:
    """Per-preset pubs / shots / minutes plus the committed / stretch split."""
    rows, lines = {}, []
    for s in specs:
        key = (s.preset or "manifest", s.card, s.stretch)
        r = rows.setdefault(key, {"pubs": 0, "shots": 0})
        r["pubs"] += 1
        r["shots"] += shots[s.name] if shots else 0
    lines.append(f"{'preset':18} {'card':24} {'part':9} {'pubs':>5} {'shots':>10} {'min@rep':>8}")
    tot_c = tot_s = 0.0
    for (pr, card, st), r in sorted(rows.items()):
        mins = r["shots"] * rep_time_s / 60
        if st:
            tot_s += mins
        else:
            tot_c += mins
        lines.append(f"{pr:18} {card:24} {'stretch' if st else 'committed':9} {r['pubs']:5d} {r['shots']:10d} {mins:8.1f}")
    lines.append(f"committed {tot_c:.1f} min, stretch {tot_s:.1f} min (at {rep_time_s * 1e6:.0f} us per shot)")
    return "\n".join(lines)


def manifest_summary(specs) -> str:
    ts = sorted({s.t for s in specs})
    n_phys = sum(1 for s in specs if not s.mirror and s.t > 0)
    n_mir = sum(1 for s in specs if s.mirror)
    n_ref = sum(1 for s in specs if s.t == 0)
    n_ctrl = sum(1 for s in specs if abs(s.dt - DT) > 1e-9)
    n_str = sum(1 for s in specs if s.stretch)
    return (f"{len(specs)} pubs: {n_ref} t=0 references, {n_phys} physics + {n_mir} mirrors over "
            f"{len({(s.t, s.dt, s.card) for s in specs if s.t > 0})} slices t = {ts[1] if len(ts) > 1 else 0}..{ts[-1]}"
            + (f" (incl. {n_ctrl} dt=0.25 control pubs)" if n_ctrl else "")
            + (f" ({n_str} stretch pubs)" if n_str else ""))


# ------------------------------------------------------------------ circuits
def _skel_hash(qc) -> str:
    sk = T.skeleton(qc)
    return hashlib.sha256(repr(sorted(sk.items())).encode()).hexdigest()[:16]


def load_cards(specs, card: dict | None = None) -> dict:
    """{card name: card dict} for every card referenced by the specs
    (``card`` is the fallback for specs without a card)."""
    out = {}
    for s in specs:
        key = s.card or (card["name"] if card else "")
        if key not in out:
            out[key] = C.load_card(key) if s.card else card
    return out


def build_pub_circuits(be, lat: Lattice, card, emb: T.Embedding, specs, basis: str = "cz",
                       seed: int = T.SEED, cache_dir=None, log=None, strict: bool = True):
    """ISA pub circuits with readout layers.  ``card`` is the default card
    (dict) or None; specs carrying their own card name are built from that
    card (composed presets).  -> (pubs {name: circuit}, info {name: dict},
    bundles {(card, family): Bundle}).  Within a card all families must
    share the block layout so one j0 mirror calibrates every physics pub
    of its group (asserted when ``strict``)."""
    cards = load_cards(specs, card)
    pubs, info, bundles = {}, {}, {}
    for cname, cdict in cards.items():
        cspecs = [s for s in specs if (s.card or (card["name"] if card else "")) == cname]
        _build_card_pubs(be, lat, cdict, emb, cspecs, basis, seed, cache_dir, log, strict, pubs, info, bundles)
    return pubs, info, bundles


def _build_card_pubs(be, lat, card, emb, specs, basis, seed, cache_dir, log, strict, pubs, info, bundles):
    center = card["center"] if lat.ns == card["ns"] else emb.center
    fams = sorted({s.family for s in specs if s.family != "qpdf" and s.preset != "gauss-midcircuit"})
    steps = {}
    for s in specs:
        if s.n_steps > 0 and s.preset != "gauss-midcircuit":
            steps.setdefault(s.n_steps, set()).add(s.dt)
    cb = {}
    for fam in fams:
        kind, off = FAMILY_GADGET[fam]
        cb[fam] = T.transpile_bundle(be, lat, card, emb, tuple(sorted(steps)), basis, kind, seed, cache_dir,
                                     log=log, gadget_center=emb.center + off if off else None)
        bundles[(card["name"], fam)] = cb[fam]
    if any(s.family == "qpdf" for s in specs):
        base = C.prep_only_circuit(lat, card, emb.center)
        isa, fl, _ = T.transpile_isa(base, be, emb.initial_layout, 3, seed, cache_dir, log,
                                     f"prep-only[{card['name']},{basis}]")
        bundles[(card["name"], "qpdf")] = (isa, fl)
    ref = cb.get("j0", next(iter(cb.values()), None))
    for fam, b in cb.items():
        if b.base_layout != ref.base_layout:
            msg = f"{card['name']} family {fam}: base final layout differs from j0 (mirror would not share its skeleton)"
            if strict:
                raise AssertionError(msg)
            T._log("WARNING " + msg, log)
    blocks = {}
    for n, dts in steps.items():
        for dt in dts:
            phys, mir = T.assign_block(ref, n, n * dt)
            blocks[(n, dt)] = (phys, mir, ref.blocks[n]["layout"])
    if any(s.preset == "gauss-midcircuit" for s in specs):
        from .gauss import gauss_circuit, patch_sites, n_rounds, gauss_ancilla_sites
        sites = patch_sites(lat, emb.center)
        if "gauss_ancillas" not in emb.info:
            raise ValueError("gauss-midcircuit needs an embedding with gauss ancillas (choose_embedding(gauss_sites=...))")
        gskel = {}
        for s in [x for x in specs if x.preset == "gauss-midcircuit"]:
            logical = gauss_circuit(lat, card, s.n_steps, s.t, emb.center, mirror=s.mirror, sites=sites)
            isa, fl, _ = T.transpile_isa(logical, be, emb.initial_layout, 1, seed, cache_dir, log,
                                         f"gauss[{s.name}]", routing="none")
            qc = C.readout_layer(isa, lat, fl, s.readout, s.anc_basis)
            pubs[s.name] = qc
            key = (s.t, s.dt, s.readout, s.anc_basis)
            gskel.setdefault(key, {})[s.mirror] = qc
            info[s.name] = {"card": card["name"], "preset": s.preset, "stretch": s.stretch, "family": s.family,
                            "t": s.t, "mirror": s.mirror, "readout": s.readout, "anc_basis": s.anc_basis,
                            "n_steps": s.n_steps, "dt": s.dt, "group": s.group, "n2q": T.count_2q(qc),
                            "depth2q": T.depth_2q(qc), "layout": list(fl), "skeleton_block": _skel_hash(qc),
                            "skeleton_full": _skel_hash(qc), "gauss_sites": sites,
                            "gauss_rounds": n_rounds(s.n_steps) if s.n_steps else 1,
                            "syndrome_clbits": list(range(lat.n_wires, qc.num_clbits)),
                            "n_clbits": int(qc.num_clbits),
                            "basis_map": {str(k): v for k, v in C.basis_map(lat, s.readout).items()} | {str(lat.ancilla): s.anc_basis},
                            "clbit_to_logical": list(range(lat.n_wires))}
        for key, pair in gskel.items():
            if True in pair and False in pair:
                T.assert_skeleton_equal(pair[False], pair[True])
        specs = [x for x in specs if x.preset != "gauss-midcircuit"]
    for s in specs:
        if s.family == "qpdf":
            isa, fl = bundles[(card["name"], "qpdf")]
            m, kind = int(s.readout.split("m")[1]), s.readout[1:3]
            bmap = C.qpdf_basis_map(lat, emb.center, m, kind)
            qc = C.readout_layer(isa, lat, fl, bmap, "Z")
            body_hash = ""
            layout = fl
        else:
            b = cb["j0"] if s.mirror else cb[s.family]
            if s.n_steps == 0:
                body, layout, body_hash = b.base, b.base_layout, ""
            else:
                phys, mir, layout = blocks[(s.n_steps, s.dt)]
                blk = mir if s.mirror else phys
                body, body_hash = b.base.compose(blk), _skel_hash(blk)
            qc = C.readout_layer(body, lat, layout, s.readout, s.anc_basis)
        pubs[s.name] = qc
        bm = C.basis_map(lat, s.readout) if s.family != "qpdf" else C.qpdf_basis_map(lat, emb.center, int(s.readout.split("m")[1]), s.readout[1:3])
        info[s.name] = {"card": card["name"], "preset": s.preset, "stretch": s.stretch, "family": s.family,
                        "t": s.t, "mirror": s.mirror, "readout": s.readout, "anc_basis": s.anc_basis,
                        "n_steps": s.n_steps, "dt": s.dt, "group": s.group, "n2q": T.count_2q(qc),
                        "depth2q": T.depth_2q(qc), "layout": list(layout), "skeleton_block": body_hash,
                        "skeleton_full": _skel_hash(qc), "n_clbits": int(qc.num_clbits),
                        "basis_map": {str(k): v for k, v in bm.items()} | {str(lat.ancilla): s.anc_basis},
                        "clbit_to_logical": list(range(lat.n_wires))}


# ------------------------------------------------------------------ shots
def default_kappa_model(n2q_by_t: dict, measured: dict = KINGSTON_KAPPA):
    """kappa(n2q) = exp(a + b n2q) fitted to the measured kappa_c at the 2q
    counts this target's physics pubs have at the measured times (linear
    n2q(t) fit if a time is missing)."""
    ts = np.array(sorted(n2q_by_t))
    ns2 = np.array([n2q_by_t[t] for t in ts], dtype=float)
    if len(ts) >= 2:
        slope, icpt = np.polyfit(ts, ns2, 1)
    else:
        slope, icpt = 0.0, ns2[0]
    x = np.array([n2q_by_t.get(t, icpt + slope * t) for t in measured])
    y = np.log(np.array(list(measured.values())))
    b, a = np.polyfit(x, y, 1)
    return lambda n2q: float(np.exp(a + b * n2q)), (float(a), float(b))


def shots_plan(specs, n2q: dict, budget_minutes: float = 36.0, rep_time_s: float = REP_TIME_HERON,
               kappa_model=None, mirror_floor: int = 30000, round_to: int = 8,
               weighting: str = "kappa", shots_per_pub: int = 60000,
               shots_by_family: dict | None = None) -> dict:
    """Per-pub shots.  weighting 'kappa': proportional to 1/kappa(t)^2 with
    every mirror >= mirror_floor, normalised so the total at ``rep_time_s``
    equals the budget; 'equal': ``shots_per_pub`` for every pub (mirrors
    still >= mirror_floor), the budget then splits into committed / stretch
    / contingency minutes.
    -> {'shots': {name: int}, 'kappa': {name: float}, 'total': int,
        'minutes': {rep_time_s: ...}, 'rows': per-slice table rows, 'fit': (a, b),
        'split': {'committed', 'stretch', 'contingency'} minutes at rep_time_s}."""
    specs = list(specs)
    if weighting == "equal":
        sbf = {"qpdf": 20000} if shots_by_family is None else shots_by_family     # prep-only bilinears: O(1) signals
        shots = {s.name: int(max(sbf.get(s.family, shots_per_pub), mirror_floor if s.mirror else 0)) for s in specs}
        return _finish_plan(specs, n2q, shots, {s.name: float("nan") for s in specs}, budget_minutes,
                            rep_time_s, None, mirror_floor)
    n2q_by_t = {s.t: n2q[s.name] for s in specs if not s.mirror and s.family == "j0"
                and s.readout == "Z" and s.anc_basis == "X" and s.t > 0 and abs(s.dt - DT) < 1e-9}
    fit = None
    if kappa_model is None:
        kappa_model, fit = default_kappa_model(n2q_by_t)
    kap = {s.name: kappa_model(n2q[s.name]) for s in specs}
    w = {s.name: 1.0 / kap[s.name] ** 2 for s in specs}
    budget = budget_minutes * 60.0 / rep_time_s
    phys = [s for s in specs if not s.mirror]
    mirs = [s for s in specs if s.mirror]
    S = budget / sum(w.values())
    for _ in range(50):
        mshots = sum(max(mirror_floor, S * w[s.name]) for s in mirs)
        S_new = max(0.0, budget - mshots) / sum(w[s.name] for s in phys)
        if abs(S_new - S) < 1e-9 * max(S, 1.0):
            break
        S = S_new
    shots = {}
    for s in specs:
        v = S * w[s.name]
        if s.mirror:
            v = max(mirror_floor, v)
        shots[s.name] = int(round_to * max(1, round(v / round_to)))
    return _finish_plan(specs, n2q, shots, kap, budget_minutes, rep_time_s, fit, mirror_floor)


def _finish_plan(specs, n2q, shots, kap, budget_minutes, rep_time_s, fit, mirror_floor) -> dict:
    total = sum(shots.values())
    rows = []
    for t, dt, card in sorted({(s.t, s.dt, s.card) for s in specs}):
        ps = [s for s in specs if s.t == t and s.dt == dt and s.card == card and not s.mirror]
        ms = [s for s in specs if s.t == t and s.dt == dt and s.card == card and s.mirror]
        rows.append({"t": t, "dt": dt, "card": card, "n2q": n2q[ps[0].name], "kappa": kap[ps[0].name],
                     "n_phys": len(ps), "shots_phys": shots[ps[0].name],
                     "n_mirror": len(ms), "shots_mirror": shots[ms[0].name] if ms else 0,
                     "slice_total": sum(shots[s.name] for s in ps + ms), "stretch": ps[0].stretch})
    committed = sum(shots[s.name] for s in specs if not s.stretch) * rep_time_s / 60
    stretch = sum(shots[s.name] for s in specs if s.stretch) * rep_time_s / 60
    return {"shots": shots, "kappa": kap, "total": total, "budget_shots": int(budget_minutes * 60 / rep_time_s),
            "minutes": {REP_TIME_HERON: total * REP_TIME_HERON / 60,
                        REP_TIME_NIGHTHAWK: total * REP_TIME_NIGHTHAWK / 60,
                        rep_time_s: total * rep_time_s / 60},
            "rows": rows, "fit": fit, "mirror_floor": mirror_floor, "rep_time_s": rep_time_s,
            "split": {"committed": committed, "stretch": stretch,
                      "contingency": budget_minutes - committed - stretch}}


def format_plan(plan: dict) -> str:
    lines = [f"{'t':>5} {'dt':>5} {'card':22} {'2q':>6} {'kappa':>6} {'phys pubs':>9} {'shots/phys':>10} "
             f"{'mirrors':>7} {'shots/mir':>9} {'slice':>9}", "-" * 100]
    for r in plan["rows"]:
        lines.append(f"{r['t']:5.2f} {r['dt']:5.2f} {r['card'][:22]:22} {r['n2q']:6d} {r['kappa']:6.3f} {r['n_phys']:9d} "
                     f"{r['shots_phys']:10d} {r['n_mirror']:7d} {r['shots_mirror']:9d} {r['slice_total']:9d}"
                     + ("  STRETCH" if r.get("stretch") else ""))
    lines.append("-" * 100)
    sp = plan["split"]
    lines.append(f"split at {plan['rep_time_s'] * 1e6:.0f} us/shot: committed {sp['committed']:.1f} min, "
                 f"stretch {sp['stretch']:.1f} min, contingency {sp['contingency']:.1f} min")
    lines.append(f"total {plan['total']} shots (budget {plan['budget_shots']}): "
                 f"{plan['minutes'][REP_TIME_HERON]:.1f} min at 250 us, "
                 f"{plan['minutes'][REP_TIME_NIGHTHAWK] / 60:.2f} h at 4 ms (Nighthawk); "
                 f"mirror floor {plan['mirror_floor']}"
                 + (f"; kappa fit ln k = {plan['fit'][0]:.3f} + {plan['fit'][1]:.2e} n2q" if plan['fit'] else ""))
    return "\n".join(lines)


# ------------------------------------------------------------------ jobs
def group_jobs(specs, max_pubs: int = 8) -> list[list[str]]:
    """Pack calibration groups (physics pubs + their mirror(s), same t /
    readout / ancilla basis) into jobs of <= max_pubs without splitting a
    group; all t = 0 references form one job."""
    groups, order = {}, []
    for s in specs:
        if s.group not in groups:
            groups[s.group] = []
            order.append(s.group)
        groups[s.group].append(s.name)
    by = {s.name: s for s in specs}
    jobs = []
    for g in order:
        if g.endswith("refs"):
            jobs.append(list(groups[g]))
    for part in (False, True):                       # committed groups first, stretch groups last
        cur = []
        for g in order:
            if g.endswith("refs") or by[groups[g][0]].stretch != part:
                continue
            names = groups[g]
            if cur and (len(cur) + len(names) > max_pubs or by[cur[0]].prefix != by[names[0]].prefix):
                jobs.append(cur)
                cur = []
            cur = cur + names
        if cur:
            jobs.append(cur)
    return jobs


def _sampler(mode, fractional: bool, shots_default: int | None = None):
    from qiskit_ibm_runtime import SamplerV2

    smp = SamplerV2(mode=mode)
    smp.options.twirling.enable_gates = not fractional
    smp.options.twirling.enable_measure = True
    smp.options.dynamical_decoupling.enable = True
    smp.options.dynamical_decoupling.sequence_type = "XY4"
    if shots_default:
        smp.options.default_shots = shots_default
    return smp


GUARD_OK, GUARD_UNVERIFIABLE, GUARD_OVER = "ok", "unverifiable", "over"


class BudgetError(RuntimeError):
    """The allocation cannot be shown to cover what is about to be submitted."""


def remaining_seconds(service, log=None):
    """Allocation left, in QPU seconds, or None when it cannot be read."""
    if service is None:
        return None
    try:
        return float(service.usage()["usage_remaining_seconds"])
    except Exception as e:
        T._log(f"usage_remaining_seconds unavailable ({type(e).__name__}: {e})", log)
        return None


def preflight_budget(service, est_seconds: float, guard: float = 0.85, log=None):
    """Before job 0: the WHOLE campaign against the remaining allocation.
    The per-job guard cannot see this -- 112 jobs each individually inside the
    allocation still overrun it together.  -> (state, info)."""
    rem = remaining_seconds(service, log)
    info = {"estimated_s": float(est_seconds), "remaining_s": rem, "guard": guard}
    if rem is None:
        T._log(f"budget preflight: campaign ~{est_seconds:.0f} s, remaining UNKNOWN", log)
        return GUARD_UNVERIFIABLE, info
    T._log(f"budget preflight: campaign ~{est_seconds:.0f} s vs remaining {rem:.0f} s "
           f"({est_seconds / max(rem, 1e-9):.0%} of it)", log)
    return (GUARD_OVER if est_seconds > guard * rem else GUARD_OK), info


def budget_guard(job, service=None, guard: float = 0.85, log=None) -> str:
    """Per-job check against the remaining allocation
    (scripts/ibm_hardware.py:287-306).  Tri-state, because "the platform did
    not tell us" is not the same as "it fits": -> 'ok' | 'unverifiable' |
    'over'.  The caller decides what an unverifiable estimate means; only
    'over' cancels the job here."""
    rem = remaining_seconds(service, log)
    est_s = None
    try:
        ue = job.usage_estimation
        est_s = ue.get("quantum_seconds", None) if ue else None
    except Exception as e:
        T._log(f"usage_estimation unavailable ({type(e).__name__})", log)
    T._log(f"budget: estimated {est_s} s, remaining {rem} s", log)
    if est_s is None or rem is None:
        return GUARD_UNVERIFIABLE
    if est_s > guard * rem:
        job.cancel()
        T._log(f"CANCELLED {job.job_id()}: estimate {est_s} s exceeds {guard:.0%} of remaining {rem} s", log)
        return GUARD_OVER
    return GUARD_OK


def submit(mode, pubs: dict, info: dict, shots: dict, jobs: list[list[str]], lat: Lattice,
           emb: T.Embedding, basis: str, out_dir: str = "data/hw", service=None, guard: float = 0.85,
           tag: str = "", log=None, est_seconds: float | None = None, strict_guard: bool = True,
           batch: bool = True, job_offset: int = 0) -> list[dict]:
    """Submit one SamplerV2 job per name list; write data/hw/htq_job_<id>.json.
    ``mode`` is a real backend (from QiskitRuntimeService) or a fake backend /
    AerSimulator for local testing mode.

    Three things this owes a 3-hour allocation.  The whole campaign is
    pre-flighted against the remaining seconds before job 0, since 112 jobs
    that each fit can still overrun together.  An unverifiable estimate stops
    a real submission unless ``strict_guard=False``, rather than passing for
    the same reason a verified one does.  And a job the guard cancels stops
    the loop and still gets its metadata written, so the run neither spends
    the rest of the allocation nor loses the ids of what it did spend.

    -> [{'job': job, 'meta': dict, 'path': str}] for the jobs kept."""
    os.makedirs(out_dir, exist_ok=True)
    fractional = basis == "rzz"
    real = T.is_real_backend(mode)
    if real:
        if est_seconds is None:
            raise BudgetError("a real submission needs est_seconds to pre-flight the allocation")
        state, binfo = preflight_budget(service, est_seconds, guard, log)
        if state == GUARD_OVER:
            raise BudgetError(
                f"the campaign needs ~{binfo['estimated_s']:.0f} QPU s but only "
                f"{binfo['remaining_s']:.0f} s remain (guard {guard:.0%}); nothing was submitted. "
                f"Reduce it with --times / --shots-scale, or raise --guard deliberately.")
        if state == GUARD_UNVERIFIABLE and strict_guard:
            raise BudgetError(
                "the remaining allocation could not be read, so this submission cannot be shown to "
                "fit; nothing was submitted. Check the instance, or pass --no-strict-guard to "
                "submit without that assurance.")
    records, cancelled = [], None
    with _batch_ctx(mode, real and batch, log) as bctx:
        smp = _sampler(bctx if bctx is not None else mode, fractional)
        for k, names in enumerate(jobs):
            job = smp.run([(pubs[n], None, int(shots[n])) for n in names])
            jid = job.job_id()
            T._log(f"job {k + job_offset}: {jid} with {len(names)} pubs {names}", log)
            meta = {"job_id": jid, "backend": T.backend_label(mode) if hasattr(mode, "name") else str(mode),
                    "htq_hw": __version__, "tag": tag, "basis": basis, "ns": lat.ns, "center": emb.center,
                    "n_logical": lat.n_wires, "embedding": emb.kind, "initial_layout": emb.layout,
                    "pub_names": names, "shots": {n: int(shots[n]) for n in names},
                    "pubs": {n: info[n] for n in names}, "job_index": k + job_offset,
                    "options": {"twirling_gates": not fractional, "twirling_measure": True, "dd": "XY4"},
                    "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}
            path = os.path.join(out_dir, f"htq_job_{jid}.json")
            state = budget_guard(job, service, guard, log) if real else GUARD_OK
            stop = state == GUARD_OVER or (state == GUARD_UNVERIFIABLE and strict_guard and real)
            if stop and state == GUARD_UNVERIFIABLE:
                job.cancel()
            meta["guard"] = state
            meta["cancelled"] = bool(stop)
            with open(path, "w") as f:                      # written either way: never lose an id
                json.dump(meta, f, indent=1)
            if stop:
                cancelled = (k + job_offset, state)
                break
            records.append({"job": job, "meta": meta, "path": path})
    if cancelled is not None:
        k, state = cancelled
        left = len(jobs) - (k - job_offset) - 1
        T._log(f"STOPPED at job {k} ({state}); {left} job(s) not submitted. Resume with "
               f"--only-jobs {' '.join(str(k + job_offset + 1 + i) for i in range(min(left, 8)))}"
               + (" ..." if left > 8 else ""), log)
    return records


@contextlib.contextmanager
def _batch_ctx(mode, use_batch: bool, log=None):
    """One Batch around the whole campaign (jobs run back to back inside the
    allocation instead of re-queueing 112 times), and one sampler hoisted out
    of the loop.  Local testing mode gets neither: Batch needs a real backend."""
    if not use_batch:
        yield None
        return
    try:
        from qiskit_ibm_runtime import Batch
    except Exception as e:
        T._log(f"Batch unavailable ({type(e).__name__}); submitting job by job", log)
        yield None
        return
    with Batch(backend=mode) as b:
        T._log(f"Batch {getattr(b, 'session_id', '?')} open on {T.backend_label(mode)}", log)
        yield b


def fetch(job, meta: dict, out_dir: str = "data/hw", log=None) -> str:
    """Job result -> data/hw/htq_bits_<id>.npz: one uint8 array (shots,
    n_logical) per pub name, column i = logical qubit i (the readout layer
    measures physical layout[i] into clbit i; asserted from the metadata)."""
    if isinstance(job, str):
        from qiskit_ibm_runtime import QiskitRuntimeService
        job = QiskitRuntimeService().job(job)
    res = job.result()
    n_logical = int(meta["n_logical"])
    out, problems = {}, []
    for name, pub in zip(meta["pub_names"], res):
        d = pub.data
        if hasattr(d, "c"):
            ba = d.c
        else:                                    # a pub with its own register name
            keys = list(d.keys())
            if len(keys) != 1:
                problems.append((name, f"{len(keys)} classical registers: {keys}"))
                continue
            ba = d[keys[0]]
        arr = ba.to_bool_array(order="little").astype(np.uint8)
        # the gauss-midcircuit family carries syndrome clbits ABOVE the logical
        # register, so the width is n_logical only for the ordinary families
        want = int(meta["pubs"].get(name, {}).get("n_clbits", n_logical))
        if arr.shape[1] != want:
            problems.append((name, f"width {arr.shape[1]}, expected {want}"))
        cmap = meta["pubs"].get(name, {}).get("clbit_to_logical")
        if cmap is not None and list(cmap) != list(range(n_logical)):
            problems.append((name, "clbit_to_logical is not the identity"))
        out[name] = arr                          # keep it either way
    # write BEFORE validating: a shape surprise must never lose a paid result
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"htq_bits_{meta['job_id']}.npz")
    np.savez_compressed(path, job_id=meta["job_id"], backend=meta["backend"],
                        pub_names=np.array(meta["pub_names"]), **out)
    T._log(f"fetched {len(out)} pubs -> {path}", log)
    if problems:
        raise ValueError(f"{path} was written, but {len(problems)} pub(s) look wrong: "
                         + "; ".join(f"{n}: {w}" for n, w in problems))
    return path


def to_legacy_losch_bits(bits) -> dict:
    """Reversed-column arrays keyed t0.5 / m0.5 (j0 family, Z readout) so
    scripts/losch_t3_pool.py:80-92 reads them unchanged."""
    if isinstance(bits, str):
        z = np.load(bits, allow_pickle=True)
        bits = {k: z[k] for k in z.files}
    out = {}
    for name, arr in bits.items():
        parts = split_prefix(name)[1].split("_")
        if len(parts) == 3 and parts[0] == "j0" and parts[2] == "Z":     # dt = 0.5, ancilla X only
            out[parts[1]] = np.asarray(arr, dtype=np.uint8)[:, ::-1]
    return out


def sim_meta(lat, emb, basis, names, shots, info, backend="rehearsal", job_id=None) -> dict:
    """Metadata record for a locally sampled (rehearsal) job."""
    jid = job_id or f"rehearsal_{int(time.time())}"
    return {"job_id": jid, "backend": backend, "htq_hw": __version__, "basis": basis, "ns": lat.ns,
            "center": emb.center, "n_logical": lat.n_wires, "embedding": emb.kind,
            "initial_layout": emb.layout, "pub_names": list(names),
            "shots": {n: int(shots[n]) for n in names}, "pubs": {n: info[n] for n in names}}


def specs_from_names(names, specs) -> list:
    by = {s.name: s for s in specs}
    return [by[n] for n in names]


def spec_dicts(specs) -> list[dict]:
    return [asdict(s) for s in specs]
