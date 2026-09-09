"""One command that says whether this package may be shipped or run.

Every gap the pre-handoff audit found was something no single command
checked: cards without grids, grids too short for the slices they serve, a
default grid path that pointed at files that do not exist, an embedding that
silently degraded, a "real" submission that was not real.  Each of those is a
step below, so the failure mode becomes a red line in a record rather than a
surprise on a paid device.

  python -m htq_hw acceptance --preset relA-core prod-bridge vac-w00 qpdf-scan \
      --target fake:nighthawk --level fast --record data/hw/acceptance.json

Levels: ``fast`` runs A0-A5 (environment, target, embedding, card/grid
coverage, circuits, plan) in minutes; ``full`` adds A6 check and A7
rehearse+analyze per card, and is what bundle requires.
"""

import time

import numpy as np

from . import campaign as CP
from . import circuits as C
from . import record as R
from . import target as T
from .model import Lattice

STEPS_FAST = ("env", "target", "embedding", "cards", "circuits", "plan")
STEPS_FULL = STEPS_FAST + ("check", "rehearse")


class Result:
    def __init__(self):
        self.steps, self.failures, self.warnings, self.extra = [], [], [], {}

    def add(self, name, status, detail="", **kw):
        self.steps.append({"name": name, "status": status, "detail": detail, **kw})
        if status == R.FAIL:
            self.failures.append(f"{name}: {detail}")
        elif status == R.WARN:
            self.warnings.append(f"{name}: {detail}")
        return status

    @property
    def status(self):
        return R.FAIL if self.failures else (R.WARN if self.warnings else R.PASS)


def _families_for(specs, cname: str) -> set:
    """Families a card actually needs a grid for: those its specs measure,
    plus j0 whenever anything is mirror-calibrated against it."""
    fams = {s.family for s in specs if (s.card or "") == cname and s.family != "qpdf"}
    if any(s.mirror for s in specs if (s.card or "") == cname):
        fams.add("j0")
    return fams


def _surrogate_for(cname: str, template, cards=None):
    """The wing surrogate serving a card, by coupling tag ('prod' / 'relA').
    -> (surrogate or None, reason) -- a mismatched coupling is a reason, not a
    silent None, because it would bias every anchored slice."""
    from . import analyze as A
    from . import circuits as C
    path = A.wing_path_for(template, cname)
    if not path:
        return None, "no --wing-surrogate given"
    try:
        eta = (cards or {}).get(cname, C.load_card(cname))["couplings"]["eta"]
    except Exception:
        eta = None
    try:
        return A.load_wing_surrogate(path, eta=eta), ""
    except FileNotFoundError:
        return None, f"{path} does not exist"
    except ValueError as e:
        return None, str(e).split(". ")[0]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _sur_has(sur, t: float) -> bool:
    ts = sur["times"]
    return bool(len(ts)) and abs(ts[int(np.argmin(np.abs(ts - t)))] - t) <= 1e-6


def _step_qpdf(res, bits_path, lat, center, refs, shots, log):
    """The width scan reduced end to end: bits -> connected h(m) -> <x>, each
    card against its ideal.  Sampling those pubs and never checking what they
    reduce to is how the preset stayed unanalysable for as long as it did."""
    from . import analyze as A
    from . import circuits as C
    by = A.qpdf_bits_by_card([bits_path])
    if not by:
        return
    ref = None
    if refs:
        try:
            ref = np.load(refs, allow_pickle=True)
        except Exception:
            ref = None
    rows, bad = {}, []
    for card in sorted(by):
        vac = A.qpdf_vacuum_card(card, by)
        if vac is None:
            continue                                   # a vacuum card is the subtrahend
        try:
            k0 = C.load_card(card)["block"]["k0"]
            r = A.qpdf_reduce(by[card], lat, center, k0, vac_bits_by_setting=by[vac], seed=1)
        except Exception as e:
            bad.append(f"{card}: {type(e).__name__}: {e}")
            continue
        row = {"x": r["x"], "x_err": r["x_err"], "norm": r["norm"], "vacuum": vac,
               "h0_measured": r["h0_measured"]}
        key = _qpdf_ref_key(card)
        if ref is not None and f"{key}_x" in ref.files:
            xr = float(ref[f"{key}_x"])
            row["x_ideal"], row["pull"] = xr, (r["x"] - xr) / max(r["x_err"], 1e-12)
            if abs(row["pull"]) > 5.0:
                bad.append(f"{card}: <x> {r['x']:.4f} vs ideal {xr:.4f} ({row['pull']:+.1f} sigma)")
        if not r["h0_measured"]:
            bad.append(f"{card}: no qZ pub, so h(0) is unmeasured")
        rows[card] = row
    if not rows:
        return
    pulls = [abs(v["pull"]) for v in rows.values() if "pull" in v]
    res.add("qpdf", R.FAIL if bad else R.PASS,
            f"{len(rows)} card(s) reduced at {shots} shots"
            + (f", worst |pull| {max(pulls):.1f} sigma vs the ideal <x>" if pulls else
               " (no reference file to compare against)")
            + ("; " + "; ".join(bad[:3]) if bad else ""),
            qpdf=rows)


def _qpdf_ref_key(card: str) -> str:
    tag = "relA" if card.startswith("relA") else "prod"
    for part in card.split("_"):
        if part.startswith("s") and part[1:].replace(".", "").isdigit():
            return f"{tag}_s{float(part[1:]):.2f}"
    return f"{tag}_vac"


def _rehearse_times(specs, per_card: int = 1) -> list:
    """t = 0 plus the shortest evolved slice each card has: enough to exercise
    base, physics, mirror and the whole calibration path without paying for
    the deep circuits, whose only new ingredient is Trotter depth."""
    out = {0.0}
    for c in {s.card or "" for s in specs}:
        ts = sorted({s.t for s in specs if (s.card or "") == c and s.t > 0})
        out.update(ts[:per_card])
    return sorted(out)


def run(target: str, presets, ns: int = 50, basis: str = "cz", level: str = "fast",
        mode: str = "auto", require_real: bool = False, cache_dir=None, threads: int = 2,
        shots_scale: float = 1.0, budget: float = 180.0, rep_time: float = 250e-6,
        tol: float = 5e-3, times=None, wing_surrogate: str | None = None,
        rehearse_shots: int = 4000, kappa_tol: float = 0.25,
        refs: str = "data/qpdf_card_refs.npz", log=print) -> dict:
    """-> the record dict (also the return value of the CLI)."""
    res, t0 = Result(), time.time()
    steps = STEPS_FULL if level == "full" else STEPS_FAST

    # ---- A0 environment -------------------------------------------------
    env = R.env_stamp()
    git = R.git_stamp()
    res.add("env", R.PASS if env["pinned_ok"] else R.FAIL,
            "pinned" if env["pinned_ok"] else "; ".join(env["pin_mismatches"]))
    if not git.get("commit"):
        # an unpacked bundle is not a checkout: provenance comes from the
        # record and MANIFEST.sha256 instead, which is the normal case for a
        # recipient and must not stop the gate
        res.add("git", R.PASS, "not a git checkout (provenance from the record and MANIFEST.sha256)")
    elif git.get("dirty"):
        res.add("git", R.WARN, f"uncommitted changes under htq_hw: {git['dirty_files'][:3]}")
    else:
        res.add("git", R.PASS, f"{git.get('tag') or ''} {git['commit'][:8]}".strip())

    # ---- A1 target ------------------------------------------------------
    be = T.resolve_backend(target, fractional=(basis == "rzz"), allow_standin=not require_real)
    tstamp = R.backend_stamp(be)
    if require_real:
        try:
            T.require_real_backend(be, target, fractional=(basis == "rzz"))
            res.add("target", R.PASS, f"{tstamp['label']} (live)")
        except T.BackendSafetyError as e:
            res.add("target", R.FAIL, str(e).replace("\n", " "))
    else:
        res.add("target", R.PASS if tstamp["is_real"] else R.WARN,
                f"{tstamp['label']}" + ("" if tstamp["is_real"] else " (offline stand-in: an "
                                        "embedding validated here is not the real device's)"))

    # ---- A2 embedding ---------------------------------------------------
    specs = CP.compose_presets(list(presets))
    cards = CP.load_cards(specs, None)
    from . import CENTER as _CENTER
    center = _CENTER if ns == 50 else ns // 2 - 1
    gauss_sites = None
    if any("gauss" in p for p in presets):
        from .gauss import gauss_ancilla_sites
        gauss_sites = gauss_ancilla_sites(Lattice(ns), center)
    lat = Lattice(ns)
    emb = None
    try:
        graph, greport = T.operational_graph(be, log=log)
        emb = T.choose_embedding(be, ns, center, mode=mode, log=log, gauss_sites=gauss_sites,
                                 graph=graph, allow_transpiler=False)
        red = emb.info.get("redundancy", 1)
        res.add("embedding", R.PASS if red >= 2 else R.WARN,
                f"{emb.kind}, redundancy {red}, spare {emb.info.get('spare_qubits')}, "
                f"{len(greport['dropped_qubits'])} dead qubit(s) excluded",
                embedding={"kind": emb.kind, "redundancy": red,
                           "spare_qubits": emb.info.get("spare_qubits"),
                           "layout": list(emb.layout) if emb.layout else None},
                operational=greport)
        res.extra["embedding"] = {"kind": emb.kind, "redundancy": red,
                                  "spare_qubits": emb.info.get("spare_qubits")}
    except T.EmbeddingError as e:
        res.add("embedding", R.FAIL, str(e).replace("\n", " "))

    # ---- A3 card and grid coverage -------------------------------------
    from . import analyze as A
    tpl = str(C.CARD_DIR / "{card}" / "ideal_{family}.npz")
    coverage = {}
    for cname in sorted(cards):
        need = _families_for(specs, cname)
        want_t = sorted({s.t for s in specs if (s.card or "") == cname})
        entry = {"families_needed": sorted(need), "times_needed": want_t, "grids": {}}
        if not need:
            entry["note"] = "preparation-only card (no evolved pubs)"
            coverage[cname] = entry
            res.add(f"cards:{cname}", R.PASS, "preparation-only")
            continue
        missing, short, covered = [], [], []
        sur, sur_why = _surrogate_for(cname, wing_surrogate, cards)
        for fam in sorted(need):
            path = A._fmt(tpl, fam, cname)
            try:
                g = A.IdealGrid(path, expect_ns=ns)
            except (FileNotFoundError, ValueError) as e:
                missing.append(f"{fam} ({type(e).__name__})")
                continue
            absent = [t for t in want_t if t > 0 and not g.has_row(t)]
            entry["grids"][fam] = {"times": [float(x) for x in g.times],
                                   "missing_rows": absent, "id_a": g.id_a, "c_a": g.c_a}
            if not g.has_row(0.0):
                missing.append(f"{fam} (no t=0 row: cannot calibrate)")
            elif absent:
                gap = [t for t in absent if sur is None or not _sur_has(sur, t)]
                entry["grids"][fam]["surrogate_covers"] = [t for t in absent if t not in gap]
                if gap:
                    short.append(f"{fam} missing {len(gap)} row(s) {gap[:4]}")
                else:
                    covered.append(fam)
        coverage[cname] = entry
        if missing:
            res.add(f"cards:{cname}", R.FAIL, "grids missing: " + "; ".join(missing))
        elif short:
            res.add(f"cards:{cname}", R.WARN,
                    "; ".join(short) + f" -- no wing target ({sur_why or 'not in the surrogate'}): "
                                      "the anchor degrades to wing_applied=False on those slices")
        elif covered:
            res.add(f"cards:{cname}", R.PASS,
                    f"{len(need)} family grid(s); {len(covered)} rely on the wing surrogate "
                    f"({sur['path'] if sur else '?'}) beyond t={max(float(x) for x in A.IdealGrid(A._fmt(tpl, covered[0], cname)).times)}")
        else:
            res.add(f"cards:{cname}", R.PASS, f"{len(need)} family grid(s) cover every slice")
    res.extra["cards"] = coverage

    # ---- A4 circuits, A5 plan ------------------------------------------
    pubs = info = None
    if emb is not None and "circuits" in steps:
        try:
            pubs, info, bundles = CP.build_pub_circuits(be, lat, None, emb, specs, basis,
                                                        cache_dir=cache_dir, log=log, strict=True)
            n2q = {n: info[n]["n2q"] for n in pubs}
            res.add("circuits", R.PASS,
                    f"{len(pubs)} pubs built, 2q {min(n2q.values())}-{max(n2q.values())}, "
                    f"skeletons asserted")
        except Exception as e:
            res.add("circuits", R.FAIL, f"{type(e).__name__}: {e}")
    if pubs is not None and "plan" in steps:
        try:
            n2q_map = {n: info[n]["n2q"] for n in pubs}
            plan = CP.shots_plan(specs, n2q_map, budget_minutes=budget, rep_time_s=rep_time,
                                 weighting="equal")
            jobs = CP.group_jobs(specs, 8)
            floor = 30000
            mirror_floor_ok = all(plan["shots"].get(s.name, 0) >= floor
                                  for s in specs if s.mirror)
            mins = plan.get("minutes", plan.get("committed_minutes"))
            if isinstance(mins, dict):          # keyed by rep time
                mins = mins.get(rep_time, next(iter(mins.values())))
            over = mins is not None and mins > budget * 1.001
            res.add("plan", R.WARN if (over or not mirror_floor_ok) else R.PASS,
                    f"{len(jobs)} jobs, {sum(plan['shots'].values()):.3g} shots, "
                    f"{mins:.1f} min at {rep_time*1e6:.0f} us (budget {budget:g})"
                    + ("; OVER BUDGET" if over else "")
                    + ("" if mirror_floor_ok else "; a mirror is below the shot floor"),
                    jobs=len(jobs), minutes=mins)
        except Exception as e:
            res.add("plan", R.FAIL, f"{type(e).__name__}: {e}")

    # ---- A6 check -------------------------------------------------------
    if "check" in steps and emb is not None:
        from . import sim as S
        try:
            ct = times if times is not None else [0.0, 0.5]
            r = S.check(be, lat, cards, emb, tpl, ct, basis=basis, tol=tol,
                        threads=threads, cache_dir=cache_dir, log=log, specs=specs)
            worst = max(r.values()) if r else 0.0
            res.add("check", R.PASS if worst <= tol else R.FAIL,
                    f"{len(r)} circuits, worst {worst:.2e} (tol {tol:g})", worst=worst)
        except Exception as e:
            res.add("check", R.FAIL, f"{type(e).__name__}: {str(e)[:200]}")

    # ---- A7 rehearse and analyze -----------------------------------------
    if "rehearse" in steps and emb is not None and pubs is not None:
        import shutil
        import tempfile
        from . import sim as S
        out_dir = tempfile.mkdtemp(prefix="htq_accept_")
        try:
            rt = list(times) if times is not None else _rehearse_times(specs)
            sub = [s for s in specs if any(abs(s.t - x) < 1e-9 for x in rt)]
            shots = {s.name: int(rehearse_shots) for s in sub}
            out = S.rehearse(be, lat, cards, emb, sub, shots, tpl, out_dir, basis=basis,
                             threads=threads, cache_dir=cache_dir, log=log,
                             wing_surrogate=wing_surrogate)
            worst, bad, per_card = 0.0, [], {}
            for key, path in out["slices"].items():
                cname = key[0] if len(key) == 3 else ""
                z = np.load(path, allow_pickle=True)
                k = float(z["kappa_v"][0][emb.center])
                ok = ~z["mask"][0]
                d = abs(k - 1.0)
                worst = max(worst, d)
                per_card.setdefault(cname, []).append({"slice": key[-2:], "kappa_center": k,
                                                       "masked": int((~ok).sum())})
                if not np.isfinite(k) or d > kappa_tol or not ok.any():
                    bad.append(f"{key}: kappa(center)={k:.3f}, {int((~ok).sum())}/{ns} masked")
            missing = [c for c in cards if c not in per_card and _families_for(specs, c)]
            res.add("rehearse", R.FAIL if (bad or missing) else R.PASS,
                    (f"{len(out['slices'])} slices from {len(sub)} pubs at t={rt} "
                     f"({rehearse_shots} shots), worst |kappa-1| {worst:.3f}")
                    + ("; no slices for " + ", ".join(missing) if missing else "")
                    + ("; " + "; ".join(bad[:3]) if bad else ""),
                    slices=len(out["slices"]), worst_kappa_dev=worst, rehearsal=per_card)
            _step_qpdf(res, out["bits"], lat, emb.center, refs, rehearse_shots, log)
        except Exception as e:
            res.add("rehearse", R.FAIL, f"{type(e).__name__}: {str(e)[:200]}")
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    payload = {"level": level, "target": tstamp, "presets": list(presets), "ns": ns,
               "basis": basis, "seconds": round(time.time() - t0, 1),
               "steps": res.steps, "failures": res.failures, "warnings": res.warnings,
               "cards": coverage, "files": R.file_hashes(C.CARD_DIR.parent),
               **res.extra}
    return {"status": res.status, **payload}
