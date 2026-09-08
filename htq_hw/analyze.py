"""Bitstring analysis: per-shot estimators, mirror calibration, four-component
correlator slices.

Estimators follow the reference implementation htensor/readout.py (not
imported) and the tier-3 calibration of scripts/ibm_hardware.py:346-380 and
scripts/losch_t3_pool.py:58-146:

  Z setting     : J0_v = (-1)^v/2 - z_v/2, xJ0_v = x_anc J0_v,
                  Gauss syndromes G_n = (-1)^n z_n x_{l-} x_{l+}
  XY settings   : T_b = s_a s_l s_b (x seam_sign on the seam bond); XYA reads
                  term 1 (Y_a X_b) on even bonds and term 2 (X_a Y_b) on odd
                  bonds, XYB the complement; <X J1_b> = (eta/4)(<X T1_b> - <X T2_b>)
  calibration   : per probe per readout basis from the same-job mirror,
                  kappa = sx_mirror / sx_ideal(0), guard |sx_ideal(0)| > 0.02
                  else median fill, mask kappa < 0.05; J0 probes also get
                  beta = (B_m - 0.5)/(B_i0 - 0.5), b_cal = 0.5 + (B_p - 0.5)/beta
                  (published +0.5 anchor at every site), wing anchor per parity
                  for the 00 component; chi2-inflated inverse-variance merge
                  across jobs.

Ideal grids per family: npz files written by scripts/hw_cal_grids.py (or
``python -m htq_hw ideal``): keys times, probes, bonds, id_a, c_a, insert_1pt,
one_pt_J0, one_pt_J1, XB_v/YB_v/B_v, XT1_b/YT1_b/T1_b, XT2_b/YT2_b/T2_b, X, Y.

Output per slice: data/hw/slice_{comp}_t{T:.1f}.npz with the tier-3 key set
(times, probes, C, C_cal, C_err, kappa_v, beta_v, b_cal, id_a, c_a, id_b,
tier, backend, nshot, merged_jobs) plus component and t; consumed by
scripts/hw_w_tensor.py.
"""

import json
import os

import numpy as np

from . import DT, ETA
from .campaign import parse_pub_name, pub_name, split_prefix
from .model import Lattice

KAP_GUARD = 0.02
KAP_MASK = 0.05
WING_MIN = 6
READOUTS = ("Z", "XYA", "XYB")
COMPONENTS = ("00", "10", "01", "11")
# component -> (insertion families, probe kind)
COMPONENT_LAYOUT = {"00": (("j0",), "J0"), "10": (("j0",), "J1"),
                    "01": (("j1p1", "j1p2"), "J0"), "11": (("j1p1", "j1p2"), "J1"),
                    # dither control: the SAME 00 observable with the insertion moved to
                    # site CENTER+1.  Its id_a, c_a, A0 and wing-anchor target come from
                    # the j0d ideal grid (id_a = -1/2 at the odd site, insert_1pt of its
                    # own insertion); the damping still comes from the j0 mirror it shares
                    # a job with, so kappa keeps using the j0 grid's t=0 row.  Not in
                    # COMPONENTS: request it with --components 00d.
                    "00d": (("j0d",), "J0"), "10d": (("j0d",), "J1")}
FAMILY_COEFF = {"j0": -0.5, "j0d": -0.5, "j1p1": +0.25, "j1p2": -0.25}   # c_a (x eta for j1)


# ------------------------------------------------------------------ per-shot estimators
def signs_from_bits(bits) -> np.ndarray:
    """0/1 (shots, qubits), column q = logical qubit q -> +-1 (htensor/readout.py:68-70)."""
    return 1.0 - 2.0 * np.asarray(bits, dtype=np.float64)


def j1_term_in_setting(bond: int, setting: str) -> int:
    """1 = Y_a X_b (coeff +eta/4), 2 = X_a Y_b (coeff -eta/4) (htensor/readout.py:57-65)."""
    a_even = bond % 2 == 0
    if setting == "XYA":
        return 1 if a_even else 2
    if setting == "XYB":
        return 2 if a_even else 1
    raise ValueError(setting)


def estimate_probes(bits, lat: Lattice, setting: str, eta: float = ETA) -> dict:
    """Per-shot estimators from one readout setting (htensor/readout.py:73-104).

    Z:  J0, J0_err, xJ0, xJ0_err (ns,), id_b, gauss (ns,) = <G_n>, gauss_err
    XY: T, T_err, xT, xT_err (ns,) for the term read on each bond, term (ns,)
    both: xa = <x_anc>, xa_err, N, and sx/sxe = the connected ancilla signal
          sx = xJ0 - id_b <x_anc> (J0) or xT (J1), with per-shot errors."""
    s = signs_from_bits(bits)
    anc = lat.ancilla
    xa = s[:, anc]
    N = len(s)
    out = {"xa": float(xa.mean()), "xa_err": float(xa.std() / np.sqrt(N)), "N": N,
           "setting": setting}
    if setting == "Z":
        z = s[:, [lat.site_qubit(v) for v in range(lat.ns)]]
        idb = np.array([(-1) ** v / 2 for v in range(lat.ns)])
        J0 = idb[None, :] - z / 2
        xJ0 = xa[:, None] * J0
        conn = xJ0 - idb[None, :] * xa[:, None]
        x_links = s[:, [lat.link_qubit(n) for n in range(lat.ns)]]
        G = np.array([(-1) ** n * z[:, n] * x_links[:, (n - 1) % lat.ns] * x_links[:, n]
                      for n in range(lat.ns)]).T
        out.update(J0=J0.mean(0), J0_err=J0.std(0) / np.sqrt(N),
                   xJ0=xJ0.mean(0), xJ0_err=xJ0.std(0) / np.sqrt(N), id_b=idb,
                   sx=conn.mean(0), sxe=conn.std(0) / np.sqrt(N),
                   B=J0.mean(0), Be=J0.std(0) / np.sqrt(N),
                   gauss=G.mean(0), gauss_err=G.std(0) / np.sqrt(N))
        return out
    T = np.empty((N, lat.ns))
    term = np.empty(lat.ns, int)
    for b in range(lat.ns):
        a, bb, l = lat.bond_qubits(b)
        sign = lat.seam_sign if lat.is_seam(b) else 1
        T[:, b] = sign * s[:, a] * s[:, l] * s[:, bb]
        term[b] = j1_term_in_setting(b, setting)
    xT = xa[:, None] * T
    out.update(T=T.mean(0), T_err=T.std(0) / np.sqrt(N),
               xT=xT.mean(0), xT_err=xT.std(0) / np.sqrt(N), term=term,
               sx=xT.mean(0), sxe=xT.std(0) / np.sqrt(N))
    return out


def combine_j1(resA: dict, resB: dict, eta: float = ETA):
    """<X J1_b> = (eta/4)(<X T1_b> - <X T2_b>) from the two XY settings
    (htensor/readout.py:107-116) -> (value, err)."""
    ns = len(resA["term"])
    val, err = np.empty(ns), np.empty(ns)
    for b in range(ns):
        r1, r2 = (resA, resB) if resA["term"][b] == 1 else (resB, resA)
        val[b] = eta / 4 * (r1["xT"][b] - r2["xT"][b])
        err[b] = eta / 4 * np.hypot(r1["xT_err"][b], r2["xT_err"][b])
    return val, err


def term_split(resA: dict, resB: dict) -> tuple[dict, dict]:
    """Regroup two XY-setting results by TERM: -> (term-1 result, term-2
    result), each with xT, xTe, T, Te (ns,) ordered by bond."""
    ns = len(resA["term"])
    out = []
    for k in (1, 2):
        d = {key: np.empty(ns) for key in ("xT", "xTe", "T", "Te")}
        for b in range(ns):
            r = resA if resA["term"][b] == k else resB
            d["xT"][b], d["xTe"][b] = r["xT"][b], r["xT_err"][b]
            d["T"][b], d["Te"][b] = r["T"][b], r["T_err"][b]
        d["N"] = min(resA["N"], resB["N"])
        out.append(d)
    return out[0], out[1]


# ------------------------------------------------------------------ ideal grids
class IdealGrid:
    """Wrapper around an hw_cal_grids-format npz."""

    def __init__(self, path: str, expect_ns: int | None = None):
        self.path = path
        self.z = np.load(path, allow_pickle=True)
        self.times = np.asarray(self.z["times"], dtype=float)
        self.ns = len(self.z["probes"])
        if expect_ns is not None and self.ns != expect_ns:
            # comparing against a grid of another size silently produces
            # order-one "deviations" that look like a physics failure
            raise ValueError(f"{path}: grid is Ns={self.ns} but Ns={expect_ns} was requested; "
                             f"use the card whose grids match, or --ns {self.ns}")
        self.id_a = float(self.z["id_a"])
        self.c_a = float(self.z["c_a"])
        self.A0 = float(np.real(self.z["insert_1pt"]))
        self.family = str(self.z["family"]) if "family" in self.z.files else None

    def row(self, t: float) -> int:
        i = int(np.argmin(np.abs(self.times - t)))
        if abs(self.times[i] - t) > 1e-6:
            raise KeyError(f"{self.path}: no ideal row at t={t} (have {self.times})")
        return i

    def has_row(self, t: float) -> bool:
        """True when an ideal row exists at t.  Calibration only ever divides
        by the t=0 row; deeper rows are the wing-anchor target, so a short grid
        degrades the anchor (wing_applied=False) rather than blocking a slice."""
        return bool(len(self.times)) and abs(self.times[int(np.argmin(np.abs(self.times - t)))] - t) <= 1e-6

    def has(self, key: str) -> bool:
        return key in self.z.files

    def vec(self, tag: str, t: float) -> np.ndarray:
        """(ns,) of key f'{tag}_{v}' at time t, e.g. tag 'XB', 'B', 'XT1', 'T2'."""
        i = self.row(t)
        return np.array([float(self.z[f"{tag}_{v}"][i]) for v in range(self.ns)])

    def scalar(self, key: str, t: float) -> float:
        return float(self.z[key][self.row(t)])

    def id_b(self) -> np.ndarray:
        return np.array([(-1) ** v / 2 for v in range(self.ns)])

    def sx_ideal0(self, probe: str, term: int | None = None) -> np.ndarray:
        """Connected ancilla signal at t=0: XB_v - id_b X (J0) or XT{term}_b (J1)."""
        if probe == "J0":
            return self.vec("XB", 0.0) - self.id_b() * self.scalar("X", 0.0)
        return self.vec(f"XT{term}", 0.0)

    def b_ideal(self, t: float) -> np.ndarray:
        return self.vec("B", t)


def _fmt(template: str, family: str, card: str | None = None) -> str:
    """Fill {family} and, when present, {card} in an ideal-grid template."""
    out = template.replace("{family}", family)
    if "{card}" in out:
        if not card:
            raise ValueError(f"template {template!r} needs a card name")
        out = out.replace("{card}", card)
    return out


def load_ideal_grids(template: str, families, card=None, expect_ns=None) -> dict:
    """{family: IdealGrid} from a template with '{family}'."""
    return {f: IdealGrid(_fmt(template, f, card), expect_ns=expect_ns) for f in families}


# ------------------------------------------------------------------ calibration
def kappa_from_mirror(sx_m, sxe_m, sx_i0):
    """kappa = sx_m / sx_ideal0 with the |sx_ideal0| > 0.02 guard (median
    fill) and its relative error kerr (scripts/losch_t3_pool.py:129-136)."""
    ok = np.abs(sx_i0) > KAP_GUARD
    kap = np.where(ok, sx_m / np.where(ok, sx_i0, 1.0), np.nan)
    if np.all(np.isnan(kap)):
        kap = np.ones_like(sx_m)
    kap = np.where(np.isnan(kap), np.nanmedian(kap), kap)
    kerr = np.where(ok, sxe_m / np.maximum(np.abs(sx_i0), 1e-12),
                    np.median(sxe_m) / KAP_GUARD) / np.maximum(np.abs(kap), 1e-12)
    return kap, kerr


def beta_from_mirror(B_m, b_i0):
    """Published +0.5 anchor at every site (scripts/losch_t3_pool.py:131-133)."""
    ok = np.abs(b_i0 - 0.5) > KAP_GUARD
    return np.where(ok, (B_m - 0.5) / np.where(ok, b_i0 - 0.5, 1.0), 1.0)


def wing_anchor(b_cal, b_ideal, center: int):
    """Per-parity wing-residual subtraction + scatter systematic
    (scripts/losch_t3_pool.py:58-67)."""
    ns = len(b_cal)
    idx = np.arange(ns)
    wing = np.abs(idx - center) >= WING_MIN
    out, syst = b_cal.copy(), np.zeros(ns)
    for par in (0, 1):
        s = wing & (idx % 2 == par)
        allp = idx % 2 == par
        if s.sum() == 0:
            continue
        delta = np.mean(b_cal[s] - b_ideal[s])
        out[allp] -= delta
        syst[allp] = np.std(b_cal[s] - b_ideal[s]) / np.sqrt(s.sum())
    return out, syst


def rebuild_C(anc, b_cal, id_a, id_b, A0):
    """Full correlator from the calibrated ancilla signal and probe one-point
    (scripts/losch_t3_pool.py:94-95)."""
    return anc + id_a * (b_cal - id_b) + id_b * (A0 - id_a) + id_a * id_b


class JobBits:
    """One job's bit arrays (name -> (shots, n_logical) uint8) and estimator cache."""

    def __init__(self, bits: dict, lat: Lattice, job_id: str = "", eta: float = ETA):
        self.bits = bits
        self.lat = lat
        self.job_id = job_id
        self.eta = eta
        self._cache = {}

    def has(self, name: str) -> bool:
        return name in self.bits

    def est(self, name: str, setting: str) -> dict:
        if name not in self._cache:
            self._cache[name] = estimate_probes(self.bits[name], self.lat, setting, self.eta)
        return self._cache[name]

    def mirror_name(self, t: float, readout: str, anc: str, family: str = "j0",
                    dt: float = DT) -> str | None:
        """Mirror pub calibrating (t, readout, dt): the family's own mirror if
        present, else the j0 mirror; the physics pub itself at t = 0."""
        for fam in (family, "j0"):
            n = pub_name(fam, t, True, readout, anc, dt)
            if n in self.bits:
                return n
        return None


def _calib_J0(job: JobBits, family: str, t: float, ideal_fam: IdealGrid, ideal_j0: IdealGrid,
              center: int, anc: str = "X", wing: bool = True, dt: float = DT) -> dict | None:
    """One job, one family, J0 probes (Z readout): calibrated ancilla signal,
    b_cal, errors (scripts/ibm_hardware.py:358-380, losch_t3_pool.py:124-146)."""
    pname = pub_name(family, t, False, "Z", anc, dt)
    if not job.has(pname):
        return None
    p = job.est(pname, "Z")
    mname = job.mirror_name(t, "Z", anc, family, dt) if t > 0 else pname
    if mname is None:
        return None
    m = job.est(mname, "Z")
    sx_i0 = ideal_j0.sx_ideal0("J0")
    b_i0 = ideal_j0.b_ideal(0.0)
    if t > 0:
        kap, kerr = kappa_from_mirror(m["sx"], m["sxe"], sx_i0)
        bet = beta_from_mirror(m["B"], b_i0)
    else:   # t = 0 references are reported raw (kappa = beta = 1)
        kap, kerr, bet = np.ones(lat_ns(job)), np.zeros(lat_ns(job)), np.ones(lat_ns(job))
    b_cal = 0.5 + (p["B"] - 0.5) / bet
    sN = np.zeros_like(b_cal)
    if wing:
        b_cal, sN = wing_anchor(b_cal, ideal_fam.b_ideal(t), center)
    c_a = FAMILY_COEFF[family] * (job.eta if family.startswith("j1") else 1.0)
    id_a = ideal_fam.id_a
    anc_cal = c_a * p["sx"] / kap
    anc_raw = c_a * p["sx"]
    var = ((c_a / kap) ** 2 * p["sxe"] ** 2 + (c_a / kap) ** 2 * p["sx"] ** 2 * kerr ** 2
           + (id_a * p["Be"] / np.abs(bet)) ** 2 + (id_a * sN) ** 2)
    return dict(anc_cal=anc_cal, anc_raw=anc_raw, var=var, kap=kap, bet=bet, b_cal=b_cal,
                B=p["B"], Be=p["Be"], bvar=p["Be"] ** 2 / bet ** 2 + sN ** 2, N=p["N"],
                gauss=p["gauss"], gauss_m=m["gauss"], xa=p["xa"], mirror=mname)


def _calib_J1(job: JobBits, family: str, t: float, ideal_j0: IdealGrid, anc: str = "X",
              dt: float = DT) -> dict | None:
    """One job, one family, J1 probes (XYA + XYB readouts): calibrated
    <X J1_b> = (eta/4)(xT1/kappa1 - xT2/kappa2), raw one-point <J1_b>."""
    names = [pub_name(family, t, False, r, anc, dt) for r in ("XYA", "XYB")]
    if not all(job.has(n) for n in names):
        return None
    pA, pB = (job.est(n, r) for n, r in zip(names, ("XYA", "XYB")))
    p1, p2 = term_split(pA, pB)
    if t > 0:
        mnames = [job.mirror_name(t, r, anc, family, dt) for r in ("XYA", "XYB")]
        if any(mn is None for mn in mnames):
            return None
        mA, mB = (job.est(n, r) for n, r in zip(mnames, ("XYA", "XYB")))
        m1, m2 = term_split(mA, mB)
    else:
        m1, m2, mnames = p1, p2, names
    if t > 0:
        kap1, kerr1 = kappa_from_mirror(m1["xT"], m1["xTe"], ideal_j0.sx_ideal0("J1", 1))
        kap2, kerr2 = kappa_from_mirror(m2["xT"], m2["xTe"], ideal_j0.sx_ideal0("J1", 2))
    else:
        kap1 = kap2 = np.ones(lat_ns(job))
        kerr1 = kerr2 = np.zeros(lat_ns(job))
    eta = job.eta
    c_a = FAMILY_COEFF[family] * (eta if family.startswith("j1") else 1.0)
    xJ1_cal = eta / 4 * (p1["xT"] / kap1 - p2["xT"] / kap2)
    xJ1_raw = eta / 4 * (p1["xT"] - p2["xT"])
    one = eta / 4 * (p1["T"] - p2["T"])
    one_e = eta / 4 * np.hypot(p1["Te"], p2["Te"])
    var = (c_a * eta / 4) ** 2 * ((p1["xTe"] ** 2 + p1["xT"] ** 2 * kerr1 ** 2) / kap1 ** 2
                                  + (p2["xTe"] ** 2 + p2["xT"] ** 2 * kerr2 ** 2) / kap2 ** 2)
    return dict(anc_cal=c_a * xJ1_cal, anc_raw=c_a * xJ1_raw, var=var,
                kap=np.minimum(kap1, kap2), kap1=kap1, kap2=kap2, one=one, one_e=one_e,
                N=min(p1["N"], p2["N"]), mirror=mnames)


# ------------------------------------------------------------------ slices
def _merge(slabs: list[dict], key: str = "anc_cal", var: str = "var"):
    """Chi2-inflated inverse-variance merge (scripts/ibm_hardware.py:405-425)."""
    w = np.array([1.0 / np.maximum(s[var], 1e-18) for s in slabs])
    wsum = w.sum(0)
    mean = sum(w[i] * slabs[i][key] for i in range(len(slabs))) / wsum
    err = 1.0 / np.sqrt(wsum)
    if len(slabs) > 1:
        chi2 = sum(w[i] * np.abs(slabs[i][key] - mean) ** 2 for i in range(len(slabs))) / (len(slabs) - 1)
        err = err * np.sqrt(np.maximum(chi2, 1.0))
    return mean, err


def component_slice(comp: str, t: float, jobs: list[JobBits], ideals: dict, lat: Lattice,
                    center: int, backend: str = "", anc: str = "X", wing: bool = True,
                    dt: float = DT) -> dict | None:
    """Assemble one (component, t, dt) slice from all jobs that carry its pubs.
    t = 0 references are raw: kappa_v = 1 explicitly and raw_reference = True."""
    fams, probe = COMPONENT_LAYOUT[comp]
    ideal_j0 = ideals["j0"]
    A0 = ideals[fams[0]].A0
    id_b = ideal_j0.id_b() if probe == "J0" else np.zeros(lat.ns)
    id_a = ideals[fams[0]].id_a
    per_job = []
    for job in jobs:
        parts = []
        for fam in fams:
            r = (_calib_J0(job, fam, t, ideals[fam], ideal_j0, center, anc, wing, dt) if probe == "J0"
                 else _calib_J1(job, fam, t, ideal_j0, anc, dt))
            if r is None:
                parts = []
                break
            parts.append(r)
        if not parts:
            continue
        slab = {"anc_cal": sum(p["anc_cal"] for p in parts),
                "anc_raw": sum(p["anc_raw"] for p in parts),
                "var": sum(p["var"] for p in parts),
                "kap": np.mean([p["kap"] for p in parts], 0),
                "N": sum(p["N"] for p in parts), "job": job.job_id}
        if probe == "J0":
            slab["bet"] = np.mean([p["bet"] for p in parts], 0)
            wb = np.array([1.0 / np.maximum(p["bvar"], 1e-18) for p in parts])
            slab["b_cal"] = sum(wb[i] * parts[i]["b_cal"] for i in range(len(parts))) / wb.sum(0)
            slab["bvar"] = 1.0 / wb.sum(0)
            slab["B"] = np.mean([p["B"] for p in parts], 0)
            slab["gauss"] = np.mean([p["gauss"] for p in parts], 0)
        else:
            slab["kap1"] = np.mean([p["kap1"] for p in parts], 0)
            slab["kap2"] = np.mean([p["kap2"] for p in parts], 0)
            slab["one"] = np.mean([p["one"] for p in parts], 0)
        per_job.append(slab)
    if not per_job:
        return None
    anc_cal, err = _merge(per_job)
    anc_raw, _ = _merge(per_job, key="anc_raw")
    kap = np.ones(lat.ns) if t == 0 else np.mean([s["kap"] for s in per_job], 0)
    out = {"times": np.array([t]), "probes": np.arange(lat.ns), "id_a": id_a, "dt": float(dt),
           "raw_reference": bool(t == 0),
           "c_a": FAMILY_COEFF[fams[0]] * (lat_eta(jobs) if fams[0].startswith("j1") else 1.0),
           "id_b": id_b, "kappa_v": kap[None, :], "C_err": err[None, :],
           "tier": 4, "backend": backend, "nshot": int(sum(s["N"] for s in per_job)),
           "merged_jobs": np.array([s["job"] for s in per_job]), "component": comp, "t": float(t),
           "mask": (kap < KAP_MASK)[None, :]}
    if probe == "J0":
        b_cal, _ = _merge(per_job, key="b_cal", var="bvar")
        C_cal = rebuild_C(anc_cal, b_cal, id_a, id_b, A0)
        C_raw = rebuild_C(anc_raw, np.mean([s["B"] for s in per_job], 0), id_a, id_b, A0)
        out.update(C=(C_raw + 0j)[None, :], C_cal=(C_cal + 0j)[None, :], b_cal=(b_cal + 0j)[None, :],
                   beta_v=np.mean([s["bet"] for s in per_job], 0)[None, :],
                   gauss=np.mean([s["gauss"] for s in per_job], 0)[None, :])
    else:
        one = np.mean([s["one"] for s in per_job], 0)
        C_cal = anc_cal + id_a * one + id_b * (A0 - id_a) + id_a * id_b
        C_raw = anc_raw + id_a * one
        out.update(C=(C_raw + 0j)[None, :], C_cal=(C_cal + 0j)[None, :],
                   beta_v=np.ones((1, lat.ns)), one_pt=one[None, :],
                   kappa1_v=np.mean([s["kap1"] for s in per_job], 0)[None, :],
                   kappa2_v=np.mean([s["kap2"] for s in per_job], 0)[None, :])
    return out


def lat_eta(jobs) -> float:
    return jobs[0].eta if jobs else ETA


def lat_ns(job: JobBits) -> int:
    return job.lat.ns


SLICE_KEYS = ("times", "probes", "C", "C_cal", "C_err", "kappa_v", "beta_v", "id_a", "c_a",
              "id_b", "tier", "backend", "nshot", "merged_jobs", "component", "t", "dt", "raw_reference")


def slice_path(out_template: str, comp: str, t: float, dt: float = DT) -> str:
    """slice_{comp}_t{T}.npz, or slice_{comp}_t{T}_dt0.25.npz for the control."""
    path = out_template.format(comp=comp, t=t)
    if abs(dt - DT) > 1e-9:
        root, ext = os.path.splitext(path)
        path = f"{root}_dt{dt:g}{ext}"
    return path


def load_job_bits(bits_path: str, lat: Lattice, eta: float = ETA, prefix: str | None = None) -> JobBits:
    """data/hw/htq_bits_<id>.npz -> JobBits (pub arrays only; meta keys skipped).
    ``prefix`` ('preset.card:' from campaign.name_prefix) keeps only that
    preset's pubs and strips the prefix, so composed campaigns are analysed
    one card at a time; None keeps un-prefixed pubs only."""
    z = np.load(bits_path, allow_pickle=True)
    bits = {}
    for k in z.files:
        if z[k].dtype != np.uint8 or z[k].ndim != 2:
            continue
        pre, bare = split_prefix(k)
        if (prefix or "") == pre:
            bits[bare] = z[k]
    job_id = str(z["job_id"]) if "job_id" in z.files else os.path.basename(bits_path)
    return JobBits(bits, lat, job_id, eta)


def analyze(bits_paths, ideal_template: str, out_template: str, ns: int, center: int,
            times=None, components=COMPONENTS, eta: float = ETA, backend: str = "",
            anc: str = "X", wing: bool = True, log=print, prefix: str | None = None) -> dict:
    """Bits files (one per job) -> slice npz files.  -> {(comp, t): path}.
    ``prefix`` selects one preset/card of a composed campaign."""
    lat = Lattice(ns)
    jobs = [load_job_bits(p, lat, eta, prefix) for p in bits_paths]
    jobs = [j for j in jobs if j.bits]
    fams = sorted({f for c in components for f in COMPONENT_LAYOUT[c][0]} | {"j0"})
    ideals = load_ideal_grids(ideal_template, fams)
    present = {(p["t"], p["dt"]) for j in jobs for p in map(parse_pub_name, j.bits) if p["family"] != "qpdf"}
    if times is not None:
        present = {(t, dt) for t, dt in present if any(abs(t - x) < 1e-9 for x in times)}
    written = {}
    os.makedirs(os.path.dirname(out_template.format(comp="00", t=0.0)) or ".", exist_ok=True)
    for comp in components:
        for t, dt in sorted(present):
            sl = component_slice(comp, t, jobs, ideals, lat, center, backend, anc, wing, dt)
            if sl is None:
                continue
            path = slice_path(out_template, comp, t, dt)
            np.savez(path, **sl)
            written[(comp, t) if abs(dt - DT) < 1e-9 else (comp, t, dt)] = path
            ok = ~sl["mask"][0]
            if log:
                log(f"slice {comp} t={t:.1f}{'' if abs(dt - DT) < 1e-9 else f' dt={dt:g}'}: "
                    f"kappa(center) {sl['kappa_v'][0][center]:.3f}, "
                    f"masked {int((~ok).sum())}/{ns}, median err "
                    f"{np.median(sl['C_err'][0][ok]) if ok.any() else np.nan:.4f}, "
                    f"nshot {sl['nshot']} -> {path}")
    return written


def kappa_summary(jobs: list[JobBits], ideals: dict, times, lat: Lattice, anc: str = "X") -> dict:
    """{f'{fam}_{probe}_kappa_t{T:.1f}', ..._mask_...} in the data/test/kappa_ns10.npz style."""
    out = {}
    ideal_j0 = ideals["j0"]
    for t in times:
        for job in jobs:
            r0 = _calib_J0(job, "j0", t, ideal_j0, ideal_j0, lat.ns // 2, anc, wing=False)
            if r0 is not None:
                out[f"j0_J0_kappa_t{t:.1f}"] = r0["kap"]
                out[f"j0_J0_mask_t{t:.1f}"] = r0["kap"] < KAP_MASK
            r1 = _calib_J1(job, "j0", t, ideal_j0, anc)
            if r1 is not None:
                for k in (1, 2):
                    out[f"j0_J1T{k}_kappa_t{t:.1f}"] = r1[f"kap{k}"]
                    out[f"j0_J1T{k}_mask_t{t:.1f}"] = r1[f"kap{k}"] < KAP_MASK
    return out


# ------------------------------------------------------------------ quasi-PDF bilinears
def qpdf_term(bits, lat: Lattice, center: int, m: int, sign: int) -> np.ndarray:
    """Per-shot value of the Wilson-line string between site ``center`` and
    site center + sign*2m read in a qpdf setting: product of the +-1 outcomes
    of the two end site qubits and of every qubit strictly between them
    (htensor/quasipdf.py:24-40: X/Y at the ends, Z on the interior)."""
    s = signs_from_bits(bits)
    qa, qb = sorted((lat.site_qubit(center), lat.site_qubit(center + sign * 2 * m)))
    cols = list(range(qa, qb + 1))
    return np.prod(s[:, cols], axis=1)


def estimate_qpdf(bits_by_setting: dict, lat: Lattice, center: int, ms=(1, 2, 3, 4, 5),
                  vac_bits_by_setting: dict | None = None) -> dict:
    """h(z) = <psi-bar(z) W psi(0)> for z = +-2m from the four settings
    qXXm, qYYm, qXYm, qYXm (Pauli at the centre / at the far ends):
      O_R = (<XX> + <YY>)/2,   O_I = (<X_a Y_b> - <Y_a X_b>)/2  with a < b,
    so for z > 0 (a = centre) X_a Y_b comes from qXY and Y_a X_b from qYX,
    for z < 0 (b = centre) X_a Y_b comes from qYX and Y_a X_b from qXY.
    With ``vac_bits_by_setting`` the vacuum value is subtracted (connected).
    The taste phase (-1)^m and the window are applied downstream
    (scripts/quasipdf_analysis.py).  -> {z: (h complex, err complex)}."""
    out = {}
    for m in ms:
        for sign in (+1, -1):
            def mean_err(setting, src):
                v = qpdf_term(src[f"q{setting}m{m}"], lat, center, m, sign)
                return v.mean(), v.std() / np.sqrt(len(v))
            vals = {k: mean_err(k, bits_by_setting) for k in ("XX", "YY", "XY", "YX")}
            xy, yx = (vals["XY"], vals["YX"]) if sign > 0 else (vals["YX"], vals["XY"])
            h = 0.5 * (vals["XX"][0] + vals["YY"][0]) + 0.5j * (xy[0] - yx[0])
            err = 0.5 * np.hypot(vals["XX"][1], vals["YY"][1]) + 0.5j * np.hypot(xy[1], yx[1])
            if vac_bits_by_setting is not None:
                vv = {k: mean_err(k, vac_bits_by_setting) for k in ("XX", "YY", "XY", "YX")}
                vxy, vyx = (vv["XY"], vv["YX"]) if sign > 0 else (vv["YX"], vv["XY"])
                h -= 0.5 * (vv["XX"][0] + vv["YY"][0]) + 0.5j * (vxy[0] - vyx[0])
                err = (np.hypot(err.real, 0.5 * np.hypot(vv["XX"][1], vv["YY"][1]))
                       + 1j * np.hypot(err.imag, 0.5 * np.hypot(vxy[1], vyx[1])))
            out[sign * 2 * m] = (complex(h), complex(err))
    return out


def qpdf_amplitude(bits_by_setting: dict, lat: Lattice, center: int, ms=(1, 2, 3, 4, 5),
                   vac_bits_by_setting: dict | None = None, z_bits=None, vac_z_bits=None) -> dict:
    """A(z) in the reference convention of scripts/quasipdf_analysis.py:

        A(z) = <chi^dag(c+z) W chi(c)> = (C_R - i C_I) / 2,   z != 0
        A(0) = <n(c)>                  (the local density = J0 on an even site)

    with C_R = <O_R>, C_I = <O_I> the Wilson-line bilinear expectation values
    (htensor/quasipdf.py:24-40), i.e. A = conj(h)/2 with h from
    ``estimate_qpdf``.  ``vac_bits_by_setting`` / ``vac_z_bits`` (the matching
    vacuum card's pubs) make every entry connected.  ``z_bits`` is a Z-setting
    bit array, needed only for the z = 0 entry.
    -> {z: (A complex, err complex)}, z = 0 and +-2m."""
    out = {}
    for z, (h, err) in estimate_qpdf(bits_by_setting, lat, center, ms, vac_bits_by_setting).items():
        out[z] = (complex(h.real - 1j * h.imag) / 2, complex(err.real + 1j * err.imag) / 2)
    if z_bits is not None:
        r = estimate_probes(z_bits, lat, "Z")
        val, err = r["J0"][center % lat.ns], r["J0_err"][center % lat.ns]
        if vac_z_bits is not None:
            rv = estimate_probes(vac_z_bits, lat, "Z")
            val = val - rv["J0"][center % lat.ns]
            err = float(np.hypot(err, rv["J0_err"][center % lat.ns]))
        out[0] = (complex(val), complex(err))
    return out


def qpdf_h_of_m(amplitudes: dict, ms=(-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)):
    """{z: (A, err)} -> (h array, err array) on the same-sublattice grid
    h(m) = A(2m) used by data/qpdf_card_refs.npz (key '<card>_h', m = -5..5).
    The staggered taste phase (-1)^m and the window are applied downstream."""
    h = np.array([amplitudes.get(2 * m, (np.nan, np.nan))[0] for m in ms], dtype=complex)
    e = np.array([amplitudes.get(2 * m, (np.nan, np.nan))[1] for m in ms], dtype=complex)
    return h, e
