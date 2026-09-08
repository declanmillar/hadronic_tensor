"""Circuit builders: state preparation, Trotter evolution, insertion gadgets,
readout layers.  All gates are <= 2-qubit and explicit (no PauliEvolutionGate,
no HamiltonianGate), so the circuits transpile deterministically and the
physics/mirror 2q skeletons can be matched by construction.

Ported from htensor/trotter.py, htensor/stateprep.py, htensor/wavepacket.py
and htensor/measure.py; provenance is cited per function.
"""

import json
import pathlib

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from . import (CENTER, DEFAULT_CARD, DT, ETA, G2, KINDS, M0, MIRROR_EPS,
               VACUUM_THETAS)
from .model import Lattice

CARD_DIR = pathlib.Path(__file__).parent / "cards"
N_VACUUM_PARAMS_PER_LAYER = 4   # htensor/stateprep.py:134


# ------------------------------------------------------------------ primitives
LINK_REFS = ("+", "-")


def strong_coupling_vacuum(lat: Lattice, qc: QuantumCircuit | None = None,
                           link_ref: str = "+") -> QuantumCircuit:
    """|0> even matter, |1> odd matter, links |+> (link_ref "+", production
    default) or |-> (link_ref "-": Z after H on every link): Clifford,
    Gauss-law physical either way since each site sees two links
    (htensor/trotter.py:33-52).  The "-" reference is needed where the
    interacting vacuum sits closer to sigma^x = -1 (e.g. the relA point)."""
    if link_ref not in LINK_REFS:
        raise ValueError(f"link_ref must be '+' or '-', got {link_ref!r}")
    qc = qc if qc is not None else QuantumCircuit(lat.n_qubits)
    for n in range(1, lat.ns, 2):
        qc.x(lat.site_qubit(n))
    for q in lat.link_qubits:
        qc.h(q)
        if link_ref == "-":
            qc.z(q)
    return qc


def single_qubit_layer(qc: QuantumCircuit, lat: Lattice, m0, g2, dt) -> None:
    """exp(-i dt [-(m0/2) sum (-1)^n Z_n + (g2/2) sum X_l]) (htensor/trotter.py:42-45).
    ``dt`` may be a ParameterExpression."""
    for n in range(lat.ns):
        qc.rz(-m0 * (-1) ** n * dt, lat.site_qubit(n))
    for q in lat.link_qubits:
        qc.rx(g2 * dt, q)


HOP_FORMS = ("rxxryy", "xy2cx")


def hop(qc: QuantumCircuit, lat: Lattice, bond: int, theta, form: str = "rxxryy") -> None:
    """exp(-i theta (X_a X_b + Y_a Y_b) Z_l) = CZ(l,b) . XY(theta)(a,b) . CZ(l,b)
    (htensor/trotter.py:14-16, 49-55), with XY(theta) = exp(-i theta (XX+YY)) as

      rxxryy : RXX(2 theta) RYY(2 theta)      (htensor form; 2 rzz in the
               fractional basis, but 4 CZ when theta is a Parameter)
      xy2cx  : (Rx(-pi/2) (x) Rx(-pi/2)) CX (Rx(2 theta) (x) Rz(2 theta)) CX
               (Rx(pi/2) (x) Rx(pi/2))     (2 CX for symbolic theta: CX maps
               X_a -> X_a X_b, Z_b -> Z_a Z_b, and Rx(pi/2) maps ZZ -> YY)
    Both are Operator-identical (htq_hw/tests/test_circuits.py)."""
    qa, qb, ql = lat.bond_qubits(bond)
    qc.cz(ql, qb)
    if form == "rxxryy":
        qc.rxx(2 * theta, qa, qb)
        qc.ryy(2 * theta, qa, qb)
    elif form == "xy2cx":
        qc.rx(-np.pi / 2, qa)
        qc.rx(-np.pi / 2, qb)
        qc.cx(qa, qb)
        qc.rx(2 * theta, qa)
        qc.rz(2 * theta, qb)
        qc.cx(qa, qb)
        qc.rx(np.pi / 2, qa)
        qc.rx(np.pi / 2, qb)
    else:
        raise ValueError(form)
    qc.cz(ql, qb)


def hop_layer(qc: QuantumCircuit, lat: Lattice, eta, dt, parity: int,
              form: str = "rxxryy") -> None:
    """All bonds of one parity, seam bond with the parity-trick sign
    (htensor/trotter.py:58-63)."""
    for b in lat.bonds:
        if b % 2 != parity:
            continue
        s = lat.seam_sign if lat.is_seam(b) else 1
        hop(qc, lat, b, s * eta * dt / 4, form)


# ------------------------------------------------------------------ vacuum
def vacuum_ansatz(lat: Lattice, thetas=VACUUM_THETAS, link_ref: str = "+") -> QuantumCircuit:
    """Layered variational vacuum on top of the strong-coupling state
    (htensor/stateprep.py:31-43).  thetas = (th_e, th_o, th_m, th_g) per
    layer, the number of layers = len(thetas) // 4; link_ref "+" | "-"."""
    thetas = np.asarray(thetas, dtype=float).reshape(-1, N_VACUUM_PARAMS_PER_LAYER)
    qc = strong_coupling_vacuum(lat, link_ref=link_ref)
    for th_e, th_o, th_m, th_g in thetas:
        hop_layer(qc, lat, eta=1.0, dt=th_e, parity=0)
        hop_layer(qc, lat, eta=1.0, dt=th_o, parity=1)
        single_qubit_layer(qc, lat, m0=1.0, g2=0.0, dt=th_m)
        single_qubit_layer(qc, lat, m0=0.0, g2=1.0, dt=th_g)
    return qc


# ------------------------------------------------------------------ Trotter
def trotter_step(lat: Lattice, dt, m0=M0, g2=G2, eta=ETA,
                 qc: QuantumCircuit | None = None, form: str = "rxxryy") -> QuantumCircuit:
    """Palindromic second-order step U1(dt/2) U2(dt/2) U3(dt) U2(dt/2) U1(dt/2)
    (htensor/trotter.py:66-73).  ``dt`` may be a ParameterExpression; ``form``
    selects the hop-gate form (see ``hop``)."""
    qc = qc if qc is not None else QuantumCircuit(lat.n_qubits)
    single_qubit_layer(qc, lat, m0, g2, dt / 2)
    hop_layer(qc, lat, eta, dt / 2, 0, form)
    hop_layer(qc, lat, eta, dt, 1, form)
    hop_layer(qc, lat, eta, dt / 2, 0, form)
    single_qubit_layer(qc, lat, m0, g2, dt / 2)
    return qc


def trotter_block(lat: Lattice, n_steps: int, t, m0=M0, g2=G2, eta=ETA,
                  n_wires: int | None = None, form: str = "rxxryy") -> QuantumCircuit:
    """n_steps steps of total time t (htensor/trotter.py:76-83).

    ``t`` is normally a qiskit Parameter: transpile the parametric block once,
    then ``assign(block, t, n*DT)`` for physics and ``assign(block, t,
    MIRROR_EPS)`` for the depth-matched mirror -> identical 2q skeletons by
    construction.  ``n_wires`` widens the circuit (e.g. to include the
    ancilla wire) without touching it.  ``form``: 'rxxryy' for the fractional
    (rzz) basis, 'xy2cx' for the CZ basis (4 two-qubit gates per hop either
    way; see ``hop``)."""
    qc = QuantumCircuit(n_wires if n_wires is not None else lat.n_qubits)
    if n_steps == 0:
        return qc
    for _ in range(n_steps):
        trotter_step(lat, t / n_steps, m0, g2, eta, qc=qc, form=form)
    return qc


def assign(qc: QuantumCircuit, t, value: float) -> QuantumCircuit:
    """Bind the block parameter.  Accepts the Parameter object or its name
    (qpy round trips rebuild Parameter objects)."""
    if isinstance(t, str):
        params = {p: value for p in qc.parameters if p.name == t}
    else:
        params = {t: value}
    return qc.assign_parameters(params)


def physics_and_mirror(block: QuantumCircuit, t, t_phys: float,
                       eps: float = MIRROR_EPS) -> tuple[QuantumCircuit, QuantumCircuit]:
    """(physics, mirror) from one parametric block (scripts/ibm_hardware.py:123-130)."""
    return assign(block, t, t_phys), assign(block, t, eps)


# ------------------------------------------------------------------ wavepacket
def bond_rotation(qc: QuantumCircuit, lat: Lattice, bond: int, kind: str, angle) -> None:
    """exp(-i angle G_kind) for the two gauge-invariant bond quadratures
    (htensor/wavepacket.py:227-242), with explicit 2q gates:

        hop:  G = (s/4)(X_a X_b + Y_a Y_b) Z_l            -> hop(theta = s*angle/4)
        cur:  G = (s/4)(X_a Y_b - Y_a X_b) Z_l = S_b G_hop S_b^dag
                                                    -> Sdg(b) . hop . S(b)
    s = seam_sign on the seam bond.  Verified against PauliEvolutionGate in
    htq_hw/tests/test_circuits.py."""
    qa, qb, ql = lat.bond_qubits(bond)
    s = lat.seam_sign if lat.is_seam(bond) else 1
    if kind == "hop":
        hop(qc, lat, bond, s * angle / 4)
    elif kind == "cur":
        qc.sdg(qb)
        hop(qc, lat, bond, s * angle / 4)
        qc.s(qb)
    else:
        raise ValueError(kind)


def params_from_vector(vec, offsets, n_layers: int) -> dict:
    """Flat vector -> {(layer, kind, offset): angle}, layer-major, kinds in
    KINDS order, offsets in the given order (htensor/wavepacket.py:284-291)."""
    expected = n_layers * len(KINDS) * len(offsets)
    if len(vec) != expected:
        raise ValueError(f"vector length {len(vec)} != {expected}")
    p, i = {}, 0
    for l in range(n_layers):
        for kind in KINDS:
            for off in offsets:
                p[(l, kind, off)] = float(vec[i])
                i += 1
    return p


def wavepacket_block(lat: Lattice, center: int, params: dict,
                     qc: QuantumCircuit | None = None) -> QuantumCircuit:
    """Local packet block (htensor/wavepacket.py:262-281): per layer, kinds in
    KINDS order, offsets ascending; bond offsets from bond ``center``, site
    and link offsets from site/link ``center``.  |angle| < 1e-14 is skipped."""
    qc = qc if qc is not None else QuantumCircuit(lat.n_qubits)
    for l in sorted({k[0] for k in params}):
        for kind in KINDS:
            for (_, _, off) in sorted(k for k in params if k[0] == l and k[1] == kind):
                ang = params[(l, kind, off)]
                if abs(ang) < 1e-14:
                    continue
                if kind in ("cur", "hop"):
                    bond_rotation(qc, lat, (center + off) % lat.ns, kind, ang)
                elif kind == "site":
                    qc.rz(ang, lat.site_qubit(center + off))
                else:
                    qc.rx(ang, lat.link_qubit(center + off))
    return qc


# ------------------------------------------------------------------ cards
def load_card(name_or_path: str = DEFAULT_CARD) -> dict:
    """Parameter card (couplings, vacuum thetas, block vector) as a dict."""
    p = pathlib.Path(name_or_path)
    if not p.is_file():
        p = CARD_DIR / name_or_path / "card.json"
    with open(p) as f:
        return json.load(f)


def card_block_params(card: dict) -> dict:
    """{} for vacuum cards (no block)."""
    blk = card.get("block") or {}
    if not blk.get("vec"):
        return {}
    return params_from_vector(blk["vec"], blk["offsets"], blk["n_layers"])


def prep_only_circuit(lat: Lattice, card: dict, center: int | None = None) -> QuantumCircuit:
    """prep on n_wires with the ancilla idle (no gadget), for prep-only pubs."""
    center = center if center is not None else card["center"]
    qc = QuantumCircuit(lat.n_wires)
    qc.compose(prep_circuit(card, lat.ns, center), range(lat.n_qubits), inplace=True)
    return qc


def prep_circuit(card: dict, ns: int | None = None, center: int | None = None) -> QuantumCircuit:
    """vacuum ansatz + wavepacket block on the system qubits
    (scripts/ibm_hardware.py:72-80).  ``ns``/``center`` default to the card;
    smaller volumes reuse the same angles (scalable-circuits transfer)."""
    lat = Lattice(ns if ns is not None else card["ns"])
    center = center if center is not None else card["center"]
    qc = vacuum_ansatz(lat, card["vacuum"]["thetas"], link_ref=card_link_ref(card))
    wavepacket_block(lat, center, card_block_params(card), qc=qc)
    return qc


def card_couplings(card: dict) -> tuple[float, float, float]:
    """(m0, g2, eta) of a card.  EVERY Trotter construction must be given
    these: the module-level M0/G2/ETA are the production point only, and a
    card at another coupling (e.g. relA 0.4/1.4/2.3) would otherwise be
    evolved under the wrong Hamiltonian while its preparation stayed right."""
    c = card["couplings"]
    return float(c["m0"]), float(c["g2"]), float(c["eta"])


def card_link_ref(card: dict) -> str:
    """Reference link state of a card: top-level ``link_ref`` (default "+")."""
    return card.get("link_ref", card.get("vacuum", {}).get("link_ref", "+"))


def card_n_layers(card: dict) -> int:
    """Vacuum layers inferred from the theta count."""
    return len(card["vacuum"]["thetas"]) // N_VACUUM_PARAMS_PER_LAYER


# ------------------------------------------------------------------ gadgets
# Hadamard test: ancilla in |+>, controlled Pauli P, evolve, read X_anc (x)
# probe -> Re <probe(t) P> (htensor/measure.py:78-86).
#
#   J0(v)  = (-1)^v/2 - Z_v/2         -> one controlled-Z on the site qubit
#   J1(b)  = (eta/4) Y_a Z_l X_b  -  (eta/4) X_a Z_l Y_b   (two terms)
#
# Term coefficients (units of eta for J1) live in GADGET_COEFF for the analysis.
GADGET_COEFF = {"J0": -0.5, "J1a": +0.25, "J1b": -0.25}
J1_PAULIS = {"J1a": ("Y", "Z", "X"), "J1b": ("X", "Z", "Y")}   # on (a, l, b)
ACCUMULATE = ("ring", "ladder", "direct")

# basis change V with V P V^dag = Z (circuit order), and its inverse
_BASIS_IN = {"X": ("h",), "Y": ("sdg", "h"), "Z": ()}
_BASIS_OUT = {"X": ("h",), "Y": ("h", "s"), "Z": ()}


def _cx_path(accumulate: str, a: int, l: int, b: int) -> list[tuple[int, int]]:
    """CX chain that folds the parity of (a, l, b) onto ``a`` using only
    coupling-map edges: ring a-l-b consecutive; ladder l pendant on b, a-b rail."""
    if accumulate == "ring":
        return [(b, l), (l, a)]
    if accumulate == "ladder":
        return [(l, b), (b, a)]
    raise ValueError(accumulate)


def controlled_pauli(qc: QuantumCircuit, anc: int, ops: dict[int, str]) -> None:
    """One controlled-P per factor (htensor/measure.py:73-75)."""
    for q, p in sorted(ops.items()):
        getattr(qc, "c" + p.lower())(anc, q)


def j1_gadget(qc: QuantumCircuit, anc: int, a: int, l: int, b: int,
              term: str, accumulate: str = "ring") -> None:
    """Controlled 3-local Pauli P_a Z_l P_b via a parity ladder: basis-change
    to Z, fold parity onto ``a`` with two CX, one CZ(anc, a), unfold.
    Operator-exact vs three controlled Paulis; both J1 terms share one 2q
    skeleton (cx, cx, cz, cx, cx).  ``accumulate='direct'`` = three
    controlled Paulis (reference)."""
    pa, pl, pb = J1_PAULIS[term]
    ops = {a: pa, l: pl, b: pb}
    if accumulate == "direct":
        controlled_pauli(qc, anc, ops)
        return
    path = _cx_path(accumulate, a, l, b)
    for q, p in ops.items():
        for g in _BASIS_IN[p]:
            getattr(qc, g)(q)
    for c, t in path:
        qc.cx(c, t)
    qc.cz(anc, a)
    for c, t in reversed(path):
        qc.cx(c, t)
    for q, p in ops.items():
        for g in _BASIS_OUT[p]:
            getattr(qc, g)(q)


def cz_via_neighbor(qc: QuantumCircuit, anc: int, target: int, via: int) -> None:
    """Controlled-Z on ``target`` when the ancilla is adjacent to ``via`` only
    (dither insertion at site center+1 on the ladder): CX(target, via)
    conjugation turns CZ(anc, via) into controlled Z_via Z_target, and one
    more CZ(anc, via) removes Z_via; four edge gates, layout preserving."""
    qc.cz(anc, via)
    qc.cx(target, via)
    qc.cz(anc, via)
    qc.cx(target, via)


def insertion_gadget(lat: Lattice, kind: str, center: int = CENTER,
                     accumulate: str = "ring", anc_site: int | None = None) -> QuantumCircuit:
    """Controlled insertion on n_wires (ancilla = last wire), no h(anc).
    kind: 'J0' (site ``center``), 'J1a'/'J1b' (bond ``center``).  ``anc_site``
    (default ``center``) is the site whose qubit the ancilla is adjacent to;
    a J0 insertion elsewhere goes through that qubit on the ladder."""
    qc = QuantumCircuit(lat.n_wires)
    anc = lat.ancilla
    anc_site = center if anc_site is None else anc_site
    if kind == "J0":
        if accumulate == "ladder" and (center % lat.ns) != (anc_site % lat.ns):
            cz_via_neighbor(qc, anc, lat.site_qubit(center), lat.site_qubit(anc_site))
        else:
            qc.cz(anc, lat.site_qubit(center))
    elif kind in J1_PAULIS:
        a, b, l = lat.bond_qubits(center)
        j1_gadget(qc, anc, a, l, b, kind, accumulate)
    else:
        raise ValueError(kind)
    return qc


def base_circuit(lat: Lattice, card: dict, kind: str = "J0",
                 center: int | None = None, accumulate: str = "ring",
                 gadget_center: int | None = None) -> QuantumCircuit:
    """prep + h(anc) + insertion gadget on n_wires (scripts/ibm_hardware.py:152-159).
    ``gadget_center`` (default ``center``) lets the dither family insert J0 at
    site center+1 while the packet stays centred."""
    center = center if center is not None else card["center"]
    gadget_center = center if gadget_center is None else gadget_center
    qc = QuantumCircuit(lat.n_wires)
    qc.compose(prep_circuit(card, lat.ns, center), range(lat.n_qubits), inplace=True)
    qc.h(lat.ancilla)
    qc.compose(insertion_gadget(lat, kind, gadget_center, accumulate, anc_site=center), inplace=True)
    return qc


# ------------------------------------------------------------------ readout
BASES = ("Z", "XYA", "XYB")

# ISA-native single-qubit sequences V (circuit order) with V P V^dag = Z
ISA_ROTATION = {
    "Z": (),
    "X": (("rz", np.pi / 2), ("sx", None), ("rz", np.pi / 2)),   # H
    "Y": (("sx", None), ("rz", np.pi / 2)),                     # H . Sdg
}


QPDF_KINDS = ("XX", "YY", "XY", "YX")     # (Pauli at the centre, Pauli at the far ends)


def qpdf_setting(m: int, kind: str) -> str:
    return f"q{kind}m{m}"


QPDF_Z = "qZ"


def qpdf_basis_map(lat: Lattice, center: int, m: int, kind: str) -> dict[int, str]:
    """Readout setting covering the Wilson-line bilinears psi-bar(z) W psi(0)
    for z = +2m and z = -2m at once (htensor/quasipdf.py:24-40): the centre
    site qubit and the two end site qubits (center +- 2m) in the given
    Paulis, every qubit strictly between them in Z, all other system qubits
    in Z as well.  kind 'XY' = X at the centre, Y at both far ends."""
    if kind not in QPDF_KINDS or m < 1:
        raise ValueError((m, kind))
    pc, pe = kind[0], kind[1]
    bm = {q: "Z" for q in range(lat.n_qubits)}
    bm[lat.site_qubit(center)] = pc
    for sign in (+1, -1):
        v = center + sign * 2 * m
        if not 0 <= v < lat.ns:
            raise ValueError(f"z = {sign * 2 * m} leaves the ring at Ns = {lat.ns}")
        bm[lat.site_qubit(v)] = pe
    return bm


def qpdf_setting(setting: str):
    """'qXYm3' -> ('XY', 3);  'qZ' -> (None, None), the density setting that
    supplies h(0) = <n(centre)> from the same preparation-only circuit."""
    if setting == QPDF_Z:
        return None, None
    if not setting.startswith("q") or "m" not in setting:
        raise ValueError(f"not a qpdf setting: {setting!r}")
    return setting[1:3], int(setting.split("m")[1])


def qpdf_readout_map(lat: Lattice, center: int, setting: str) -> dict[int, str]:
    """Basis map of any qpdf setting, bilinear or density."""
    kind, m = qpdf_setting(setting)
    if kind is None:
        return {q: "Z" for q in range(lat.n_qubits)}
    return qpdf_basis_map(lat, center, m, kind)


def basis_map(lat: Lattice, basis) -> dict[int, str]:
    """System-qubit measurement bases.
      'Z'  : matter Z, links X   -> J0 profile + Gauss checks (-1)^n Z_n X_l- X_l+
      'XYA': even matter Y, odd matter X, links Z -> J1 term a on even bonds,
             term b on odd bonds
      'XYB': even matter X, odd matter Y, links Z -> the complement."""
    if isinstance(basis, dict):
        return dict(basis)
    if basis == "Z":
        m = {q: "Z" for q in lat.matter_qubits}
        m.update({q: "X" for q in lat.link_qubits})
        return m
    if basis in ("XYA", "XYB"):
        even, odd = ("Y", "X") if basis == "XYA" else ("X", "Y")
        m = {lat.site_qubit(n): (even if n % 2 == 0 else odd) for n in range(lat.ns)}
        m.update({q: "Z" for q in lat.link_qubits})
        return m
    raise ValueError(basis)


def readout_layer(isa: QuantumCircuit, lat: Lattice, layout, basis,
                  ancilla_basis: str = "X") -> QuantumCircuit:
    """Append ISA-native basis rotations on the physical qubits given by
    ``layout`` (logical wire -> physical index, e.g. final_index_layout())
    and measure logical wire i into clbit i (scripts/ibm_loschmidt_run.py:76-93)."""
    bmap = basis_map(lat, basis)
    has_ancilla = isa.num_qubits >= lat.n_wires
    if has_ancilla:
        bmap[lat.ancilla] = ancilla_basis
    n_meas = lat.n_wires if has_ancilla else lat.n_qubits
    out = QuantumCircuit(isa.num_qubits, max(n_meas, isa.num_clbits))   # extra clbits = syndromes
    out.compose(isa, inplace=True)
    for logical, p in bmap.items():
        phys = layout[logical]
        for g, ang in ISA_ROTATION[p]:
            getattr(out, g)(*([ang, phys] if ang is not None else [phys]))
    for logical in range(n_meas):
        out.measure(layout[logical], logical)
    return out
