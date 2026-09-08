"""gauss-midcircuit preset (RISK-GATED): mid-circuit Gauss-law syndromes.

j0 x Z family, 8 slices t = 0.5 .. 4.0.  After every second Trotter step the
five Gauss checks G_n = (-1)^n Z_n X_{l-} X_{l+} of the width-2 patch around
the centre (sites CENTER-2 .. CENTER+2) are extracted onto five spare-qubit
ancillas (one per site, adjacent to its matter site on the ladder), measured
and RESET for reuse (dynamic circuit).  Each check is a CX parity ladder over
nearest-neighbour ladder edges:

    H(l-) H(l+)                                  X_l -> Z_l
    CX(n+1, n) CX(l+, n+1) CX(n+1, n) CX(l-, n)  accumulate Z_{l-} Z_{l+} onto site n
    CX(n, a_n)                                   copy the parity to the ancilla
    (undo the four CX)  H(l-) H(l+)
    measure a_n -> syndrome clbit; reset a_n

(l- = link (n-1, n), pendant on site n; l+ = link (n, n+1), pendant on site
n+1, hence the detour through n+1: 8 two-qubit gates per check plus the
copy to the ancilla.  A single spare-qubit ancilla next to site centre+1
serves all five checks through rail relays (relay_copy: 1, 4, 10, 22 CX for
0..3 hops), 81 two-qubit gates per extraction round; the ladder then needs
only one extra node next to one site.)  On Gauss-law states every syndrome is (-1)^n-trivial,
i.e. the measured bit equals (1 - (-1)^n)/2; a flipped bit flags a leaked
shot.  Syndromes are stored per shot in extra clbits (n_wires + round*5 + k).

Analysis (`acceptance`, `postselect`): acceptance = fraction of shots with
all syndromes trivial, per round (vs depth); the calibrated slice is formed
from all shots and from the accepted shots only.
"""

import numpy as np
from qiskit import QuantumCircuit

from . import CENTER, DT
from . import circuits as C
from .model import Lattice

PATCH_OFFSETS = (-2, -1, 0, 1, 2)
GAUSS_TIMES = tuple(round(0.5 * k, 2) for k in range(1, 9))     # 8 slices, t = 0.5 .. 4.0
STEPS_PER_ROUND = 2


def gauss_specs(preset: str = "gauss-midcircuit", card: str = "prod_k1.26_s0.75_ns50"):
    """Pub specs: j0 x Z physics + j0 x Z mirror per slice (both with
    syndromes), one t = 0 reference (final syndrome round only)."""
    from .campaign import manifest
    return manifest(times=GAUSS_TIMES, families=("j0",), readouts=("Z",), card=card, preset=preset)


def patch_sites(lat: Lattice, center: int = CENTER) -> list[int]:
    """The five checked sites (centre-2 .. centre+2)."""
    return [(center + o) % lat.ns for o in PATCH_OFFSETS]


def gauss_ancilla_sites(lat: Lattice, center: int = CENTER) -> list[int]:
    """One spare-qubit ancilla, adjacent to site centre+1: the five checks are
    relayed to it through the rail (relay_chain), so the ladder needs a single
    extra node next to one site (the centre's own free neighbour is taken by
    the Hadamard-test ancilla)."""
    return [(center + 1) % lat.ns]


def relay_plan(lat: Lattice, center: int = CENTER) -> dict:
    """{checked site: (ancilla site, chain of intermediate sites from the
    checked site to the ancilla site)}."""
    a = (center + 1) % lat.ns
    plan = {}
    for o in PATCH_OFFSETS:
        n = (center + o) % lat.ns
        if o <= 1:
            chain = [(center + k) % lat.ns for k in range(o + 1, 2)]      # n+1, ..., a
        else:
            chain = [(center + k) % lat.ns for k in range(o - 1, 0, -1)]   # n-1, ..., a
        plan[n] = (a, chain)
    return plan


def relay_copy(qc: QuantumCircuit, lat: Lattice, site: int, chain: list[int], anc_wire: int) -> None:
    """a ^= z_site through the adjacent-site chain [v1, .., vk] (a adjacent to
    vk):  copy(vk..) ; CX(site, v1) ; copy(v1, ..) ; CX(site, v1) -- the two
    copies of z_v1 cancel, v1 is restored.  Cost 1, 4, 10, 22 CX for 0..3 hops."""
    if not chain:
        qc.cx(lat.site_qubit(site), anc_wire)
        return
    v = chain[0]
    relay_copy(qc, lat, v, chain[1:], anc_wire)
    qc.cx(lat.site_qubit(site), lat.site_qubit(v))
    relay_copy(qc, lat, v, chain[1:], anc_wire)
    qc.cx(lat.site_qubit(site), lat.site_qubit(v))


def syndrome_extraction(qc: QuantumCircuit, lat: Lattice, site: int, anc_wire: int, clbit: int,
                        chain=()) -> None:
    """One Gauss check of ``site`` onto ``anc_wire`` (adjacent to the last site
    of ``chain``, or to ``site`` itself when the chain is empty), measured
    into ``clbit`` and reset."""
    n, n1 = lat.site_qubit(site), lat.site_qubit(site + 1)
    lm, lp = lat.link_qubit(site - 1), lat.link_qubit(site)
    qc.h(lm)
    qc.h(lp)
    ladder = [(n1, n), (lp, n1), (n1, n), (lm, n)]
    for c, t in ladder:
        qc.cx(c, t)
    relay_copy(qc, lat, site, list(chain), anc_wire)
    for c, t in reversed(ladder):
        qc.cx(c, t)
    qc.h(lm)
    qc.h(lp)
    qc.measure(anc_wire, clbit)
    qc.reset(anc_wire)


def n_rounds(n_steps: int) -> int:
    return n_steps // STEPS_PER_ROUND


def gauss_circuit(lat: Lattice, card: dict, n_steps: int, t: float, center: int = CENTER,
                  mirror: bool = False, sites=None, n_extra_wires: int | None = None) -> QuantumCircuit:
    """Logical circuit: base (prep + h(anc) + J0 gadget) + Trotter steps with a
    syndrome round after every STEPS_PER_ROUND steps (+ a final round at t = 0
    for the reference).  Ancilla wires 2Ns+1.., syndrome clbits from n_wires
    on; the system/ancilla readout is added later by readout_layer."""
    sites = list(sites or patch_sites(lat, center))
    k = len(sites)
    anc_sites = gauss_ancilla_sites(lat, center)             # spare-qubit ancillas, wires n_wires + j
    anc_wire = {x: lat.n_wires + j for j, x in enumerate(anc_sites)}
    plan = relay_plan(lat, center)
    rounds = n_rounds(n_steps) if n_steps else 1
    n_wires = lat.n_wires + len(anc_sites)
    qc = QuantumCircuit(n_wires, lat.n_wires + rounds * k)
    qc.compose(C.base_circuit(lat, card, "J0", center=center, accumulate="ladder"), range(lat.n_wires), inplace=True)
    dt = (t / n_steps) if n_steps else DT
    eps_dt = 1e-8 / n_steps if n_steps else 0.0
    done = 0
    for r in range(rounds):
        if n_steps:
            for _ in range(STEPS_PER_ROUND if done + STEPS_PER_ROUND <= n_steps else n_steps - done):
                blk = C.trotter_step(lat, eps_dt if mirror else dt, *C.card_couplings(card), form="xy2cx")
                qc.compose(blk, range(lat.n_qubits), inplace=True)
                done += 1
        for j, site in enumerate(sites):
            a_site, chain = plan[site]
            syndrome_extraction(qc, lat, site, anc_wire[a_site], lat.n_wires + r * k + j, chain=chain)
    while done < n_steps:                                   # leftover steps (odd n)
        qc.compose(C.trotter_step(lat, eps_dt if mirror else dt, *C.card_couplings(card), form="xy2cx"),
                   range(lat.n_qubits), inplace=True)
        done += 1
    return qc


def trivial_syndromes(lat: Lattice, sites) -> np.ndarray:
    """Expected syndrome bit per check on physical states: G_n = +1 means
    Z_n X X = (-1)^n, so the parity bit is 1 for odd sites."""
    return np.array([n % 2 for n in sites], dtype=np.uint8)


def acceptance(bits: np.ndarray, lat: Lattice, sites, rounds: int) -> dict:
    """Syndrome statistics from the fetch-format bits (columns beyond n_wires
    are the syndromes, round-major).  -> {'per_round': cumulative acceptance
    after each round, 'accepted': boolean mask of shots with all syndromes
    trivial, 'flip_rate': mean flip fraction per check}."""
    k = len(sites)
    syn = bits[:, lat.n_wires:lat.n_wires + rounds * k].reshape(len(bits), rounds, k)
    triv = trivial_syndromes(lat, sites)[None, None, :]
    ok_round = np.all(syn == triv, axis=2)
    cum = np.cumprod(ok_round, axis=1)
    return {"per_round": cum.mean(axis=0), "accepted": cum[:, -1].astype(bool),
            "flip_rate": (syn != triv).mean(axis=(0, 1))}


def postselect(bits: np.ndarray, accepted: np.ndarray, lat: Lattice) -> np.ndarray:
    """System + ancilla columns of the accepted shots (fetch format)."""
    return bits[accepted][:, :lat.n_wires]
