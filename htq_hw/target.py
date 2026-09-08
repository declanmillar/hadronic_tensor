"""Hardware targets: backend resolution, qubit embeddings, transpilation.

Embeddings (logical wire -> physical qubit) for the Ns-site ring:
  ring   : heavy-hex (Heron).  Logical qubits 0..2Ns-1 sit consecutively on
           a 2Ns-cycle of the coupling map; the ancilla is the free third
           neighbour of the hub carrying site ``center``.  The hop gate
           couples sites a, b across the link l between them, so routing
           inserts local SWAPs (the transpiler does that; the cycle keeps
           them local).
  ladder : square lattice (Nighthawk).  Matter sites on an Ns-cycle, the link
           (n, n+1) on a private pendant qubit adjacent to site n+1 (so the
           hop gate CZ(l,b) RXX(a,b) RYY(a,b) CZ(l,b) is routing-free), the
           ancilla adjacent to site ``center``.
  grid   : square-lattice fallback: 2Ns-cycle + ancilla (ring semantics).
  transpiler : no initial layout; O3 chooses.

Transpilation (transpile_bundle): base = prep + h(anc) + gadget at O3 with
the embedding as initial layout; the Trotter step is transpiled ONCE with a
symbolic Parameter t at O3 (numeric O3 would delete the mirror's gates),
starting from the base's final layout; physics = t -> n*DT, mirror = t ->
MIRROR_EPS.  Skeletons are asserted equal (scripts/ibm_hardware.py:183-195).
Fractional gates: real backends via use_fractional_gates=True; fakes lack
rzz, so a GenericBackendV2 twin with the same coupling map and
[rz, sx, x, cz, rzz] stands in.  FoldRzzAngle brings the angles into
[0, pi/2] after assignment (scripts/ibm_loschmidt_run.py:96-111, 215-227).
"""

import hashlib
import io
import json
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Parameter
from qiskit.circuit.library import IGate, XGate, YGate, ZGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from . import CENTER, DT, MIRROR_EPS, __version__
from . import circuits as C
from .model import Lattice

SEED = 7
ISA_BASIS = ["rz", "sx", "x", "cz"]
FRACTIONAL_BASIS = ISA_BASIS + ["rzz"]
TWO_Q = ("cz", "rzz", "cx", "ecr")
FAKES = {"boston": "FakeBoston", "kingston": "FakeKingston", "fez": "FakeFez",
         "pittsburgh": "FakePittsburgh", "marrakesh": "FakeMarrakesh",
         "torino": "FakeTorino", "nighthawk": "FakeNighthawk", "miami": "FakeMiami"}


def _log(msg, log=None):
    if log:
        log(msg)


# ------------------------------------------------------------------ backends
class BackendSafetyError(RuntimeError):
    """A --real submission would not reach the requested physical device.

    Raised rather than letting a run proceed against a stand-in, a
    GenericBackendV2 fractional twin, a simulator, or a different device than
    the one named.  Every such case would otherwise look like a successful
    hardware run and consume (or silently NOT consume) the allocation."""


def canonical_backend_name(spec: str) -> str:
    """'phoenix' -> 'ibm_phoenix'; anything else unchanged."""
    return "ibm_phoenix" if spec in ("phoenix", "ibm_phoenix") else spec


def resolve_backend(spec: str, fractional: bool = False, allow_standin: bool = True):
    """'fake:<name>' | 'grid:RxC' | 'heavyhex:d' | real IBM backend name.
    Real names go through QiskitRuntimeService (saved account) and are never
    touched by the tests.  ``fractional`` requests an rzz-capable target
    (GenericBackendV2 twin for fakes that lack rzz).

    ``allow_standin=False`` (what --real uses) refuses every offline
    substitution: no stand-in for an invisible device, and no fractional twin
    of a real one."""
    basis = FRACTIONAL_BASIS if fractional else ISA_BASIS
    if spec.startswith("fake:"):
        import warnings
        from qiskit_ibm_runtime import fake_provider
        name = spec[5:].lower()
        if name not in FAKES:
            raise ValueError(f"unknown fake backend {name}; known: {sorted(FAKES)}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            be = getattr(fake_provider, FAKES[name])()
    elif spec.startswith("grid:"):
        r, c = (int(x) for x in spec[5:].lower().split("x"))
        be = GenericBackendV2(r * c, coupling_map=CouplingMap.from_grid(r, c),
                              basis_gates=basis, seed=SEED)
    elif spec.startswith("heavyhex:"):
        cm = CouplingMap.from_heavy_hex(int(spec[9:]))
        be = GenericBackendV2(cm.size(), coupling_map=cm, basis_gates=basis, seed=SEED)
    elif spec in ("phoenix", "ibm_phoenix"):
        be = _resolve_real_or_standin("ibm_phoenix", "fake:nighthawk", fractional,
                                      allow_standin=allow_standin)
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        be = QiskitRuntimeService().backend(spec, use_fractional_gates=fractional)
    if fractional and "rzz" not in be.operation_names:
        if is_real_backend(be):
            # substituting a simulator here is how a "real" run silently stops
            # being real; the user must choose the CZ basis instead
            raise BackendSafetyError(
                f"{be.name} exposes no rzz (fractional gates): re-run with --basis cz, or ask IBM to "
                f"enable fractional gates on this device. Refusing to substitute a simulated twin.")
        be = fractional_twin(be)
    return be


PHOENIX_NOTE = ("ibm_phoenix = IBM Nighthawk r2: 120 qubits on a square lattice with 218 couplers "
                "(the FakeNighthawk/FakeMiami coupling map), reset elements on every qubit, dynamic "
                "circuits; per-shot time ~250 us assumed until measured.")


def _resolve_real_or_standin(real_name: str, standin_spec: str, fractional: bool,
                             allow_standin: bool = True):
    """The real device when the saved account can see it, else the offline
    stand-in (same coupling map).  The returned backend carries
    ``_htq_standin_for`` when it is the stand-in.  With ``allow_standin=False``
    the underlying error is raised instead, so a --real run fails in seconds
    rather than silently simulating."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        return QiskitRuntimeService().backend(real_name, use_fractional_gates=fractional)
    except Exception as e:  # no account, instance cannot see it, offline
        if not allow_standin:
            raise BackendSafetyError(
                f"{real_name} is not reachable from the saved account ({type(e).__name__}: {e}). "
                f"Save the instance that carries the allocation "
                f"(QiskitRuntimeService.save_account(name=...)) and retry.") from e
        be = resolve_backend(standin_spec, fractional)
        be._htq_standin_for = real_name
        be._htq_standin_reason = f"{type(e).__name__}"
        be._htq_standin_error = repr(e)
        return be


def require_real_backend(be, spec: str, fractional: bool = False) -> dict:
    """Assert that ``be`` IS the live IBM device named by ``spec``.

    Collects every failing condition so one run reports all of them, and
    returns a stamp for the job metadata / acceptance record."""
    want = canonical_backend_name(spec)
    why = []
    if getattr(be, "_htq_standin_for", None):
        why.append(f"resolved to the offline stand-in {be.name} for {be._htq_standin_for} "
                   f"({getattr(be, '_htq_standin_error', be._htq_standin_reason)}): the saved account "
                   f"cannot see the device")
    if getattr(be, "_htq_twin_of", None):
        why.append(f"resolved to a GenericBackendV2 fractional twin of {be._htq_twin_of}, a simulator")
    if not is_real_backend(be):
        why.append(f"{type(be).__module__}.{type(be).__name__} is not a live IBM backend")
    if getattr(be, "simulator", False):
        why.append(f"{be.name} reports simulator=True")
    if is_real_backend(be) and be.name != want:
        why.append(f"asked for {want} but resolved {be.name}")
    if fractional and "rzz" not in be.operation_names:
        why.append(f"{be.name} exposes no rzz but --basis rzz was requested")
    try:
        if be.status().operational is False:
            why.append(f"{be.name} reports operational=False")
    except Exception as e:
        why.append(f"could not read {be.name}.status() ({type(e).__name__})")
    if why:
        raise BackendSafetyError(
            "refusing to submit as a real hardware run:\n  - " + "\n  - ".join(why))
    stamp = {"name": be.name, "label": backend_label(be), "num_qubits": be.num_qubits,
             "basis_gates": sorted(be.operation_names), "is_real": True}
    try:
        stamp["n_edges"] = len(be.coupling_map.get_edges()) // 2
    except Exception:
        pass
    for attr in ("instance", "last_update_date"):
        try:
            stamp[attr] = str(getattr(be, attr))
        except Exception:
            pass
    return stamp


def is_real_backend(be) -> bool:
    """True for a QiskitRuntimeService backend (not a fake, twin or stand-in)."""
    return (not getattr(be, "_htq_standin_for", None) and not getattr(be, "_htq_twin_of", None)
            and type(be).__module__.startswith("qiskit_ibm_runtime.ibm_backend"))


def per_shot_time(be, circuit) -> dict:
    """Per-shot wall time from the target: circuit duration (readout layer and
    measurements included) + reset (the target's reset instruction duration,
    if any) + a fixed 100 us guess for the platform overhead when the target
    gives no better number.  -> {'circuit_us', 'reset_us', 'overhead_us', 'total_s'}."""
    dur = duration_us(circuit, be)
    reset_us = None
    try:
        props = be.target["reset"]
        d = [p.duration for p in props.values() if p is not None and p.duration is not None]
        reset_us = 1e6 * max(d) if d else None
    except Exception:
        reset_us = None
    overhead_us = 100.0
    total = ((dur or 0.0) + (reset_us or 0.0) + overhead_us) * 1e-6
    return {"circuit_us": dur, "reset_us": reset_us, "overhead_us": overhead_us, "total_s": total}


def fractional_twin(be):
    """Same coupling map, ISA basis + rzz, for offline fractional-gate audits."""
    twin = GenericBackendV2(be.num_qubits, coupling_map=be.coupling_map,
                            basis_gates=FRACTIONAL_BASIS, seed=SEED)
    twin._htq_twin_of = be.name
    return twin


def backend_label(be) -> str:
    base = getattr(be, "_htq_twin_of", None) or be.name
    standin = getattr(be, "_htq_standin_for", None)
    return f"{base} (stand-in for {standin})" if standin else base


# ------------------------------------------------------------------ graph
class Graph:
    """Undirected adjacency of a coupling map."""

    def __init__(self, n: int, edges):
        self.n = n
        self.adj = [set() for _ in range(n)]
        for u, v in edges:
            if u != v:
                self.adj[u].add(v)
                self.adj[v].add(u)

    @classmethod
    def from_coupling_map(cls, cm: CouplingMap) -> "Graph":
        return cls(cm.size(), cm.get_edges())

    @classmethod
    def from_backend(cls, be) -> "Graph":
        return cls.from_coupling_map(be.coupling_map)

    @property
    def n_edges(self) -> int:
        return sum(len(a) for a in self.adj) // 2

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    def neighbors(self, v: int) -> list[int]:
        return sorted(self.adj[v])

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adj[u]

    def bfs(self, src: int) -> list[int]:
        """Edge distances from src (-1 = unreachable)."""
        dist = [-1] * self.n
        dist[src] = 0
        frontier = [src]
        while frontier:
            nxt = []
            for u in frontier:
                for v in self.adj[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        nxt.append(v)
            frontier = nxt
        return dist


# ------------------------------------------------------------------ heavy-hex cycles
def _chains(g: Graph):
    """Chain reduction: hubs = nodes of degree != 2; chains = maximal paths
    through degree-2 nodes.  -> (hubs, {hub: [(chain_id, other_hub, interior)]})."""
    hubs = [v for v in range(g.n) if g.degree(v) not in (0, 2)]
    hubset = set(hubs)
    chains = {}
    for u in hubs:
        for w in g.neighbors(u):
            prev, cur, interior = u, w, []
            while cur not in hubset:
                interior.append(cur)
                nxt = [x for x in g.adj[cur] if x != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
            if cur not in hubset:
                continue
            key = (min(u, cur), max(u, cur), tuple(sorted(interior)))
            chains.setdefault(key, (u, cur, interior))
    by_hub = {h: [] for h in hubs}
    for cid, (u, v, interior) in enumerate(chains.values()):
        by_hub[u].append((cid, v, interior))
        by_hub[v].append((cid, u, interior[::-1]))
    return hubs, by_hub


def _chain_dfs_cycles(g: Graph, length: int, max_cycles: int | None, max_visits: int):
    """Exhaustive chain-reduced DFS for short cycles (faces and face pairs)."""
    hubs, by_hub = _chains(g)
    seen, found, visits = set(), 0, 0
    for start in hubs:
        dist = g.bfs(start)
        stack = [(start, [start], frozenset([start]), frozenset(), 0)]
        while stack:
            visits += 1
            if visits > max_visits:
                return
            h, nodes, used, chains_used, total = stack.pop()
            for cid, v, interior in by_hub[h]:
                if cid in chains_used:
                    continue
                new_total = total + len(interior) + 1
                if new_total > length:
                    continue
                if v == start:
                    cyc = nodes + interior
                    key = frozenset(cyc)
                    if new_total == length and len(key) == length and key not in seen:
                        seen.add(key)
                        found += 1
                        yield cyc
                        if max_cycles and found >= max_cycles:
                            return
                    continue
                if v in used or v < start or new_total + dist[v] > length:
                    continue
                stack.append((v, nodes + interior + [v], used | {v},
                              chains_used | {cid}, new_total))


def _arc(cyc, i, j):
    """Forward arc cyc[i], ..., cyc[j] (cyclic indices, inclusive)."""
    L = len(cyc)
    return [cyc[(i + k) % L] for k in range((j - i) % L + 1)]


def _ear_expansions(g: Graph, cyc: list[int], max_len: int):
    """Cycles obtained by replacing one arc of ``cyc`` with a detour through
    non-cycle qubits (shortest detour per endpoint pair), longer than
    ``cyc`` and at most ``max_len`` long."""
    L, pos, in_cyc = len(cyc), {v: i for i, v in enumerate(cyc)}, set(cyc)
    done = set()
    for i, u in enumerate(cyc):
        parent, frontier, hits = {u: None}, [u], {}
        while frontier:
            nxt = []
            for x in frontier:
                for y in g.adj[x]:
                    if y in parent:
                        continue
                    if y in in_cyc:
                        if y != u and y not in hits and x != u:
                            hits[y] = x
                        continue
                    parent[y] = x
                    nxt.append(y)
            frontier = nxt
        for v, via in hits.items():
            q, x = [v], via
            while x is not None:
                q.append(x)
                x = parent[x]
            q = q[::-1]                      # u ... v through non-cycle nodes
            j = pos[v]
            key = (min(i, j), max(i, j), frozenset(q[1:-1]))
            if key in done:
                continue
            done.add(key)
            for new in (q[:-1] + _arc(cyc, j, i)[:-1],            # Q then arc v->u
                        _arc(cyc, i, j) + q[::-1][1:-1]):          # arc u->v then Q back
                if L < len(new) <= max_len:
                    yield new


def _grown_cycles(g: Graph, seeds, length: int, max_cycles: int | None, max_visits: int):
    """DFS over ear expansions from short seed cycles until ``length`` is hit."""
    seen, found, visits = set(), 0, 0
    stack = [list(c) for c in seeds]
    while stack:
        visits += 1
        if visits > max_visits:
            return
        cyc = stack.pop()
        if len(cyc) == length:
            found += 1
            yield cyc
            if max_cycles and found >= max_cycles:
                return
            continue
        exps = sorted(_ear_expansions(g, cyc, length), key=len)
        for new in exps:
            key = frozenset(new)
            if key not in seen:
                seen.add(key)
                stack.append(new)


def heavyhex_cycles(g: Graph, length: int, max_cycles: int | None = None,
                    max_visits: int = 200_000, max_grow_visits: int = 1000):
    """Simple cycles of exactly ``length`` nodes (ordered node lists).

    Short cycles (<= 24) come from an exhaustive chain-reduced DFS (hubs =
    degree != 2 nodes, chains through degree-2 nodes).  Longer cycles are
    grown from the shortest cycles by ear expansion (replace an arc by a
    longer detour), which on heavy-hex adds one face per step; a 2Ns-cycle
    on the 156-qubit Heron lattice is found in well under a second.  The
    growth search gives up after ``max_grow_visits`` expansions (a few
    seconds), e.g. for lengths that are not multiples of 4 on heavy-hex."""
    if length <= 24:
        yield from _chain_dfs_cycles(g, length, max_cycles, max_visits)
        return
    seeds = []
    for short in range(4, 26, 2):
        seeds = list(_chain_dfs_cycles(g, short, 64, max_visits))
        if seeds:
            break
    yield from _grown_cycles(g, seeds, length, max_cycles, max_grow_visits)


# ------------------------------------------------------------------ square lattice
def grid_coordinates(g: Graph):
    """(rows, cols, {node: (r, c)}) if the graph is (a subgraph of) a
    CouplingMap.from_grid(rows, cols) with the same numbering, else None."""
    for rows in range(2, g.n // 2 + 1):
        if g.n % rows:
            continue
        cols = g.n // rows
        if cols < 2:
            continue
        coords = {v: divmod(v, cols) for v in range(g.n)}
        ok = all(abs(coords[u][0] - coords[v][0]) + abs(coords[u][1] - coords[v][1]) == 1
                 for u in range(g.n) for v in g.adj[u])
        if ok:
            return rows, cols, coords
    return None


def _rect_boundary(r0, c0, r1, c1):
    """Clockwise boundary of the inclusive rectangle as (r, c) tuples."""
    top = [(r0, c) for c in range(c0, c1 + 1)]
    right = [(r, c1) for r in range(r0 + 1, r1 + 1)]
    bottom = [(r1, c) for c in range(c1 - 1, c0 - 1, -1)]
    left = [(r, c0) for r in range(r1 - 1, r0, -1)]
    return top + right + bottom + left


def _notch_slots(r0, c0, r1, c1):
    """Candidate depth-1 U-notches: (u, v, inward) for consecutive non-corner
    boundary pairs, interleaved round-robin over the four sides."""
    sides = {
        "top": [((r0, c), (r0, c + 1), (1, 0)) for c in range(c0 + 1, c1 - 1, 2)],
        "bottom": [((r1, c), (r1, c + 1), (-1, 0)) for c in range(c0 + 1, c1 - 1, 2)],
        "left": [((r, c0), (r + 1, c0), (0, 1)) for r in range(r0 + 1, r1 - 1, 2)],
        "right": [((r, c1), (r + 1, c1), (0, -1)) for r in range(r0 + 1, r1 - 1, 2)],
    }
    out = []
    for i in range(max(len(s) for s in sides.values())):
        for s in sides.values():
            if i < len(s):
                out.append(s[i])
    return out


def _reserved_segments(rect, length: int):
    """All runs of ``length`` consecutive non-corner boundary nodes (every
    side, every offset, centre-most first) that can be kept free of notches
    so each keeps two off-cycle neighbours (pendant + ancilla)."""
    r0, c0, r1, c1 = rect
    sides = [[(r0, c) for c in range(c0 + 1, c1)], [(r1, c) for c in range(c0 + 1, c1)],
             [(r, c0) for r in range(r0 + 1, r1)], [(r, c1) for r in range(r0 + 1, r1)]]
    out = []
    for side in sorted(sides, key=len, reverse=True):
        n = len(side) - length + 1
        if n <= 0:
            continue
        mid = (n - 1) / 2
        for start in sorted(range(n), key=lambda x: abs(x - mid)):
            out.append(side[start:start + length])
    return out


def _reserved_segment(rect, length: int):
    segs = _reserved_segments(rect, length)
    return segs[0] if segs else None


def _notch_option(u, v, dr, dc, d, rows, cols, in_cycle, reserved, avoid):
    """Nodes (new, pend) of a depth-d notch at slot (u, v), or None if it does not fit."""
    e = (v[0] - u[0], v[1] - u[1])
    new_u = [(u[0] + j * dr, u[1] + j * dc) for j in range(1, d + 1)]
    new_v = [(v[0] + j * dr, v[1] + j * dc) for j in range(1, d + 1)]
    new = new_u + new_v[::-1]
    pend = [(new_u[-1][0] + dr, new_u[-1][1] + dc), (new_v[-1][0] + dr, new_v[-1][1] + dc)]
    pend += [(q[0] - e[0], q[1] - e[1]) for q in new_u[:-1]] + [(q[0] + e[0], q[1] + e[1]) for q in new_v[:-1]]
    if any(not (0 <= q[0] < rows and 0 <= q[1] < cols) for q in new + pend):
        return None
    if any(q in in_cycle or q in reserved for q in new + pend):
        return None
    if avoid and any(abs(q[0] - a[0]) + abs(q[1] - a[1]) == 1 for q in new for a in avoid):
        return None
    return new, pend


def _place_notches(slots, need, rows, cols, in_cycle, reserved, avoid, depths, backtrack):
    """Greedy pass over the slots (largest depth first); if it does not reach
    ``need`` nodes and ``backtrack`` is set, a depth-first search over
    skip/depth choices per slot.  -> {frozenset(u, v): (u, new)} or None."""
    def greedy():
        ic, rs, out, left = set(in_cycle), set(reserved), {}, need
        for u, v, (dr, dc) in slots:
            if left <= 0:
                break
            if u in avoid or v in avoid:
                continue
            for d in sorted(depths, reverse=True):
                if 2 * d > left:
                    continue
                opt = _notch_option(u, v, dr, dc, d, rows, cols, ic, rs, avoid)
                if opt:
                    out[frozenset((u, v))] = (u, opt[0])
                    ic.update(opt[0])
                    rs.update(opt[1])
                    left -= 2 * d
                    break
        return out if left <= 0 else None

    res = greedy()
    if res is not None or not backtrack:
        return res

    def dfs(i, ic, rs, out, left):
        if left <= 0:
            return out
        if i >= len(slots) or 2 * max(depths) * (len(slots) - i) < left:
            return None
        u, v, (dr, dc) = slots[i]
        if not (u in avoid or v in avoid):
            for d in sorted(depths, reverse=True):
                if 2 * d > left:
                    continue
                opt = _notch_option(u, v, dr, dc, d, rows, cols, ic, rs, avoid)
                if opt:
                    r = dfs(i + 1, ic | set(opt[0]), rs | set(opt[1]), {**out, frozenset((u, v)): (u, opt[0])}, left - 2 * d)
                    if r is not None:
                        return r
        return dfs(i + 1, ic, rs, out, left)

    return dfs(0, set(in_cycle), set(reserved), {}, need)


def _notched_cycle(rows, cols, rect, n_notch, avoid=(), depths=(1,), backtrack=False):
    """Rectangle boundary plus inward U-notches adding 2 * n_notch nodes in
    total (depth-d notch = 2d nodes: u, u+p, .., u+dp, v+dp, .., v+p, v),
    reserving each notch's inward and lateral neighbours as future pendants.
    Notch slots touching ``avoid`` (boundary nodes) or creating nodes next to
    them are skipped.  Returns the ordered (r, c) cycle or None."""
    r0, c0, r1, c1 = rect
    cyc = _rect_boundary(r0, c0, r1, c1)
    inserts = _place_notches(_notch_slots(r0, c0, r1, c1), 2 * n_notch, rows, cols, set(cyc), set(),
                             set(avoid), depths, backtrack)
    if inserts is None:
        return None
    out = []
    for i, x in enumerate(cyc):
        out.append(x)
        hit = inserts.get(frozenset((x, cyc[(i + 1) % len(cyc)])))
        if hit is not None:
            u, new = hit
            out.extend(new if x == u else new[::-1])
    return out


def _pendant_matching(g: Graph, cycle: list[int], extra_for=None):
    """Maximum bipartite matching cycle vertex -> private non-cycle neighbour.
    ``extra_for`` (vertex or list of vertices) adds a second demand on those
    vertices (the Hadamard-test ancilla, and the gauss-midcircuit ancillas).
    Returns (pendant dict, ancilla or list of ancillas) or None if not perfect."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import maximum_bipartite_matching

    in_cyc = set(cycle)
    extras = [] if extra_for is None else (list(extra_for) if isinstance(extra_for, (list, tuple)) else [extra_for])
    rows = list(cycle) + extras
    cols = sorted({w for v in rows for w in g.adj[v] if w not in in_cyc})
    if len(cols) < len(rows):
        return None
    col_ix = {w: j for j, w in enumerate(cols)}
    data = [(i, col_ix[w]) for i, v in enumerate(rows) for w in g.adj[v] if w not in in_cyc]
    mat = sp.csr_matrix((np.ones(len(data)), ([i for i, _ in data], [j for _, j in data])),
                        shape=(len(rows), len(cols)))
    match = maximum_bipartite_matching(mat, perm_type="column")
    if np.any(match < 0):
        return None
    pend = {v: cols[match[i]] for i, v in enumerate(cycle)}
    ancs = [cols[match[len(cycle) + k]] for k in range(len(extras))]
    if extra_for is None:
        return pend, None
    return pend, (ancs if isinstance(extra_for, (list, tuple)) else ancs[0])


def _rect_candidates(rows, cols, ns):
    """Rectangles with a 1-node margin, boundary <= ns, largest first."""
    out = []
    for h in range(2, rows - 1):
        for w in range(2, cols - 1):
            b = 2 * (h + w) - 4
            if b > ns or (ns - b) % 2:
                continue
            for r0 in range(1, rows - h):
                for c0 in range(1, cols - w):
                    out.append((b, h * w, (r0, c0, r0 + h - 1, c0 + w - 1)))
    out.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [r for _, _, r in out]


def grid_ladder(g: Graph, ns: int, center: int = CENTER, coords=None,
                max_embeddings: int = 8, gauss_sites=None) -> list["Embedding"]:
    """Ladder embeddings on a square lattice: matter Ns-cycle with a private
    pendant per site and an ancilla adjacent to site ``center``; one
    Embedding per ancilla-capable centre vertex (up to ``max_embeddings``)."""
    grid = grid_coordinates(g) if coords is None else coords
    if grid is None:
        return []
    rows, cols, coords = grid
    node = {rc: v for v, rc in coords.items()}
    gs = sorted(int(n) for n in (gauss_sites or []))
    for rect in _rect_candidates(rows, cols, ns):
        n_notch = (ns - (2 * (rect[2] - rect[0] + rect[3] - rect[1]))) // 2
        segments = [None]
        if gs:                              # reserve a straight boundary segment for the patch
            span = max(gs + [center]) - min(gs + [center]) + 1
            segments = _reserved_segments(rect, span)
        for segment in segments:
            cyc_rc = _notched_cycle(rows, cols, rect, n_notch, avoid=segment or (), depths=(2, 1) if gs else (1,),
                                    backtrack=bool(gs))
            if cyc_rc is None or len(cyc_rc) != ns:
                continue
            cyc = [node[rc] for rc in cyc_rc]
            if not all(g.has_edge(cyc[i], cyc[(i + 1) % ns]) for i in range(ns)):
                continue
            if _pendant_matching(g, cyc) is None:
                continue
            rotations = range(ns)
            if segment:                     # centre on the segment's first patch node, either direction
                rotations = [cyc.index(node[segment[center - min(gs + [center])]])]
            embs = []
            for k in rotations:
                v = cyc[k]
                extra = [v] + [cyc[(k + int(n) - center) % ns] for n in gs]
                m = _pendant_matching(g, cyc, extra_for=extra)
                if m is None:
                    continue
                pend, ancs = m
                e = _ladder_embedding(cyc, pend, ancs[0], k, ns, center, info={"rect": rect, "notches": n_notch})
                if gs:
                    e.layout += ancs[1:]
                    e.info["gauss_ancillas"] = {int(n): 2 * ns + 1 + j for j, n in enumerate(gs)}
                    e.info["reserved_segment"] = [node[rc] for rc in segment]
                embs.append(e)
                if len(embs) >= max_embeddings:
                    break
            if embs:
                return embs
    return []


def _ladder_embedding(cyc, pend, anc, k, ns, center, info):
    layout = [0] * (2 * ns + 1)
    for n in range(ns):
        v = cyc[(k + n - center) % ns]
        layout[2 * n] = v
        layout[2 * ((n - 1) % ns) + 1] = pend[v]       # link (n-1, n) pendant on site n
    layout[2 * ns] = anc
    return Embedding("ladder", ns, center, layout, info=dict(info, cycle=list(cyc)))


def _serpentine(r0, c0, h, w):
    """Hamiltonian cycle of an h x w rectangle (h even): column c0 returns."""
    if h % 2:
        raise ValueError("serpentine needs an even number of rows")
    cyc = []
    for i in range(h):
        cs = range(c0 + 1, c0 + w) if i % 2 == 0 else range(c0 + w - 1, c0, -1)
        cyc += [(r0 + i, c) for c in cs]
    cyc += [(r0 + i, c0) for i in range(h - 1, -1, -1)]
    return cyc


def _cut_uturns(cyc, n_cuts):
    """Shorten a grid cycle by 2 per cut: a unit square with three cycle
    edges v1-v2-v3-v4 becomes v1-v4 (drops v2, v3)."""
    cyc = list(cyc)
    for _ in range(n_cuts):
        L = len(cyc)
        for i in range(L):
            v1, v4 = cyc[i], cyc[(i + 3) % L]
            if abs(v1[0] - v4[0]) + abs(v1[1] - v4[1]) == 1:
                drop = {(i + 1) % L, (i + 2) % L}
                cyc = [v for j, v in enumerate(cyc) if j not in drop]
                break
        else:
            return None
    return cyc


def grid_cycle(g: Graph, ns: int, center: int = CENTER, coords=None,
               max_embeddings: int = 8) -> list["Embedding"]:
    """Fallback ring on a square lattice: 2Ns-cycle (serpentine on the
    largest even-row rectangle, U-turns cut to length) + ancilla adjacent to
    logical qubit 2*center."""
    grid = grid_coordinates(g) if coords is None else coords
    if grid is None:
        return []
    rows, cols, coords = grid
    node = {rc: v for v, rc in coords.items()}
    L = 2 * ns
    transpose = rows % 2 == 1 and cols % 2 == 0
    h, w = (cols, rows) if transpose else (rows, cols)
    if h % 2 or h * w < L + 1:
        return []
    cyc_rc = _cut_uturns(_serpentine(0, 0, h, w), (h * w - L) // 2)
    if cyc_rc is None:
        return []
    if transpose:
        cyc_rc = [(c, r) for r, c in cyc_rc]
    cyc = [node[rc] for rc in cyc_rc]
    if not all(g.has_edge(cyc[i], cyc[(i + 1) % L]) for i in range(L)):
        return []
    return _ring_embeddings_from_cycle(g, cyc, ns, center, "grid", max_embeddings)


def _ring_embeddings_from_cycle(g, cyc, ns, center, kind, max_embeddings):
    """All rotations/directions placing logical 2*center on a cycle vertex
    with a free (non-cycle) neighbour for the ancilla."""
    L, in_cyc, out = 2 * ns, set(cyc), []
    for k, v in enumerate(cyc):
        free = [w for w in g.neighbors(v) if w not in in_cyc]
        if not free:
            continue
        for direction in (1, -1):
            layout = [cyc[(k + direction * (i - 2 * center)) % L] for i in range(L)]
            layout.append(free[0])
            out.append(Embedding(kind, ns, center, layout,
                                 info={"cycle": list(cyc), "direction": direction}))
            if len(out) >= max_embeddings:
                return out
    return out


def ring_embeddings(g: Graph, ns: int, center: int = CENTER, max_cycles: int = 4,
                    max_embeddings: int = 64) -> list["Embedding"]:
    """Heavy-hex ring embeddings from up to ``max_cycles`` 2Ns-cycles."""
    out = []
    for cyc in heavyhex_cycles(g, 2 * ns, max_cycles=max_cycles):
        out += _ring_embeddings_from_cycle(g, cyc, ns, center, "ring",
                                           max_embeddings - len(out))
        if len(out) >= max_embeddings:
            break
    return out


# ------------------------------------------------------------------ embedding
class EmbeddingError(ValueError):
    """No structured embedding fits the device.

    Raised rather than silently falling back to a transpiler-chosen layout:
    that fallback abandons the ladder (569 vs 300 two-qubit gates per Trotter
    step), leaves ``initial_layout`` unset so the layout-preservation
    assertion is skipped, and thereby voids the shared physics/mirror skeleton
    the whole mirror mitigation rests on.  Carries diagnostics so the operator
    can see WHY it does not fit.  Subclasses ValueError so callers that caught
    the previous failure mode keep working."""

    def __init__(self, msg, diagnostics=None):
        super().__init__(msg)
        self.diagnostics = diagnostics or {}


def operational_graph(be, exclude_qubits=(), exclude_edges=(), max_2q_error=None,
                      max_readout_error=None, log=None):
    """Coupling graph restricted to hardware we are willing to use.

    Drops qubits and edges the backend marks non-operational, anything the
    caller excludes, and (when thresholds are given) anything whose calibrated
    error is too high.  -> (Graph, report dict).  Real devices always have some
    disabled elements, so every embedding search should start here rather than
    from the pristine coupling map."""
    g0 = Graph.from_backend(be)
    dead_q, dead_e, reasons = set(exclude_qubits), {tuple(sorted(e)) for e in exclude_edges}, {}
    for q in dead_q:
        reasons[f"q{q}"] = "excluded by request"
    for e in dead_e:
        reasons[f"e{e}"] = "excluded by request"
    props = None
    try:
        props = be.properties()
    except Exception:
        props = None
    if props is not None:
        for q in range(g0.n):
            try:
                if not props.is_qubit_operational(q):
                    dead_q.add(q); reasons[f"q{q}"] = "not operational"
                elif max_readout_error is not None and props.readout_error(q) > max_readout_error:
                    dead_q.add(q); reasons[f"q{q}"] = f"readout {props.readout_error(q):.3g}"
            except Exception:
                pass
        for u in range(g0.n):
            for v in g0.adj[u]:
                if u >= v:
                    continue
                for gate in ("cz", "ecr", "cx", "rzz"):
                    try:
                        if not props.is_gate_operational(gate, [u, v]):
                            dead_e.add((u, v)); reasons[f"e{(u, v)}"] = f"{gate} not operational"
                        elif max_2q_error is not None and props.gate_error(gate, [u, v]) > max_2q_error:
                            dead_e.add((u, v)); reasons[f"e{(u, v)}"] = f"{gate} error {props.gate_error(gate, [u, v]):.3g}"
                    except Exception:
                        continue
                    break
    edges = [(u, v) for u in range(g0.n) for v in g0.adj[u]
             if u < v and u not in dead_q and v not in dead_q and (u, v) not in dead_e]
    g = Graph(g0.n, edges)
    report = {"n_qubits": g0.n, "dropped_qubits": sorted(dead_q), "dropped_edges": sorted(dead_e),
              "reasons": reasons, "edges_before": g0.n_edges, "edges_after": g.n_edges,
              "live_qubits": g0.n - len(dead_q), "properties": props is not None}
    if dead_q or dead_e:
        _log(f"operational graph: {len(dead_q)} qubit(s) and {len(dead_e)} edge(s) excluded "
             f"({g.n_edges}/{g0.n_edges} edges live)", log)
    return g, report


def matching_deficit(g: Graph, cycle, extras=()):
    """(unmatched pendant demands, spare free neighbours) for a candidate cycle.
    0 deficit means every cycle site can own a private link qubit."""
    in_cyc = set(cycle)
    rows = list(cycle) + list(extras)
    cols = sorted({w for v in rows for w in g.adj[v] if w not in in_cyc})
    if not cols:
        return len(rows), 0
    m = _pendant_matching(g, list(cycle), extra_for=list(extras) if extras else None)
    if m is not None:
        return 0, len(cols) - len(rows)
    # count how many demands a maximum matching can actually satisfy
    import scipy.sparse as sp
    from scipy.sparse.csgraph import maximum_bipartite_matching
    col_ix = {w: j for j, w in enumerate(cols)}
    data = [(i, col_ix[w]) for i, v in enumerate(rows) for w in g.adj[v] if w not in in_cyc]
    if not data:
        return len(rows), 0
    mat = sp.csr_matrix((np.ones(len(data)), ([i for i, _ in data], [j for _, j in data])),
                        shape=(len(rows), len(cols)))
    match = maximum_bipartite_matching(mat, perm_type="column")
    return int(np.sum(match < 0)), len(cols) - len(rows)


def unit_cycles(g: Graph, length: int = 4, limit: int = 4000):
    """Chordless cycles of ``length`` (the lattice faces) -- the growth seeds."""
    out, seen = [], set()
    for a in range(g.n):
        for b in sorted(x for x in g.adj[a] if x > a):
            for c in sorted(x for x in g.adj[b] if x > a and x != a):
                for d in sorted(x for x in g.adj[c] if x > a and x not in (a, b)):
                    if a in g.adj[d]:
                        key = frozenset((a, b, c, d))
                        if len(key) == 4 and key not in seen:
                            seen.add(key)
                            out.append([a, b, c, d])
                            if len(out) >= limit:
                                return out
    return out


def grow_cycles(g: Graph, ns: int, beam: int = 64, time_budget_s: float = 60.0,
                seed: int = SEED, log=None):
    """Beam search for simple cycles of exactly ``ns`` vertices on the ACTUAL
    device graph.

    Starts from lattice faces and repeatedly replaces one arc by a detour
    (_ear_expansions), keeping the ``beam`` best partial cycles ranked by how
    close they are to admitting a perfect pendant matching.  This replaces
    enumerating idealized rectangles, of which exactly one fits a pristine
    Nighthawk and none fit once a single element is disabled."""
    import time
    rng = np.random.default_rng(seed)
    t0 = time.time()
    frontier = [tuple(c) for c in unit_cycles(g)]
    if not frontier:
        return []
    best_at = {}
    seen = set()
    while frontier and time.time() - t0 < time_budget_s:
        nxt = []
        for cyc in frontier:
            for cand in _ear_expansions(g, list(cyc), max_len=ns):
                key = frozenset(cand)
                if len(cand) > ns or key in seen:
                    continue
                seen.add(key)
                nxt.append(tuple(cand))
        if not nxt:
            break
        scored = []
        for c in nxt:
            deficit, spare = matching_deficit(g, list(c))
            scored.append((deficit, -spare, -len(c), rng.random(), c))
            if len(c) == ns and deficit == 0:
                best_at.setdefault(frozenset(c), list(c))
        scored.sort()
        frontier = [c for *_, c in scored[:beam]]
        if len(best_at) >= beam:
            break
    out = list(best_at.values())
    _log(f"grow_cycles: {len(out)} cycle(s) of length {ns} in {time.time() - t0:.1f}s", log)
    return out


def search_ladder(g: Graph, ns: int, center: int = CENTER, gauss_sites=None, beam: int = 64,
                  time_budget_s: float = 60.0, max_embeddings: int = 8, seed: int = SEED,
                  log=None):
    """Ladder embeddings found on the real graph: an ns-cycle of matter sites,
    a private pendant link qubit for each, and an ancilla adjacent to the
    centre site.  Needs no grid coordinates and tolerates dead qubits/edges."""
    out = []
    gs = sorted(gauss_sites or [])
    for cyc in grow_cycles(g, ns, beam=beam, time_budget_s=time_budget_s, seed=seed, log=log):
        for k in range(len(cyc)):
            v = cyc[k]
            extra = [v] + [cyc[(k + int(n) - center) % ns] for n in gs]
            m = _pendant_matching(g, cyc, extra_for=extra)
            if m is None:
                continue
            pend, ancs = m
            e = _ladder_embedding(cyc, pend, ancs[0], k, ns, center,
                                  info={"source": "graph-search", "rot": k})
            if gs:
                e.info["gauss_ancillas"] = ancs[1:]
            out.append(e)
            if len(out) >= max_embeddings:
                return out
    return out


def largest_feasible_ns(g: Graph, center: int = CENTER, ns_max: int = 50, ns_min: int = 40, **kw):
    """Diagnostic for a device that cannot host ns_max: the largest even ns
    that does fit.  Reported when an embedding fails, because dropping Ns is a
    physics decision (every reference grid is Ns-specific), not a transpiler one."""
    for ns in range(ns_max - (ns_max % 2), ns_min - 1, -2):
        if search_ladder(g, ns, center=min(center, ns - 1), max_embeddings=1, **kw):
            return ns
    return None


@dataclass
class Embedding:
    """Logical wire i (0..2Ns-1 system, 2Ns ancilla) -> physical qubit."""

    kind: str
    ns: int
    center: int
    layout: list[int] | None
    info: dict = field(default_factory=dict)

    @property
    def initial_layout(self):
        return None if self.layout is None else list(self.layout)

    def required_edges(self) -> list[tuple[int, int]]:
        """Logical pairs that must be coupling-map edges."""
        L, anc = 2 * self.ns, 2 * self.ns
        if self.kind == "ladder":
            edges = [(2 * n, (2 * n + 2) % L) for n in range(self.ns)]
            edges += [(2 * n + 1, (2 * n + 2) % L) for n in range(self.ns)]
        else:
            edges = [(i, (i + 1) % L) for i in range(L)]
        return edges + [(anc, 2 * self.center)]

    def physical_edges(self) -> list[tuple[int, int]]:
        return [(self.layout[a], self.layout[b]) for a, b in self.required_edges()]

    def with_gauss_ancillas(self, g: Graph, sites) -> "Embedding":
        """Ladder only: add one spare-qubit ancilla adjacent to each matter site
        in ``sites`` (gauss-midcircuit), as extra wires 2Ns+1, 2Ns+2, ...;
        recorded in info['gauss_ancillas'] = {site: wire}.  Re-runs the pendant
        matching with the extra demands; raises ValueError if it does not fit."""
        if self.kind != "ladder":
            raise ValueError("gauss ancillas need the ladder embedding")
        cyc = self.info["cycle"]
        centre_vertex = self.layout[2 * self.center]
        site_vertex = {n: self.layout[2 * n] for n in range(self.ns)}
        m = _pendant_matching(g, cyc, extra_for=[centre_vertex] + [site_vertex[n] for n in sites])
        if m is None:
            raise ValueError("no room for the gauss ancillas next to the patch sites")
        pend, ancs = m
        layout = list(self.layout)
        for n in range(self.ns):                    # link (n-1, n) pendant on site n
            layout[2 * ((n - 1) % self.ns) + 1] = pend[site_vertex[n]]
        layout[2 * self.ns] = ancs[0]
        layout += ancs[1:]
        info = dict(self.info, gauss_ancillas={int(n): 2 * self.ns + 1 + k for k, n in enumerate(sites)})
        return Embedding(self.kind, self.ns, self.center, layout, info)

    def validate(self, g: Graph) -> "Embedding":
        """Raise ValueError unless the layout is injective, in range and every
        required edge exists."""
        if self.layout is None:
            return self
        n_extra = len(self.info.get("gauss_ancillas", {}))
        if len(self.layout) != 2 * self.ns + 1 + n_extra:
            raise ValueError("layout length")
        if len(set(self.layout)) != len(self.layout):
            raise ValueError("layout not injective")
        if not all(0 <= q < g.n for q in self.layout):
            raise ValueError("physical qubit out of range")
        bad = [(a, b) for a, b in self.required_edges()
               if not g.has_edge(self.layout[a], self.layout[b])]
        for site, wire in self.info.get("gauss_ancillas", {}).items():
            if not g.has_edge(self.layout[2 * int(site)], self.layout[wire]):
                bad.append((2 * int(site), wire))
        if bad:
            raise ValueError(f"{len(bad)} required edges missing, e.g. {bad[:3]}")
        return self

    def summary(self) -> str:
        if self.layout is None:
            return f"{self.kind}: transpiler-chosen layout"
        return (f"{self.kind}: site {self.center} -> q{self.layout[2 * self.center]}, "
                f"ancilla -> q{self.layout[-1]}, {len(self.required_edges())} edges")


def _target_error(target, name_options, qargs):
    for name in name_options:
        if name not in target.operation_names:
            continue
        for q in (qargs, qargs[::-1]):
            p = target[name].get(q)
            if p is not None and p.error is not None:
                return float(p.error)
    return None


def score_embedding(emb: Embedding, be) -> float:
    """Sum of target 2q errors on the required edges plus readout errors on
    the used qubits (0 where the target reports nothing; lower is better)."""
    if emb.layout is None:
        return float("inf")
    target = be.target
    s = 0.0
    for u, v in emb.physical_edges():
        s += _target_error(target, TWO_Q, (u, v)) or 0.0
    for q in emb.layout:
        s += _target_error(target, ("measure",), (q,)) or 0.0
    return s


def choose_embedding(be, ns: int, center: int = CENTER, mode: str = "auto",
                     max_cycles: int = 4, log=None, gauss_sites=None,
                     allow_transpiler: bool = False, graph=None, beam: int = 64,
                     time_budget_s: float = 60.0, seed: int = SEED) -> Embedding:
    """mode: auto | ring | ladder | grid | transpiler.  auto = ladder on a
    square lattice (grid cycle if the ladder does not fit), ring on heavy-hex.
    Ties broken by score_embedding.

    ``graph`` (from operational_graph) restricts the search to usable hardware.
    ``allow_transpiler`` is FALSE by default: a transpiler-chosen layout
    abandons the ladder and skips the layout-preservation assertion, so it must
    be asked for explicitly rather than happening silently on a device with one
    dead coupler.  Diagnostic commands pass True."""
    if mode == "transpiler":
        return Embedding("transpiler", ns, center, None)
    g = graph if graph is not None else Graph.from_backend(be)
    grid = grid_coordinates(g)
    cands = []
    if mode in ("auto", "ladder"):
        if grid:
            cands = grid_ladder(g, ns, center, coords=grid, gauss_sites=gauss_sites)
        if not cands:                       # templates need a pristine lattice; the graph search does not
            cands = search_ladder(g, ns, center, gauss_sites=gauss_sites, beam=beam,
                                  time_budget_s=time_budget_s, seed=seed, log=log)
    if not cands and mode in ("auto", "grid") and grid:
        cands = grid_cycle(g, ns, center, coords=grid)
    if not cands and (mode == "ring" or (mode == "auto" and not grid)):
        cands = ring_embeddings(g, ns, center, max_cycles=max_cycles)
    if not cands:
        diag = {"ns": ns, "center": center, "live_qubits": sum(1 for a in g.adj if a),
                "edges": g.n_edges, "needed_qubits": 2 * ns + 1, "mode": mode}
        try:
            diag["largest_feasible_ns"] = largest_feasible_ns(
                g, center, ns_max=ns, beam=max(16, beam // 2), time_budget_s=time_budget_s / 2)
        except Exception:
            pass
        if allow_transpiler:
            _log(f"no structured embedding fits ({diag}); falling back to the transpiler", log)
            return Embedding("transpiler", ns, center, None)
        raise EmbeddingError(
            f"no {mode} embedding of Ns={ns} on {backend_label(be)}: needs {2 * ns + 1} qubits, "
            f"{diag['live_qubits']} live with {g.n_edges} edges"
            + (f"; largest feasible Ns is {diag['largest_feasible_ns']}"
               if diag.get("largest_feasible_ns") else "")
            + ". Refusing to fall back to a transpiler layout (it doubles the two-qubit count and "
              "voids the physics/mirror skeleton); pass allow_transpiler=True to override.",
            diagnostics=diag)
    for e in cands:
        e.validate(g)
    scored = sorted(((score_embedding(e, be), i, e) for i, e in enumerate(cands)),
                    key=lambda x: (x[0], x[1]))
    best = scored[0][2]
    best.info["score"] = scored[0][0]
    best.info["n_candidates"] = len(cands)
    best.info["redundancy"] = len({tuple(e.layout) for e in cands})
    best.info["spare_qubits"] = g.n - len(set(best.layout or []))
    _log(f"embedding {best.summary()} (score {scored[0][0]:.4g}, "
         f"{len(cands)} candidates)", log)
    return best


# ------------------------------------------------------------------ ISA helpers
def skeleton(qc: QuantumCircuit) -> dict[int, list[tuple[str, int, int]]]:
    """2q skeleton as the per-physical-qubit sequence of (gate name, role,
    partner qubit), i.e. the two-qubit DAG of the circuit.  Invariant under
    re-serialisation of commuting layers (transpile, FoldRzzAngle, twirl);
    for a physics/mirror pair assigned from one transpiled circuit it agrees
    with the flat gate list of scripts/ibm_hardware.py:183-186."""
    per: dict[int, list[tuple[str, int, int]]] = {}
    for inst in qc.data:
        if inst.operation.num_qubits != 2:
            continue
        a, b = (qc.find_bit(q).index for q in inst.qubits)
        per.setdefault(a, []).append((inst.operation.name, 0, b))
        per.setdefault(b, []).append((inst.operation.name, 1, a))
    return per


def skeleton_list(qc: QuantumCircuit) -> list[tuple[str, tuple[int, ...]]]:
    """Flat (2q gate name, physical qubits) list in circuit order."""
    return [(inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits))
            for inst in qc.data if inst.operation.num_qubits == 2]


def assert_skeleton_equal(a: QuantumCircuit, b: QuantumCircuit) -> int:
    """Physics/mirror 2q skeletons must be identical (scripts/ibm_hardware.py:189-195).
    Returns the common 2q count."""
    sa, sb = skeleton(a), skeleton(b)
    if sa != sb:
        for q in sorted(set(sa) | set(sb)):
            if sa.get(q) != sb.get(q):
                raise AssertionError(
                    f"skeleton mismatch on physical qubit {q}: "
                    f"{len(sa.get(q, []))} vs {len(sb.get(q, []))} 2q gates "
                    f"({count_2q(a)} vs {count_2q(b)} total)")
    return count_2q(a)


def count_2q(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def count_ops_2q(qc: QuantumCircuit) -> dict:
    return {k: v for k, v in qc.count_ops().items() if k in TWO_Q}


def depth_2q(qc: QuantumCircuit) -> int:
    return qc.depth(lambda inst: inst.operation.num_qubits == 2)


def duration_us(qc: QuantumCircuit, be) -> float | None:
    """estimate_duration against the backend target, in microseconds (None
    if the target lacks durations)."""
    try:
        return 1e6 * float(qc.estimate_duration(be.target, unit="s"))
    except Exception:
        return None


def fold_rzz(qc: QuantumCircuit) -> QuantumCircuit:
    """FoldRzzAngle -> rzz angles in [0, pi/2]; literal global_phase
    instructions absorbed into qc.global_phase (scripts/ibm_loschmidt_run.py:215-227)."""
    from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

    out = PassManager([FoldRzzAngle()]).run(qc)
    bad = [float(i.operation.params[0]) for i in out.data if i.operation.name == "rzz"
           and not (-1e-9 <= float(i.operation.params[0]) <= np.pi / 2 + 1e-9)]
    if bad:
        raise AssertionError(f"{len(bad)} rzz angles out of [0, pi/2], e.g. {bad[:3]}")
    gp = [i for i in out.data if i.operation.name == "global_phase"]
    if gp:
        out.global_phase += sum(float(i.operation.params[0]) for i in gp)
        out.data = [i for i in out.data if i.operation.name != "global_phase"]
    return out


# ------------------------------------------------------------------ twirling
_PAULI = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}
_RZZ_PAIRS = [("I", "I"), ("I", "Z"), ("Z", "I"), ("Z", "Z"),
              ("X", "X"), ("X", "Y"), ("Y", "X"), ("Y", "Y")]
_CZ_MAP = {"I": {"I": ("I", "I"), "X": ("Z", "X"), "Y": ("Z", "Y"), "Z": ("I", "Z")},
           "X": {"I": ("X", "Z"), "X": ("Y", "Y"), "Y": ("Y", "X"), "Z": ("X", "I")},
           "Y": {"I": ("Y", "Z"), "X": ("X", "Y"), "Y": ("X", "X"), "Z": ("Y", "I")},
           "Z": {"I": ("Z", "I"), "X": ("I", "X"), "Y": ("I", "Y"), "Z": ("Z", "Z")}}
_ALL16 = [(a, b) for a in "IXYZ" for b in "IXYZ"]


def twirl(qc: QuantumCircuit, rng, basis=("rz", "sx", "x", "cz", "rzz")) -> QuantumCircuit:
    """Pauli-twirled instance of an ISA circuit (scripts/rzz_twirl.py:263-291):
    cz gets the 16-pair Clifford twirl, rzz the 8-pair commutant twirl;
    inserted Paulis are merged into the 1q layers, 2q skeleton unchanged."""
    from qiskit import transpile

    out = QuantumCircuit(*qc.qregs, *qc.cregs)
    out.global_phase = qc.global_phase
    for inst in qc.data:
        name = inst.operation.name
        if name in ("cz", "rzz"):
            q0, q1 = (qc.find_bit(q).index for q in inst.qubits)
            if name == "rzz":
                p1, p2 = _RZZ_PAIRS[rng.integers(len(_RZZ_PAIRS))]
                c1, c2 = p1, p2
            else:
                p1, p2 = _ALL16[rng.integers(16)]
                c1, c2 = _CZ_MAP[p1][p2]
            for p, q in ((p1, q0), (p2, q1)):
                if p != "I":
                    out.append(_PAULI[p], [q])
            out.append(inst.operation, [q0, q1])
            for c, q in ((c1, q0), (c2, q1)):
                if c != "I":
                    out.append(_PAULI[c], [q])
        else:
            out.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits],
                       [qc.find_bit(c).index for c in inst.clbits])
    return transpile(out, basis_gates=list(basis), optimization_level=1,
                     routing_method="none")


# ------------------------------------------------------------------ transpile bundle
def _circuit_key(qc: QuantumCircuit) -> str:
    h = hashlib.sha256()
    h.update(f"{qc.num_qubits}|{qc.global_phase}".encode())
    for inst in qc.data:
        h.update(inst.operation.name.encode())
        h.update(str([qc.find_bit(q).index for q in inst.qubits]).encode())
        h.update(str([str(p) for p in inst.operation.params]).encode())
    return h.hexdigest()


def _cache_paths(cache_dir, key):
    d = pathlib.Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.qpy", d / f"{key}.json"


def transpile_isa(qc: QuantumCircuit, be, initial_layout=None, opt_level: int = 3,
                  seed: int = SEED, cache_dir=None, log=None, tag: str = "", routing: str | None = None):
    """O-level transpile with an optional initial layout, qpy-cached by a
    structural hash.  -> (isa circuit, final_index_layout list, seconds)."""
    if initial_layout is not None and len(initial_layout) > qc.num_qubits:
        initial_layout = list(initial_layout)[:qc.num_qubits]      # optional extra ancilla wires
    key = hashlib.sha256("|".join([
        __version__, backend_label(be), str(be.num_qubits), str(sorted(be.operation_names)),
        str(be.coupling_map.get_edges().__len__()), str(initial_layout), str(opt_level),
        str(seed), str(routing), _circuit_key(qc)]).encode()).hexdigest()[:24]
    if cache_dir:
        qpy_path, json_path = _cache_paths(cache_dir, key)
        if qpy_path.exists() and json_path.exists():
            with open(qpy_path, "rb") as f:
                isa = qpy.load(f)[0]
            with open(json_path) as f:
                meta = json.load(f)
            _log(f"{tag}: cache hit {key}", log)
            return isa, meta["final_layout"], 0.0
    kw = {"routing_method": routing} if routing else {}
    pm = generate_preset_pass_manager(backend=be, optimization_level=opt_level,
                                      seed_transpiler=seed, initial_layout=initial_layout, **kw)
    t0 = time.time()
    isa = pm.run(qc)
    dt = time.time() - t0
    fl = list(isa.layout.final_index_layout())
    _log(f"{tag}: transpiled in {dt:.1f}s, {count_2q(isa)} 2q", log)
    if cache_dir:
        with open(qpy_path, "wb") as f:
            qpy.dump(isa, f)
        with open(json_path, "w") as f:
            json.dump({"final_layout": fl, "seconds": dt, "tag": tag}, f)
    return isa, fl, dt


@dataclass
class Bundle:
    spec: str
    basis: str
    kind: str
    lat: Lattice
    emb: Embedding
    card: dict
    base: QuantumCircuit
    base_layout: list[int]
    step: QuantumCircuit
    step_layout: list[int]
    blocks: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    layout_preserved: bool = False
    hop_form: str = "rxxryy"


HOP_FORM_FOR_BASIS = {"cz": "xy2cx", "rzz": "rxxryy"}


def transpile_bundle(be, lat: Lattice, card: dict, emb: Embedding, steps=(1,),
                     basis: str = "cz", kind: str = "J0", seed: int = SEED,
                     cache_dir=None, opt_level: int = 3, log=None,
                     hop_form: str | None = None, gadget_center: int | None = None) -> Bundle:
    """Base (prep + gadget) at O3 on the embedding; parametric Trotter step at
    O3 from the base's final layout; per n in ``steps`` a physics/mirror pair
    with asserted-equal skeletons.  Ladder: layout must be preserved by the
    step and n copies are composed; ring: the n-step block is transpiled
    whole.  basis 'rzz' folds angles into [0, pi/2] after assignment.
    ``hop_form`` defaults to HOP_FORM_FOR_BASIS (xy2cx for CZ, rxxryy for
    rzz: 4 two-qubit gates per hop in both cases)."""
    fractional = basis == "rzz"
    if fractional and "rzz" not in be.operation_names:
        be = fractional_twin(be)
    hop_form = hop_form or HOP_FORM_FOR_BASIS[basis]
    accumulate = "ladder" if emb.kind == "ladder" else "ring"
    base = C.base_circuit(lat, card, kind, center=emb.center, accumulate=accumulate,
                          gadget_center=gadget_center)
    isa_base, fl, s_base = transpile_isa(base, be, emb.initial_layout, opt_level, seed,
                                         cache_dir, log, f"base[{kind},{basis}]")
    t = Parameter("t")
    step = C.trotter_block(lat, 1, t, *C.card_couplings(card), n_wires=lat.n_wires, form=hop_form)
    isa_step, step_fl, s_step = transpile_isa(step, be, fl, opt_level, seed, cache_dir,
                                              log, f"step[{basis},{hop_form}]")
    preserved = step_fl == fl
    if emb.kind == "ladder" and not preserved:
        # routing-free by construction: force it (raises if a gate is off an edge)
        isa_step, step_fl, s_step = transpile_isa(step, be, fl, opt_level, seed, cache_dir, log,
                                                  f"step[{basis},{hop_form},routing=none]", routing="none")
        preserved = step_fl == fl
        if not preserved:
            raise AssertionError(f"ladder embedding: Trotter step did not preserve the layout (kind {kind})")
    b = Bundle(backend_label(be), basis, kind, lat, emb, card, isa_base, fl, isa_step, step_fl,
               seconds={"base": s_base, "step": s_step}, layout_preserved=preserved,
               hop_form=hop_form)
    for n in steps:
        b.blocks[n] = _make_block(b, be, lat, n, t, preserved, fractional, opt_level,
                                  seed, cache_dir, log)
    return b


def _make_block(b: Bundle, be, lat, n, t, preserved, fractional, opt_level, seed,
                cache_dir, log):
    if preserved:
        blk = b.step
        for _ in range(n - 1):
            blk = blk.compose(b.step)
        layout, t_phys, t_mirror, mode, secs = b.step_layout, DT, MIRROR_EPS / n, "composed", 0.0
    else:
        whole = C.trotter_block(lat, n, t, *C.card_couplings(b.card), n_wires=lat.n_wires, form=b.hop_form)
        blk, layout, secs = transpile_isa(whole, be, b.base_layout, opt_level, seed, cache_dir,
                                          log, f"block{n}[{b.basis},{b.hop_form}]")
        t_phys, t_mirror, mode = n * DT, MIRROR_EPS, "whole"
    phys, mir = C.assign(blk, "t", t_phys), C.assign(blk, "t", t_mirror)
    if fractional:
        phys, mir = fold_rzz(phys), fold_rzz(mir)
    n2q = assert_skeleton_equal(phys, mir)
    full = b.base.compose(phys)
    return {"physics": phys, "mirror": mir, "layout": layout, "mode": mode, "n2q": n2q,
            "parametric": blk, "n_steps": n,
            "ops2q": count_ops_2q(phys), "depth2q": depth_2q(phys),
            "duration_us": duration_us(phys, be), "full_n2q": count_2q(full),
            "full_depth2q": depth_2q(full), "full_duration_us": duration_us(full, be),
            "seconds": secs}


def assign_block(b: Bundle, n: int, t_total: float, eps: float = MIRROR_EPS):
    """(physics, mirror) for an n-step block of total time ``t_total`` from the
    bundle's parametric block (any dt, e.g. half steps); rzz angles folded."""
    blk = b.blocks[n]
    per_step = blk["mode"] == "composed"
    phys = C.assign(blk["parametric"], "t", t_total / n if per_step else t_total)
    mir = C.assign(blk["parametric"], "t", eps / n if per_step else eps)
    if b.basis == "rzz":
        phys, mir = fold_rzz(phys), fold_rzz(mir)
    assert_skeleton_equal(phys, mir)
    return phys, mir


# ------------------------------------------------------------------ report
def report(targets, ns: int = 50, steps=(1, 12), bases=("cz", "rzz"), mode: str = "auto",
           gadgets=("J0", "J1a"), card=None, cache_dir=None, seed: int = SEED,
           log=print) -> list[dict]:
    """Offline count table for each target x basis.  -> list of row dicts."""
    card = card or C.load_card()
    lat = Lattice(ns)
    center = card["center"] if ns == card["ns"] else ns // 2 - 1
    rows = []
    for spec in targets:
        be = resolve_backend(spec)
        emb = choose_embedding(be, ns, center, mode=mode, log=log)
        for basis in bases:
            be_b = resolve_backend(spec, fractional=(basis == "rzz"))
            for kind in gadgets:
                _log(f"== {spec} [{emb.kind}] basis={basis} gadget={kind}", log)
                bnd = transpile_bundle(be_b, lat, card, emb, steps, basis, kind, seed,
                                       cache_dir, log=log)
                rows.append(_row(spec, emb, bnd, be_b))
    return rows


def _row(spec, emb, bnd: Bundle, be) -> dict:
    row = {"target": spec, "embedding": emb.kind, "basis": bnd.basis, "gadget": bnd.kind,
           "prep_2q": count_2q(bnd.base), "prep_depth2q": depth_2q(bnd.base),
           "prep_us": duration_us(bnd.base, be), "layout_preserved": bnd.layout_preserved,
           "hop_form": bnd.hop_form,
           "step_2q": count_2q(bnd.step), "step_ops2q": count_ops_2q(bnd.step),
           "step_depth2q": depth_2q(bnd.step), "seconds": dict(bnd.seconds)}
    gad = C.insertion_gadget(bnd.lat, bnd.kind, emb.center,
                             "ladder" if emb.kind == "ladder" else "ring")
    row["gadget_2q_logical"] = count_2q(gad)
    for n, blk in bnd.blocks.items():
        row[f"block{n}"] = {k: blk[k] for k in ("n2q", "ops2q", "depth2q", "duration_us",
                                                 "full_n2q", "full_depth2q",
                                                 "full_duration_us", "mode", "seconds")}
    return row


def format_table(rows) -> str:
    """Fixed-width text table of the report rows."""
    def us(x):
        return "n/a" if x is None else f"{x:.0f}"

    lines = []
    hdr = (f"{'target':16} {'emb':7} {'basis':5} {'gad':4} {'prep2q':>7} {'gad2q':>5} "
           f"{'step2q':>7} {'stepD2':>6} {'lay':>3}")
    steps = sorted({int(k[5:]) for r in rows for k in r if k.startswith("block")})
    for n in steps:
        hdr += f" | {'blk'+str(n)+' 2q':>9} {'D2q':>5} {'us':>6} {'full2q':>7} {'fullus':>7}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        line = (f"{r['target']:16} {r['embedding']:7} {r['basis']:5} {r['gadget']:4} "
                f"{r['prep_2q']:7d} {r['gadget_2q_logical']:5d} {r['step_2q']:7d} "
                f"{r['step_depth2q']:6d} {'ok' if r['layout_preserved'] else 'mv':>3}")
        for n in steps:
            b = r.get(f"block{n}")
            if b is None:
                line += " | " + " " * 39
                continue
            line += (f" | {b['n2q']:9d} {b['depth2q']:5d} {us(b['duration_us']):>6} "
                     f"{b['full_n2q']:7d} {us(b['full_duration_us']):>7}")
        lines.append(line)
    return "\n".join(lines)
