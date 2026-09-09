"""Variational vacuum preparation (SC-ADAPT-inspired, fixed Hamiltonian pool).

Layered variational-Hamiltonian ansatz on top of the strong-coupling vacuum:
every generator is a translation-invariant sum of Gauss-law-commuting terms
(the Hamiltonian's own hop / mass / gauge structures), so the ansatz cannot
leak out of the physical sector at ANY parameter value, and the PBC seam is
handled by the same parity trick as time evolution.

Because the theory is gapped and confining, optimal per-layer angles become
volume-independent once ns exceeds the correlation length: optimize
classically at small ns (statevector), then reuse the SAME angles at large ns
(the scalable-circuits trick of arXiv:2308.04481, adapted to explicit links).

Layer l:  exp(-i th_e^l H_even-hop) exp(-i th_o^l H_odd-hop)
          exp(-i th_m^l/2 sum (-1)^n Z_n) exp(-i th_g^l/2 sum X_link)
"""

import numpy as np
import scipy.optimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .lattice import Z2Lattice
from . import hamiltonian as ham
from . import exact
from .trotter import strong_coupling_vacuum_circuit, _hop_layer, _single_qubit_layer

N_PARAMS_PER_LAYER = 4


def vacuum_ansatz(lat: Z2Lattice, thetas: np.ndarray,
                  link_ref: str = "+") -> QuantumCircuit:
    """thetas: flat array of length 4*L -> (th_e, th_o, th_m, th_g) per layer.
    link_ref: reference link state "+" (default) or "-", see
    trotter.strong_coupling_vacuum_circuit."""
    thetas = np.asarray(thetas, dtype=float).reshape(-1, N_PARAMS_PER_LAYER)
    qc = strong_coupling_vacuum_circuit(lat, link_ref=link_ref)
    for th_e, th_o, th_m, th_g in thetas:
        _hop_layer(qc, lat, eta=1.0, dt=th_e, parity=0)
        _hop_layer(qc, lat, eta=1.0, dt=th_o, parity=1)
        _single_qubit_layer(qc, lat, m0=1.0, g2=0.0, dt=th_m)
        _single_qubit_layer(qc, lat, m0=0.0, g2=1.0, dt=th_g)
    return qc


def ansatz_state(lat: Z2Lattice, thetas, link_ref: str = "+") -> np.ndarray:
    return np.asarray(Statevector.from_instruction(
        vacuum_ansatz(lat, thetas, link_ref=link_ref)))


def vacuum_energy(lat: Z2Lattice, m0, g2, eta, thetas, H_sparse=None,
                  link_ref: str = "+") -> float:
    if H_sparse is None:
        H_sparse = exact.to_sparse(ham.build_hamiltonian(lat, m0, g2, eta))
    psi = ansatz_state(lat, thetas, link_ref=link_ref)
    return float(np.real(np.vdot(psi, H_sparse @ psi)))


def optimize_vacuum(lat: Z2Lattice, m0, g2, eta, n_layers: int = 2,
                    x0: np.ndarray | None = None, restarts: int = 3,
                    seed: int = 7, link_ref: str = "+") -> dict:
    """Minimize <H> over the layered ansatz. Deterministic given `seed`.

    link_ref selects the reference link state ("+" default, "-" for the
    large-m0 regime; see trotter.strong_coupling_vacuum_circuit).
    Returns {thetas, energy, exact_energy, fidelity, link_ref, n_layers}
    (exact via sparse ED)."""
    H = exact.to_sparse(ham.build_hamiltonian(lat, m0, g2, eta))
    rng = np.random.default_rng(seed)

    def cost(th):
        psi = ansatz_state(lat, th, link_ref=link_ref)
        return float(np.real(np.vdot(psi, H @ psi)))

    best = None
    starts = []
    if x0 is not None:
        starts.append(np.asarray(x0, dtype=float))
    while len(starts) < restarts:
        starts.append(0.15 * rng.standard_normal(N_PARAMS_PER_LAYER * n_layers))
    for s in starts:
        res = scipy.optimize.minimize(cost, s, method="BFGS",
                                      options={"gtol": 1e-8, "maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    energies, vecs = exact.lowest_physical_states(lat, m0, g2, eta, k=1)
    psi = ansatz_state(lat, best.x, link_ref=link_ref)
    fidelity = float(abs(np.vdot(vecs[:, 0], psi)) ** 2)
    return {"thetas": best.x, "energy": best.fun,
            "exact_energy": float(energies[0]), "fidelity": fidelity,
            "link_ref": link_ref, "n_layers": n_layers}


# ------------------------------------------------------- saved-vacuum files
# Convention (2026-09, WS2-B): data/vac_<tag>.npz holds everything needed
# to rebuild the vacuum circuit at ANY volume -- the angles, the layer
# count, the reference link state, and the couplings they were optimized
# for -- so downstream scripts (train_packets.py, hardware cards) never
# re-run optimize_vacuum or guess link_ref.  Legacy files carrying only
# `thetas` (e.g. data/cgkA_vacuum_thetas.npz) load with link_ref "+" and
# n_layers = len(thetas) // 4.
def save_vacuum(path, thetas, n_layers: int, link_ref: str = "+",
                m0=None, g2=None, eta=None, extra: dict | None = None):
    thetas = np.asarray(thetas, dtype=float).ravel()
    if len(thetas) != N_PARAMS_PER_LAYER * n_layers:
        raise ValueError(f"{len(thetas)} angles for {n_layers} layers")
    if link_ref not in ("+", "-"):
        raise ValueError(f"link_ref must be '+' or '-', got {link_ref!r}")
    payload = dict(thetas=thetas, n_layers=int(n_layers), link_ref=str(link_ref),
                   m0=np.nan if m0 is None else float(m0),
                   g2=np.nan if g2 is None else float(g2),
                   eta=np.nan if eta is None else float(eta))
    for k, v in (extra or {}).items():
        if k in payload:
            raise ValueError(f"extra key {k!r} clashes with a standard key")
        payload[k] = v
    np.savez(path, **payload)


def load_vacuum(path) -> dict:
    """-> {thetas, n_layers, link_ref, m0, g2, eta, ...extra}; tolerant of
    legacy thetas-only files."""
    z = np.load(path, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["thetas"] = np.asarray(out["thetas"], dtype=float).ravel()
    out["n_layers"] = int(out["n_layers"]) if "n_layers" in out \
        else len(out["thetas"]) // N_PARAMS_PER_LAYER
    out["link_ref"] = str(out["link_ref"]) if "link_ref" in out else "+"
    for k in ("m0", "g2", "eta"):
        out[k] = float(out[k]) if k in out else None
    for k in list(out):
        if isinstance(out[k], np.ndarray) and out[k].ndim == 0:
            out[k] = out[k].item()
    return out
