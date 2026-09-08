"""Lattice geometry and Pauli term tables for the 1+1d Z2 gauge theory.

Ported from htensor/lattice.py, htensor/hamiltonian.py and
htensor/currents.py (this package never imports htensor).

Qubit map (htensor/lattice.py:8-16): matter site n -> qubit 2n, link (n, n+1)
-> qubit 2n+1, seam link (ns-1, 0) -> qubit 2ns-1.  The Hadamard-test
ancilla is qubit 2ns.  Only periodic boundary conditions (even ns) are
supported, because that is the production geometry.

A *term table* is ``[({qubit: 'X'|'Y'|'Z'}, coeff), ...]`` with real
coefficients; an empty dict is the identity.  ``to_sparse_pauli_op`` turns a
table into a qiskit SparsePauliOp (used by the tests only).

Seam convention.  The Jordan-Wigner string of the seam bond runs over the
interior matter qubits 1..ns-2 (htensor/lattice.py:60-63).  On Gauss-law
states the product of all Gauss operators fixes the fermion parity to
(-1)^(ns/2), which turns the string into the c-number ``seam_sign`` =
(-1)^(ns/2+1) multiplying the 3-local bulk form (htensor/trotter.py:9-12,
htensor/hamiltonian.py:148-150).  All hardware circuits use that
replacement; ``exact_seam=True`` in the term builders returns the string
form for tests.
"""

from dataclasses import dataclass

Term = tuple[dict[int, str], float]


@dataclass(frozen=True)
class Lattice:
    """N_s staggered sites with interleaved links, PBC (htensor/lattice.py:6-66)."""

    ns: int

    def __post_init__(self):
        if self.ns < 2 or self.ns % 2:
            raise ValueError("PBC needs an even number (>= 2) of staggered sites")

    @property
    def n_qubits(self) -> int:
        """System qubits (matter + links)."""
        return 2 * self.ns

    @property
    def ancilla(self) -> int:
        """Wire index of the Hadamard-test ancilla."""
        return 2 * self.ns

    @property
    def n_wires(self) -> int:
        """System qubits plus the ancilla."""
        return 2 * self.ns + 1

    @property
    def n_links(self) -> int:
        return self.ns

    @property
    def bonds(self) -> list[int]:
        """Bond labels n for hops (n, n+1); n = ns-1 is the seam."""
        return list(range(self.ns))

    @property
    def matter_qubits(self) -> list[int]:
        return [2 * n for n in range(self.ns)]

    @property
    def link_qubits(self) -> list[int]:
        return [2 * n + 1 for n in range(self.ns)]

    @property
    def seam_sign(self) -> int:
        """Fermion-parity replacement sign on the seam (htensor/trotter.py:27-29)."""
        return (-1) ** (self.ns // 2 + 1)

    def site_qubit(self, n: int) -> int:
        return 2 * (n % self.ns)

    def link_qubit(self, n: int) -> int:
        """Qubit of the link on bond (n, n+1)."""
        return 2 * (n % self.ns) + 1

    def bond_qubits(self, bond: int) -> tuple[int, int, int]:
        """(site a, site b, link) qubits of bond (a=bond, b=bond+1)."""
        return self.site_qubit(bond), self.site_qubit(bond + 1), self.link_qubit(bond)

    def is_seam(self, bond: int) -> bool:
        return (bond % self.ns) == self.ns - 1

    def seam_string_qubits(self) -> list[int]:
        """Matter qubits of the seam JW string (interior sites 1..ns-2)."""
        return [2 * n for n in range(1, self.ns - 1)]


def _seam_factor(lat: Lattice, bond: int, exact_seam: bool) -> tuple[int, dict[int, str]]:
    """(sign, extra Z operators) carried by a bond operator."""
    if not lat.is_seam(bond):
        return 1, {}
    if exact_seam:
        return 1, {q: "Z" for q in lat.seam_string_qubits()}
    return lat.seam_sign, {}


def charge_terms(lat: Lattice, v: int) -> list[Term]:
    """J0(v) = ((-1)^v - Z_v)/2, normal-ordered staggered charge
    (htensor/currents.py:107-112)."""
    v = v % lat.ns
    return [({}, (-1) ** v / 2), ({lat.site_qubit(v): "Z"}, -0.5)]


def hop_terms(lat: Lattice, bond: int, eta: float = 1.0, exact_seam: bool = False) -> list[Term]:
    """(eta/4)(X_a X_b + Y_a Y_b) Z_l on one bond (htensor/hamiltonian.py:173-185)."""
    qa, qb, ql = lat.bond_qubits(bond)
    s, string = _seam_factor(lat, bond, exact_seam)
    return [({qa: "X", qb: "X", ql: "Z", **string}, s * eta / 4),
            ({qa: "Y", qb: "Y", ql: "Z", **string}, s * eta / 4)]


def current_terms(lat: Lattice, bond: int, eta: float = 1.0, exact_seam: bool = False) -> list[Term]:
    """J1 on bond (a, a+1): (eta/4)(Y_a X_b - X_a Y_b) Z_l (htensor/currents.py:115-126).

    Exactly conserved: i[H, J0(v)] = -(J1_{v+1/2} - J1_{v-1/2}).  The seam
    bond uses the parity replacement (same sign as the seam hop)."""
    qa, qb, ql = lat.bond_qubits(bond)
    s, string = _seam_factor(lat, bond, exact_seam)
    return [({qa: "Y", qb: "X", ql: "Z", **string}, s * eta / 4),
            ({qa: "X", qb: "Y", ql: "Z", **string}, -s * eta / 4)]


def gauss_terms(lat: Lattice, n: int) -> list[Term]:
    """G_n = (-1)^n Z_n X_{n-1,n} X_{n,n+1}; +1 on physical states
    (htensor/hamiltonian.py:199-211)."""
    n = n % lat.ns
    ops = {lat.site_qubit(n): "Z", lat.link_qubit(n - 1): "X", lat.link_qubit(n): "X"}
    return [(ops, float((-1) ** n))]


def gauge_terms(lat: Lattice, g2: float) -> list[Term]:
    """(g2/2) sum_links X_l (htensor/hamiltonian.py:159-163)."""
    return [({q: "X"}, g2 / 2) for q in lat.link_qubits]


def mass_terms(lat: Lattice, m0: float) -> list[Term]:
    """-(m0/2) sum_n (-1)^n Z_n (htensor/hamiltonian.py:166-170)."""
    return [({lat.site_qubit(n): "Z"}, -(m0 / 2) * (-1) ** n) for n in range(lat.ns)]


def hamiltonian_terms(lat: Lattice, m0: float, g2: float, eta: float,
                      exact_seam: bool = False) -> list[Term]:
    """Full H = gauge + mass + hopping (htensor/hamiltonian.py:195-196)."""
    terms = gauge_terms(lat, g2) + mass_terms(lat, m0)
    for b in lat.bonds:
        terms += hop_terms(lat, b, eta, exact_seam)
    return terms


def to_sparse_pauli_op(n_qubits: int, terms: list[Term]):
    """Term table -> SparsePauliOp with explicit little-endian labels
    (htensor/pauli.py:72-90).  Tests only."""
    from qiskit.quantum_info import SparsePauliOp

    labels, coeffs = [], []
    for ops, c in terms:
        label = ["I"] * n_qubits
        for q, p in ops.items():
            if not 0 <= q < n_qubits:
                raise ValueError(f"qubit {q} out of range")
            if label[n_qubits - 1 - q] != "I":
                raise ValueError(f"duplicate qubit {q}")
            label[n_qubits - 1 - q] = p
        labels.append("".join(label))
        coeffs.append(c)
    return SparsePauliOp(labels, coeffs=coeffs).simplify()
