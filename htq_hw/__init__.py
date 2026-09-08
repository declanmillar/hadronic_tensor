"""htq_hw: self-contained Qiskit circuits for the Z2 hadronic-tensor hardware run.

This package re-implements, with plain qiskit only (no ``htensor`` import),
the production circuit family used in the Ns=50 (100 system qubits + 1
ancilla) hadamard-test measurement of the vector-current correlator in the
1+1d Z2 gauge theory with staggered fermions:

    prep   = strong-coupling vacuum -> 2-layer variational vacuum
             -> L=3 local wavepacket block                (circuits.prep_circuit)
    gadget = Hadamard-test insertion of J0 or J1          (circuits.insertion_gadget)
    U(t)   = second-order Trotter, palindromic, PBC seam via parity trick
                                                          (circuits.trotter_block)
    mirror = same Trotter skeleton with total angle MIRROR_EPS (net identity)

Hardware targets (target.py): IBM Heron r3 heavy-hex (ring embedding) and
Nighthawk square lattice (ladder embedding), plus offline fakes.

Cross-checked against htensor in tests/test_htq_hw_crosscheck.py.
"""

__version__ = "0.1.0"

# The pinned environment is not advisory: the mirror-skeleton invariant that
# the whole mitigation rests on depends on the transpiler choosing the same
# layout for every insertion family, and qiskit 2.4.1 does not.  Install
# htq_hw/requirements.txt into a clean venv before running anything.
PINNED = {"qiskit": "2.5.0", "qiskit_aer": "0.17.2", "qiskit_ibm_runtime": "0.47.0",
          "numpy": "2.4.6", "scipy": "1.18.0"}


def check_versions(strict: bool = False) -> list[str]:
    """-> list of 'package: found X, pinned Y' mismatches.  ``strict`` raises."""
    import importlib
    bad = []
    for mod, want in PINNED.items():
        try:
            got = importlib.import_module(mod).__version__
        except Exception as e:
            bad.append(f"{mod}: not importable ({type(e).__name__})")
            continue
        if got != want:
            bad.append(f"{mod}: found {got}, pinned {want}")
    if bad and strict:
        raise RuntimeError("environment does not match htq_hw/requirements.txt:\n  - "
                           + "\n  - ".join(bad)
                           + "\nInstall the pinned versions: pip install -r htq_hw/requirements.txt")
    return bad

# production couplings and geometry (htensor/scripts/ibm_hardware.py:48-51)
M0, G2, ETA = 0.7, 1.1, 1.3
NS, CENTER = 50, 24
DT = 0.5

# mirror block total angle -> net identity with the physics skeleton
# (scripts/ibm_hardware.py:130)
MIRROR_EPS = 1e-8

# 2-layer vacuum ansatz angles (th_even, th_odd, th_mass, th_gauge) x 2,
# trained at Ns=6 for (M0, G2, ETA); equal to
# htensor.stateprep.optimize_vacuum(Z2Lattice(6), 0.7, 1.1, 1.3, n_layers=2,
# restarts=2)["thetas"] to 5e-7.
VACUUM_THETAS = (3.089753, -3.742056, -0.785397, 1.570794,
                 -0.461068, 0.194181, 0.785389, 3.141611)

# wavepacket block generator kinds, in circuit order (htensor/wavepacket.py:224)
KINDS = ("cur", "hop", "site", "link")

DEFAULT_CARD = "prod_k1.26_s0.75_ns50"
