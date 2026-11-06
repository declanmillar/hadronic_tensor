"""Preparation RZZs must satisfy the same Runtime limits as bound evolution."""

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.primitives.containers import SamplerPub
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime.utils.validations import validate_rzz_pubs

from htq_hw import campaign as CP
from htq_hw import circuits as C
from htq_hw import target as T
from htq_hw.model import Lattice


@pytest.mark.parametrize("angle", [-1.5448765000000009, -2.1, 2.4, 7.0])
def test_bound_rzz_cold_warm_and_legacy_cache_preserve_unitary_and_layout(tmp_path, angle):
    be = T.resolve_backend("grid:1x2", fractional=True)
    qc = QuantumCircuit(2, global_phase=0.37)
    qc.rz(0.23, 0)
    qc.rzz(angle, 0, 1)
    qc.sx(1)
    expected = Operator(qc).data
    kwargs = dict(initial_layout=[0, 1], cache_dir=str(tmp_path))
    legacy = T.generate_preset_pass_manager(backend=be, optimization_level=3,
                seed_transpiler=T.SEED, initial_layout=[0, 1]).run(qc)

    def check(isa, layout):
        validate_rzz_pubs([SamplerPub.coerce((isa, None, 16))])
        assert T.skeleton(isa) == T.skeleton(legacy)
        assert T.count_2q(isa) == T.count_2q(qc) == 1
        assert isa.layout is not None
        assert layout == isa.layout.final_index_layout() == [0, 1]
        # .equiv() would overlook a lost global phase.
        np.testing.assert_allclose(Operator(isa).data, expected, atol=1e-12, rtol=0)

    cold, layout, _ = T.transpile_isa(qc, be, **kwargs)
    check(cold, layout)
    warm, warm_layout, seconds = T.transpile_isa(qc, be, **kwargs)
    check(warm, warm_layout)
    assert seconds == 0.0

    # A pre-fix cache contains the raw transpiler output with unrestricted
    # RZZ angles. Loading that existing file must repair it without re-routing.
    with pytest.raises(ValueError, match="rzz"):
        validate_rzz_pubs([SamplerPub.coerce((legacy, None, 16))])
    qpy_path, = tmp_path.glob("*.qpy")
    with qpy_path.open("wb") as stream:
        qpy.dump(legacy, stream)
    repaired, repaired_layout, seconds = T.transpile_isa(qc, be, **kwargs)
    check(repaired, repaired_layout)
    assert seconds == 0.0


def test_reference_qpdf_and_bound_evolution_pubs_validate_with_cold_and_warm_caches(tmp_path):
    """The first submission job contains references with no Trotter block.

    Include preparation-only qPDF pubs, plus ordinary, half-step and quarter
    evolution: symbolic steps must still survive until their normal binding.
    """
    card, lat = C.load_card(), Lattice(6)
    be = T.resolve_backend("grid:4x5", fractional=True)
    emb = T.choose_embedding(be, 6, 2)
    specs = CP.manifest(times=(0.5,), families=("j0",), readouts=("Z",), im=True,
                        dt_half=True, dt_half_times=(0.5,), quarter_times=(0.25,))
    specs += CP.qpdf_specs(card["name"], preset="", ms=(1,))
    snapshots = []
    for _ in range(2):
        pubs, _, bundles = CP.build_pub_circuits(be, lat, card, emb, specs, "rzz",
                                                cache_dir=str(tmp_path))
        validate_rzz_pubs([SamplerPub.coerce((qc, None, 16)) for qc in pubs.values()])
        assert all(not qc.parameters for qc in pubs.values())
        base = bundles[(card["name"], "j0")]
        assert base.base.layout is not None and base.base_layout == base.base.layout.final_index_layout()
        assert {p.name for p in base.step.parameters} == {"t"}
        assert base.step.layout is not None
        assert base.layout_preserved
        assert all(T.count_2q(pubs[s.name]) > 0 for s in specs if s.t == 0)
        for block in base.blocks.values():
            T.assert_skeleton_equal(block["physics"], block["mirror"])
        snapshots.append({name: T.skeleton(qc) for name, qc in pubs.items()})
    assert snapshots[0] == snapshots[1]
