# Data release — Hadronic tensor of 1+1d Z₂ lattice gauge theory on 101 qubits

Companion data for the paper (Fermilab, 2026). Every `.npz` array here was
produced by a script in `scripts/` of this repository; the producing script
and physics conventions are noted per group below. Plain-text `.csv`
versions of the headline tables are in `csv/`. Figures (`*.pdf`) are the
paper figures rendered from exactly these arrays.

## Physics conventions (apply to all files)

- **Model.** Z₂ gauge theory with staggered fermions on a periodic ring of
  `ns` staggered sites (`nx = ns/2` spatial sites), explicit link qubits.
  Hamiltonian `H = (g2/2) Σ σˣ_link − (m0/2) Σ (−1)ⁿ Zₙ + (η/4) Σ (XₙXₙ₊₁ +
  YₙYₙ₊₁) σᶻ_link`. Production couplings `(m0, g2, η) = (0.7, 1.1, 1.3)`
  unless a filename tags a comparison point (`dhk`, `cgkel`, `cgkinA`,
  `cgkinB`; see below). All quantities in lattice units; spatial positions
  `x = (v − v_c)/2`, minimal-image folded on the ring.
- **Circuits.** 101 qubits = 50 site + 50 link + 1 Hadamard-test ancilla,
  Qiskit Aer matrix-product-state backend. Correlators are single-ancilla
  Hadamard-test measurements `C^{0ν}(v,t) = ⟨ψ| J^ν(v,t) J⁰(v_c,0) |ψ⟩`,
  x-multiplexed over all probes per circuit.
- **Subtractions.** Connected correlators subtract the disconnected product
  `⟨J^ν(v,t)⟩⟨J⁰(v_c)⟩` pointwise; wavepacket runs additionally subtract the
  matched vacuum grid (connected-minus-connected). Production correlator
  files carry `idfix=1/2`: the assembly correction (2026-07-07) restoring
  `⟨A(0)⟩` (not `⟨A(t)⟩`) in the probe-identity term.
- **Fourier transforms.** Vacuum tensors use the completed-time (hermiticity)
  transform; wavepacket ridge physics uses **one-sided** windowed transforms
  on the raw time-domain files (Gaussian window `σ_t = 8/3`, spatial
  `σ_x = nx/6`). The `*_W.npz` files store the windowed `W^{00}(q0,q1)` grids.

## Comparison-coupling translation (their conventions → ours, η=1)

| tag | reference | (m0, g2, η) |
|-----|-----------|-------------|
| `dhk` | Davoudi–Hsieh–Kadam, arXiv:2505.20408 | (1.0, 0.6, 1.0) |
| `cgkel` | Chai–Guo–Kühn elastic, arXiv:2505.21240 | (0.1, 2.0, 1.0) |
| `cgkinA` | CGK inelastic A | (0.1, 0.4, 1.0) |
| `cgkinB` | CGK inelastic B | (0.2, 0.4, 1.0) |

## Device data versus simulation

Some `.npz` files here hold **per-shot bitstrings** and it is not always
obvious from the name whether the shots came from a quantum computer or from
a noisy simulator. The distinction:

| file | source | what it is |
|---|---|---|
| `hw/sq_bits_kingston.npz` | **ibm_kingston** | 30,000 shots, equal-time S(q) / screening cloud |
| `hw/losch_bits_*.npz` (9) | **ibm_kingston** | 7k–70k shots per pub, 29 pubs; the shots behind the tier-3 slices at t = 0.5 … 3.0 |
| `hw/primitive/sampler_*.npz` | **ibm_kingston** | the ZNE t = 0 job as IBM returned it (packed) |
| `hw/primitive/estimator_*.npz` | **ibm_kingston / marrakesh / fez** | expectation values, not shots — see `hw/primitive/README.md` |
| `2p_bits_g{0,1,2}_k1.26.npz` | **AerSimulator** | 80,000 simulated shots per gate-noise scale g, used to fit the two-parameter noise model against the measured S(q) curve (`scripts/hw_sq_2param.py`) |

Everything under `hw/` is device data. `2p_bits_*` is simulation despite
sitting alongside it. Local Aer dry-run samples (`tmp_losch2_*`) are scratch
and are deliberately not released.

## File groups

**Production hadronic tensor (101 qubits).**
`w_meson_ns50_k{0.00,1.26}_v3.npz` — W⁰⁰ on rest / boosted (k̄=2π/5) meson
wavepackets, `_v3_W.npz` the windowed grids, `_v3_axial.npz` the axial
partner. `w_meson_j1_ns50_k*.npz` — J¹-insertion variant (Ward).
`w00_vac_ns50*.npz`, `sigma_vac_ns50.npz` — vacuum polarization and optical
conductivity. Producers: `scripts/run_w_meson_ns50.py`, `run_w_j1_ns50.py`,
`run_w00_vacuum_ns50.py`, `kubo_conductivity.py`.

**Finite-volume spectra (gauge-fixed exact diagonalization).**
`deep_levels[_<tag>]_ns{8..20}.npz` — gaps, T2 momenta, reflection parities,
energies at 7 volumes for the production and four comparison couplings.
`phase_shifts_6vol.npz` — 16 parity-verified elastic δ(p) points.
`two_channel_fit.npz` — coupled MM/MM′ fit + odd-sector phases.
`predict_tables.npz` — bound states + predicted finite-volume levels per
coupling. Producers: `scripts/deep_levels_gf.py`, `phase_shifts_v2.py`,
`two_channel.py`, `predict_tables.py`.

**Scattering / structure bridges.**
`factorization_cgkA.npz` — Lellouch–Lüscher collapse (|F(E)|² across 5
volumes). `duality_sumrules.npz` — parton f-sum-rule saturation.
`dhk_Rt_NP{5,13}.npz` — exact return probability (20.8M-state ED at N_P=13).
`quasipdf_ns50_k1.26.npz`, `quasipdf_x.npz`, `quasipdf_matching.npz` —
101-qubit quasi-PDF and its x-space / volume study. Producers:
`scripts/factorization.py`, `duality.py`, `dhk_postdiction.py`,
`quasipdf_ns50.py`, `quasipdf_analysis.py`, `quasipdf_matching.py`.

**Synthesis-error study.**
`synthladder_{exact,stoc_pi*,clifford}.npz` — 101-qubit ridge vs rotation-
grid spacing, exact → stochastic-snapped (3 seeds) → Clifford endpoint.
`synthesis_pilot_ns8*.npz` — 17-qubit pilot (snapping + true gridsynth).
`error_budget.npz`, `sysscan_*.npz` — one-knob MPS/Trotter systematics.
Producers: `scripts/synthesis_ladder_ns50.py`, `clifford_endpoint_ns50.py`,
`synthesis_pilot_ns8.py`, `error_budget.py`, `lenore_batch.py`.

**State preparation.**
`wp10reg_params_k*.npz`, `wpCGKA_params_k1.26_L3.npz` — trained wavepacket
block parameters. `cgkA_vacuum_thetas.npz` — CGK-A vacuum angles.

**Hardware.**
`hw/job_tier1_*.npz` — tier-1 preparation certificates executed on
`ibm_marrakesh` and `ibm_fez` (Heron r2, 101 qubits): Gauss-law witnesses,
charge profile, ⟨H⟩, parity, total charge.

## csv/

Plain-text long-format versions of the headline tables (see per-file header
comments): vacuum & wavepacket W⁰⁰ grids, σ(ω), meson dispersion, 6-volume
and odd-channel phase shifts, factorization collapse, duality sum rules,
prediction tables, and the synthesis ladder. Regenerate with
`PYTHONPATH=. python scripts/export_tables.py`.
