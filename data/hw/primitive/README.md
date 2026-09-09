# `data/hw/primitive/` — primitive-level results, retrieved from IBM

The runtime output as the device returned it, one level below the calibrated
slices in `data/hw/`. Retrieved 2026-09-09 from the saved accounts; every job
was still `DONE` and retrievable.

## Why the tier-3 W⁰⁰ jobs have no per-shot data

The four early kingston tier-3 jobs were submitted with **`EstimatorV2`**, not
`SamplerV2`. An Estimator job returns expectation values, not shots: the
result carries `evs`, `stds` and `ensemble_standard_error` and **no bit
register at all**. So the raw shots for those jobs do not exist — not here,
and not on IBM's side. Nothing was lost to a retention window; per-shot data
was never produced.

| job | primitive | slice | what is here |
|---|---|---|---|
| `d9d3emkinv1c73aoguc0` | estimator | tier3a, t = 0.5 | 2 pubs × 152 evs + stds |
| `d9d84gsinv1c73aomlqg` | estimator | tier3a, t = 0.5 | 2 pubs × 152 evs + stds |
| `d9d45acinv1c73aohsj0` | estimator | tier3b, t = 1.0 | 2 pubs × 152 evs + stds |
| `d9d518sjeosc73fh2prg` | estimator | tier3d, t = 2.0 | 2 pubs × 152 evs + stds |
| `d9610mqf47jc73a51pr0` | estimator | tier1, marrakesh | 1 pub × 152 evs |
| `d96f12l2su3c739gsg0g` | estimator | tier1, fez | 1 pub × 152 evs |
| `d9d71gkinv1c73aolcf0` | sampler | ZNE t = 0 | 3 pubs, packed bit arrays |

`d96106otcv6s73dj7ka0` (tier1, kingston) was CANCELLED and has no result.

These files are still worth having: they are the **uncalibrated** primitive
output, so the mirror calibration (κ, β, the wing anchor) can be redone from
them rather than taken on trust from the `C_cal` in the reduced slices. What
they cannot support is anything that needs individual shots — a different
post-selection window, a resampled error, a per-shot syndrome cut.

## Format

`<primitive>_<job_id>.npz`, with `meta` a JSON string (`job_id`, `account`,
`primitive`, `backend`, `creation_date`, `n_pubs`) and one array per pub field,
named `pub<i>_<field>`: `pub0_evs`, `pub0_stds`, … for estimator jobs;
`pub0_c`, … (packed `BitArray`) for sampler jobs.

Each estimator file also carries, recovered from the job's stored inputs:

- `pub<i>_observables` — a JSON string, one entry per measured observable, each
  a list of `[pauli_string, [re, im]]` terms. This is what makes the values
  interpretable rather than 152 anonymous numbers.
- `pub<i>_n_qubits`, `pub<i>_circuit_ops` — 156 qubits, and the transpiled
  instruction count (10,355 at t = 0.5 up to 19,924 at t = 2.0).

## Observable layout

The 152 observables of a tier-3 pub are, in order,

```
XB_0, YB_0, B_0, XB_1, YB_1, B_1, ..., XB_49, YB_49, B_49, X, Y
```

that is, for each of the 50 probe sites *v*: `<X_anc J⁰(v)>`, `<Y_anc J⁰(v)>`
and `<J⁰(v)>`, then the two bare ancilla terms `<X_anc>` and `<Y_anc>`. Each
`J⁰(v) = ((-1)^v - Z_v)/2` appears as the two-term Pauli sum
`±0.5·I ∓ 0.5·Z_qv`, and the ancilla is **physical qubit 151**; the site
qubits appear in ring-embedding order (42, 44, 46, 47, 50, 51, 70, 69, …),
*not* logical order.

The index-to-site map is already in this repository: `obs_names` in the
matching `data/hw/job_tier3*.json` lists those 152 names in exactly the same
order (verified 1:1). Pair the two and the values are addressable by logical
site without needing the embedding.

**Both ancilla bases were measured.** `YB_v` observables are present, so the
**imaginary part of the correlator exists** for these four jobs even though
the published analysis used the real part only. Nothing downstream has ever
consumed it.

The calibration these feed (κ from the depth-matched mirror, β with the +0.5
anchor, the wing anchor) is in `htensor/analysis.py` and
`scripts/losch_t3_pool.py`; note κ is the mirror value against the *ideal
t = 0* reference, not a mirror-over-physics ratio.

## What is deliberately not here

The sampler jobs behind the later tier-3 slices (t = 0.5 … 3.0, RZZ basis) are
**already in this repository, unpacked**, as `data/hw/losch_bits_*.npz` — one
`(shots, 101)` uint8 array per pub. Archiving the packed form again would have
duplicated 19 MB for nothing. The two are the same data; the mapping is

```python
u = np.unpackbits(packed, axis=-1, bitorder="big")[:, ::-1][:, :101]
u = u[:, ::-1]          # losch_bits store the reversed-column legacy view
```

verified identical on `d9nrg0mij12s73fu81fg`. Likewise the S(q) sampler job
`d974uo2f47jc73a6ahig` is already here unpacked as `sq_bits_kingston.npz`.

Retrieve any of them again with:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService(name="default-ibm-quantum-platform").job("<job_id>").result()
```
