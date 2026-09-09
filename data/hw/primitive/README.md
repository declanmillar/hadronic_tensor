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
