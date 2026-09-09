# `data/hw/` — hardware results

Two kinds of file live here, and the difference matters if you are sending
data in.

## What is here now

**Reduced slices** (`job_*.npz`) — one time slice, already calibrated:
`times, probes, C, C_cal, C_err, kappa_v, beta_v, b_cal, id_a, c_a, id_b,
tier, backend, nshot, merged_jobs`. `C` is raw, `C_cal` mirror-calibrated;
`kappa_v` is the per-site damping from the same-job mirror. These are the
inputs to `scripts/hw_w_tensor.py`, which assembles `W^{μν}`.

**Job metadata** (`job_*.json`) — `job_id, tier, backend, shots, circ_names,
obs_names, circ_meta`.

**Raw bitstrings** — the per-shot samples, which are what support re-analysis
(a different post-selection window, a different calibration) without
re-running the device:

- `sq_bits_kingston.npz` — the equal-time S(q) sample, `(30000, 100)` uint8,
  column *i* = logical qubit *i*. This is what the gauge post-selection of
  `docs/METHODS.md` runs on.
- `losch_bits_*.npz` — nine jobs, `(shots, 101)` uint8 per pub, keyed by slice
  (`t0.5`, `m0.5`, `l1.0`, …). These carry the shots behind the tier-3 slices
  at t = 0.5, 1.0, 1.5, 2.0 and 3.0 in the RZZ basis. Note the column order is
  the reversed legacy view (see `primitive/README.md`).

**Primitive-level results** (`primitive/`) — the runtime output as the device
returned it, retrieved from the IBM accounts. Read that directory's README
first: the four earliest tier-3 kingston jobs were submitted with
`EstimatorV2`, so they returned expectation values and **no shots exist for
them at all**, on IBM's side or ours. Their `evs`/`stds` are archived there,
which still allows the mirror calibration to be redone independently.

## Sending raw pub results — yes, please

Reduced slices cannot be re-analysed; raw shots can. If you have per-shot
results from the kingston runs, they are worth adding. Put them in
`data/hw/raw/` in the format the current campaign package already reads and
writes, so no bespoke loader is needed:

**`htq_bits_<job_id>.npz`**

| key | type | meaning |
|---|---|---|
| `job_id` | str | the IBM job id |
| `backend` | str | e.g. `ibm_kingston` |
| `pub_names` | str array | one name per pub, in submission order |
| *one array per pub name* | uint8 `(shots, n_logical)` | **column *i* = logical qubit *i*** |

**`htq_job_<job_id>.json`** — whatever the submission recorded, at minimum:
`job_id, backend, shots, pub_names`, and if you have them: `ns`, `center`,
`initial_layout` (logical wire → physical qubit), `basis`, the per-pub
readout basis map, and the `clbit_to_logical` map.

The one thing that must be right is the **column convention**: logical wire
*i* in column *i*, with matter site *v* at wire `2v`, link *(v, v+1)* at wire
`2v+1`, ancilla last. If your arrays are in physical-qubit order, send the
layout and say so rather than permuting them yourself — the mapping is easy
to get backwards, and a silent transposition of the staggered sublattices
looks like physics.

Check a file before sending:

```
PYTHONPATH=. python scripts/check_raw_pubs.py data/hw/raw/htq_bits_<id>.npz
```

It reports shape, shots, the implied `Ns`, per-pub means, and the Gauss-law
acceptance — the quickest test that the column convention is right. On the
released kingston sample the correctly ordered array accepts 0.388 (window 1)
and 0.214 (window 2); the same array with its columns permuted gives 0.072
and 0.016. A wrong order does not fail loudly, it just looks like a noisier
device, which is exactly why it is worth checking before the data are used.

If the readout basis is not Z (matter Z, links X), say which setting each
pub used; the Gauss check only applies to the Z setting.

**Size.** Uncompressed, 30k shots × 101 qubits is 3 MB per pub. `savez_compressed`
takes that to a few hundred kB. This is a public repository: anything over
~50 MB total should go in a GitHub release asset or git-lfs rather than the
tree — send it and we will place it.

## Provenance

Whatever you send, include the device, the date, the shot count, and whether
any mitigation was applied by the runtime (twirling, DD, readout mitigation).
`docs/METHODS.md` documents what the analysis assumes about the raw sample.
