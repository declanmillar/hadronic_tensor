# Analysis methods: boost asymmetry and gauge post-selection

Companion notes for the hardware analysis in *"The hadronic tensor from a
quantum computer"*, written for collaborators who want to reproduce or reuse
the two reductions that are easiest to get wrong. Every number below is
reproducible from files in this repository; the commands are given inline.

---

## 1. The boost asymmetry

### 1.1 What it is

A packet boosted to `k0 = 2π/5` responds asymmetrically under `q¹ → -q¹`.
Everything symmetric in the momentum transfer -- the static structure
included -- is removed by the **odd projection**

```
W_odd(q⁰, q¹) = [W(q⁰, q¹) - W(q⁰, -q¹)] / 2
```

so `W_odd` is zero for an unboosted packet and is the kinematic signature the
measurement is after. The `q¹` grid is symmetric about zero by construction
(`ks = arange(-nx//2, nx//2 + 1)`, `scripts/analyze_w_meson.py:56`), so the
projection is a reversal of the last axis.

### 1.2 How it is extracted

A two-dimensional map is not a measurement, so it is reduced to one number by
a **matched filter** against the classical (MPS) reference:

```
A = Σ_R  T · W_odd^meas  /  Σ_R  T · T ,        T = W_odd^ref
```

over the wedge `R : q⁰ ∈ [-1, 2.5]` (all `q¹`). This is the least-squares
amplitude of the measured odd map along the predicted one. **`A = 1` means
the device reproduces the reference asymmetry in full; `A = 0` means no
asymmetry survived.** The template fixes the *shape*, so `A` tests the
amplitude of a predicted structure rather than fitting a free one.

Two conditions matter:

- `W^meas` and `W^ref` must come from the **same transform** -- same times,
  same window `σ_t`, same mask. The template carries the window's own
  distortion, and comparing across windows would fold that into `A`.
- The reference is windowed and masked identically to the data, including the
  `κ < 0.05` site mask, so no site enters the template that the device could
  not measure.

Implementation: `htensor/asymmetry.py`. The published extraction is
`scripts/hw_w00_coarse.py:165-187`, which stores `A_hat`, `A_err`, `A_sys`
into `data/hw_w00_coarse_t3.npz`.

```
PYTHONPATH=. python scripts/boost_asymmetry.py data/hw_w00_coarse_t3.npz
```

```
  wedge q0 in [-1, 2.5], 88 of 176 q0 rows x 25 q1 columns
  A = 0.8493   [A = 1 -> the device reproduces the reference asymmetry in full]
  stored errors: +- 0.3396 (shot) +- 0.2026 (window)  -> 0.3954 total, 2.1 sigma
  from 6 slices, t <= 3, sigma_t = 1.5
```

### 1.3 The two errors, and why they are quoted separately

**Shot error (± 0.34).** The filter is *linear* in the measured slices, so
the shot error propagates exactly: push each measured point `(t_i, x)`
through the transform on its own to get its kernel coefficient
`K_ix = Σ T · odd(FT[δ_ix]) / Σ T·T`, then
`σ_A² = Σ_ix (K_ix σ_ix)²`. No bootstrap, no resampling
(`htensor.asymmetry.shot_error`). This falls as `1/√N`.

**Window systematic (± 0.20).** The one-sided transform needs a time window
`σ_t`, and that choice is not unique. `A` is recomputed with `σ_t` scaled by
`0.75, 1.0, 1.5` and half the peak-to-peak is quoted
(`htensor.asymmetry.window_systematic`). **More shots do not shrink it.** On
the published data it is 60% as large as the shot error, which is why the
next campaign's 12 slices to `t = 6` matter as much as its shot count: a
longer record narrows the window sensitivity itself.

Total `0.395`, so `A = 0.85 ± 0.39`, a **2.1σ** measurement. The projected
phoenix campaign reaches 4–5σ (`scripts/project_campaign.py`).

**Not in either error:** the state-dependent bias in the packet region that
limited the published forward region. It is a systematic that neither error
bar covers, and measuring it is what the dither pubs of the next campaign are
for.

---

## 2. Observable-ranked gauge post-selection

### 2.1 The ranking, and why there is one

The Z₂ Gauss law gives 50 commuting stabilizers on the `Ns = 50` ring,

```
G_n = (-1)^n Z_n X_{L,n-1} X_{L,n} ,      G_n |physical⟩ = +|physical⟩
```

and **all 50 are measured for free** in the Z readout setting (matter in Z,
links in X -- the same setting that carries the `J⁰` probes). A shot that has
left the physical subspace shows `G_n = -1` somewhere, so the sample can be
filtered on its own syndrome, with no extra circuits and no reference to the
classical answer.

Demanding all 50 checks is useless: acceptance collapses far below the shot
floor. So the checks are **ranked by ring distance from the observable's
anchor** and only the top-ranked ones are imposed. For `⟨J⁰(x) J⁰(c)⟩` the
rank of check `n` is `|n - c|` (minimal image, so the seam is an ordinary
neighbour), and `window = w` imposes the `2w+1` checks
`n = c-w … c+w`. The errors that matter for an observable are the ones near
its support; each extra rank costs shots.

Implementation: `htensor/postselect.py` (`check_ranking`, `keep_mask`,
`acceptance`, `cloud_amplitude`). It is the canonical version of logic that
appears inline in `scripts/hw_w00_coarse.py:29-70` and
`scripts/hw_cloud_figure.py:36-61`; `tests/test_postselect.py` asserts the
three agree bit for bit on the released sample.

### 2.2 The trade, measured

On the released 30k-shot `ibm_kingston` equal-time sample
(`data/hw/sq_bits_kingston.npz`), against the exact MPS cloud, using the
matched-filter amplitude over `1 ≤ |x - c| ≤ 6` (the self term at `x = c` is
excluded: it is trivially reproduced and would swamp the fit):

| window | checks imposed | shots kept | fraction | cloud amplitude / exact |
|---|---|---|---|---|
| 0 | none (raw) | 30000 | 100% | 0.253 ± 0.014 |
| 1 | 3 | 11653 | 38.8% | 0.457 ± 0.021 |
| 2 | 5 | 6415 | 21.4% | **0.597 ± 0.026** |
| 3 | 7 | 3613 | 12.0% | 0.667 ± 0.034 |
| 4 | 9 | 2323 | 7.7% | 0.702 ± 0.042 |
| 5 | 11 | 1337 | 4.5% | 0.707 ± 0.055 |
| 6 | 13 | 917 | 3.1% | 0.734 ± 0.067 |
| 7 | 15 | 488 | 1.6% | 0.698 ± 0.091 |

Reproduce with `PYTHONPATH=. python -m pytest tests/test_postselect.py -q`,
or directly through `htensor.postselect.cloud_amplitude`.

The acceptance column doubles as a format check on incoming data: the same
array with its columns permuted accepts 0.072 (window 1) instead of 0.388,
so a wrong logical-to-column mapping shows up as a device that merely looks
noisier. `scripts/check_raw_pubs.py` reports it (see `data/hw/README.md`).

Read off: the raw device recovers a quarter of the connected cloud; window 2
(the published setting) recovers **60%** for a fifth of the shots; the gain
saturates around **70%** by window 4–6, after which the shot error grows
faster than the signal. Window 2–3 is the sweet spot.

> **Note for the manuscript.** The text states the post-selection "un-damps
> the connected screening cloud from ∼30% to ∼80% of its exact depth". The
> ∼30% raw figure is confirmed (0.253 by matched filter, 0.27–0.30 site by
> site). The ∼80% is not reached by these bits at any window: the published
> `PS_WIN = 2` gives 0.60, and the amplitude saturates at ∼0.70. Site by site
> at window 2 the ratios on the significant cloud sites are 0.62, 0.68, 0.62,
> 0.65. Worth reconciling before the next version.

### 2.3 Scope: equal-time only

This is a `t = 0` gain and does not propagate into the assembled tensor. A
`t > 0` circuit's syndrome is measured *after* the evolution, so selecting on
it post-selects the final state rather than projecting the whole trajectory,
and the `t = 0` slice carries no `q⁰` structure. The mid-circuit variant that
*would* propagate -- syndromes extracted every two Trotter steps on
reset-and-reused ancillas -- is the `gauss-midcircuit` preset of the next
campaign (`htq_hw/gauss.py`), gated on dynamic-circuit support.

Related but distinct: the per-shot **syndrome** post-selection in the
Loschmidt study (`scripts/losch2_analyze.py`) selects on ancilla outcomes
rather than on the Gauss sector.

---

## 3. Conventions these two share

- `J⁰(v) = ((-1)^v - Z_v)/2`. Link qubits are read in X, so link bits carry
  the X eigenvalue directly (a Hadamard precedes readout).
- Connected means the one-point product is subtracted **from the same
  sample**, so a post-selection bias largely cancels in the subtraction.
- Positions `x` are quoted in staggered-site units relative to the insertion
  site `c = 24`, minimal-image folded on the ring.
- `W` is assembled by a one-sided windowed transform, with the MPS reference
  windowed and masked identically (`scripts/hw_w_tensor.py`).
