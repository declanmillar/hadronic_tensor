# Phoenix hardware analysis

The directory date, October 6, 2026, is the hardware submission date.

## MPS inputs

These tracked reference files remain at their repository-relative paths:

| Point | Insertion | Input | Evolution step | Saved time spacing |
|---|---|---|---:|---:|
| prodR | J0 | `data/w_prodR_ns50_j0_k1.26_s0.75.npz` | 0.1 | 0.5 |
| prodR | J1 | `data/w_prodR_ns50_j1_k1.26_s0.75.npz` | 0.1 | 0.5 |
| relA | J0 | `data/w_relA_ns50_j0_k1.26_s0.75.npz` | 0.1 | 0.25 |
| relA | J1 | `data/w_relA_ns50_j1_k1.26_s0.75.npz` | 0.1 | 0.25 |

All four grids cover t = 0 through 8. The main Fourier comparisons use
hardware samples through t = 6. [File and reference checksums](archive.json)
identify the saved outputs and MPS inputs.

Future dt = 0.5 MPS calculations and comparisons belong in new run
directories, with explicit reference inputs. The plotting scripts protect
existing outputs from being overwritten. Some linked audits live in ignored
`data/work/` and are only available in this checkout.

## Results

All 118 fetched jobs have been analysed: 206 expected tensor slices and six vacuum-subtracted qPDF distributions. All expected slice times and source-job identities match the submission records, including 70 slices recovered by the split-XY reducer fix. The data show substantial calibration and hardware limitations; complete coverage does not establish quantitative agreement with MPS.

| Dataset | Main components | Time coverage | Extra controls | Saved slices |
|---|---|---|---|---:|
| Production packet, width 0.75 | 00, 10 | 0–8 in steps of 0.5 | 00d, 10d at all 17 times | 68 |
| relA packet, width 0.75 | 00, 01, 10, 11 | 0–8 in steps of 0.5 | 00d, 10d at positive times; all six components at t=0.5,1 with dt=0.25 | 112 |
| Production vacuum | 00 | 0–6 in steps of 0.5 | — | 13 |
| relA vacuum | 00 | 0–6 in steps of 0.5 | — | 13 |

The production 01/11 insertion measurements were not part of the campaign. No values for those components were manufactured. The qPDF width scan uses widths 0.75, 1.00 and 1.50 at both coupling points, with the two matching vacuum cards: 168 circuits, 20,000 shots each. The full campaign totals 821 circuit results and 38,010,000 shots.

**Tensor results and plots.** Primary Fourier comparisons use t≤6, the planned main window. They match the real-part projection, time samples, Fourier windows and site masks between hardware and MPS. Even/odd means reflection q¹→−q¹. The horizontal axis is q⁰, vertical axis q¹; cuts use q¹≈0.754, 1.257 and 2.262. Colours/fonts use the existing paper style.

| Coupling | Full tensor | Even | Odd | Cuts with uncertainty |
|---|---|---|---|---|
| Production | [PNG](prod_k1.26_s0.75_ns50/w_tensor_t6.png) | [PNG](prod_k1.26_s0.75_ns50/w_tensor_t6_even.png) | [PNG](prod_k1.26_s0.75_ns50/w_tensor_t6_odd.png) | [PNG](prod_k1.26_s0.75_ns50/w_tensor_t6_cuts.png) |
| relA | [PNG](relA_k1.26_s0.75_ns50/w_tensor_t6.png) | [PNG](relA_k1.26_s0.75_ns50/w_tensor_t6_even.png) | [PNG](relA_k1.26_s0.75_ns50/w_tensor_t6_odd.png) | [PNG](relA_k1.26_s0.75_ns50/w_tensor_t6_cuts.png) |

Every plot also has a PDF alongside it. Its NPZ stores actual component times, masks, Fourier/window arrays and source paths; the adjacent `_command.json` and `.log` record the command and analysis choices.

The propagated-error bound exceeds the peak hardware tensor for production 00 and relA 00/01. Their heatmap shapes therefore do not establish a resolved signal. The bound is computed from the saved site errors using a diagonal approximation; the displayed bands additionally contain Fourier-window variation. It is not a full covariance or systematic confidence interval.

| Coupling | Component | Peak abs(hardware W) | Saved-error bound |
|---|---|---:|---:|
| prod | 00 | 0.883 | 1.501 |
| prod | 10 | 0.359 | 0.091 |
| relA | 00 | 0.547 | 0.914 |
| relA | 01 | 0.643 | 1.166 |
| relA | 10 | 0.620 | 0.122 |
| relA | 11 | 1.683 | 0.512 |

The relA q-space Ward diagnostics are 0.588 and 0.835 versus matched-MPS values 0.253 and 0.141. These are descriptive finite-window diagnostics without complete uncertainty propagation, not a precision Ward-identity test.

**Calibration and controls.** [Calibration/coverage figure](calibration_diagnostics.png) ([PDF](calibration_diagnostics.pdf)). Five site/slice points have zero mirror ratios: production-vacuum site 16 at t=2 and 3.5; relA site 12 at t=5 in 00, 00d and 01. These points are already masked; every retained calibrated value/error is finite. Finite-safe masking was added to the existing tensor plotter so rejected NaNs cannot contaminate a Fourier transform.

The J1 mirrors have only three strong-reference bonds per term (22, 25, 27); other bonds use the existing guarded median fill. The center value is filled, not an independent center fidelity. Small beta denominators at some retained J0 sites amplify errors: the relA 00 median error reaches 0.334 at t=1. The existing kappa mask alone does not reject small beta. No extra beta cuts were introduced.

Mean measured Gauss signals fall from 0.549 to 0.126 for the production packet and 0.517 to 0.123 for relA between t=0 and 8. These diagnostics include readout errors and are not preparation fidelities. No Gauss postselection was applied. Late-current coverage deteriorates: relA 10 and 10d at t=7.5 retain only 1/50 sites. All late-time and dt=0.25 control slices are preserved.

The retained-site beta diagnostic finds up to four production 00 sites and two relA 00/01 sites per slice with abs(beta)<0.05. Individual saved errors reach 2.81 (production 00), 1.95 (relA 00), and 7.33 (relA 01), even where the median error looks moderate.

The half-step controls remain separate from the Fourier transform. The table compares dt=0.25 and dt=0.5 on shared retained sites, using the RMS of differences divided by saved errors added in quadrature. These are joint Trotter, hardware-noise and calibration differences; the nominal pulls omit full covariance and do not provide a Trotter correction.

| relA component | Nominal pull RMS, t=0.5 | Nominal pull RMS, t=1 |
|---|---:|---:|
| 00 | 2.07 | 1.10 |
| 10 | 2.94 | 1.72 |
| 01 | 1.19 | 1.18 |
| 11 | 0.74 | 0.83 |

The t=0.5 current-probe difference is not negligible under the saved-error model. [Supplemental beta and half-step diagnostics](../../../../../work/phoenix-analysis/supplemental-diagnostics.json) also contain the dither controls.

**qPDF width scan.** [Moments](qpdf/qpdf_moments.png), [normalized distributions](qpdf/qpdf_distributions.png), and [unnormalized distributions](qpdf/qpdf_unnormalized.png) are available in PNG and PDF. The displayed errors include covariance of +m/−m observables measured in the same circuit. Full reduction details are in [the qPDF report](qpdf/README.md).

| Coupling | Packet width | Hardware first moment ± shot error | MPS |
|---|---:|---:|---:|
| prod | 0.75 | 0.4009 ± 0.0051 | 0.3891 |
| prod | 1.00 | 0.3793 ± 0.0064 | 0.3728 |
| prod | 1.50 | 0.3942 ± 0.0085 | 0.3463 |
| relA | 0.75 | 0.3983 ± 0.0058 | 0.3500 |
| relA | 1.00 | 0.4001 ± 0.0066 | 0.3296 |
| relA | 1.50 | 0.4008 ± 0.0092 | 0.2989 |

All qPDF normalization integrals are positive by 51–97 shot standard errors, so unstable denominators do not explain the trend. Their magnitudes are 15–26% below MPS. relA moments remain near 0.40 instead of following the predicted decrease with packet width; discrepancies are 8–11 shot standard errors. The production width-1.50 moment also differs beyond shot noise. These errors exclude hardware systematics and reference uncertainty, and shared vacuum data correlate the widths. No width extrapolation was fitted.

**Method limits.** The tensor plots follow the existing hybrid subtraction: MPS vacuum correlators and, for current probes, MPS one-point functions are used; the measured vacuum controls are reduced separately. The existing charge-conservation projection is retained for 00/01. Hardware dt=0.5 and MPS dt=0.1 still differ: matching time samples/windows/masks does not correct Trotter error, and no tensor Trotter uncertainty has been added. Tensor errors also omit full spatial/shared-mirror covariance and hardware systematics. These are initial diagnostic results.

**Reproducibility.** [Summary JSON](summary.json), [tensor reduction parameters and output list](tensor-reduction.json), [exact tensor reduction commands](tensor-commands.json), [coverage and calibration audit](../../../../../work/phoenix-analysis/reduction-audit.json), and [input-file verification](../../../../../work/phoenix-fetch/audit-20261006T224308Z.json). Per-card reduction logs accompany the slices. Existing `htq_hw` reduction and `scripts/hw_w_tensor.py` were used; no new analysis scripts were added.
