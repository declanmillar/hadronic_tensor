# Phoenix tensor comparison with dt = 0.5 MPS

The existing calibrated hardware slices are reassembled using the new MPS
packet and vacuum grids in `data/mps/runs/phoenix/2026-10-13_12-05-21_572598Z/`.
Both hardware and MPS use dt = 0.5. The reference uses chi_max = 32 and
truncation threshold 1e-5 for preparation and evolution. These are exploratory
references; MPS convergence has not been established.

| Point | Full tensor | Even | Odd | Cuts |
|---|---|---|---|---|
| prodR | [PDF](prod_k1.26_s0.75_ns50/w_tensor_t6.pdf) | [PDF](prod_k1.26_s0.75_ns50/w_tensor_t6_even.pdf) | [PDF](prod_k1.26_s0.75_ns50/w_tensor_t6_odd.pdf) | [PDF](prod_k1.26_s0.75_ns50/w_tensor_t6_cuts.pdf) |
| relA | [PDF](relA_k1.26_s0.75_ns50/w_tensor_t6.pdf) | [PDF](relA_k1.26_s0.75_ns50/w_tensor_t6_even.pdf) | [PDF](relA_k1.26_s0.75_ns50/w_tensor_t6_odd.pdf) | [PDF](relA_k1.26_s0.75_ns50/w_tensor_t6_cuts.pdf) |

Each figure also has a PNG. The NPZ, log and command JSON accompany each
point. [Summary](summary.json) records the reference settings and metrics.

The Fourier window remains t <= 6. Real-part projection, site masks,
charge-conservation projection, momentum cuts and uncertainty prescription
are unchanged. All six component masks and time grids were checked against
the previous analysis; all assembled tensors are finite.

Hybrid subtraction still uses MPS vacuum correlators and, where required,
MPS one-point functions. Changing these references therefore also changes
the hardware-derived tensor. Calibration was not refitted: the existing
calibrated slices and their ideal calibration inputs were retained.
No new post-selection was applied.

W00 at both points and relA W01 still have saved-error bounds larger than
their peak hardware signals. Bands combine the existing diagonal error
bound with window variation; they do not include full covariance or MPS
uncertainty. Matching dt does not remove finite-step error relative to exact
evolution. Differences from the old comparison also include the change in
MPS accuracy settings, so they do not isolate Trotter error.

The [qPDF scan](../qpdf/README.md) and calibration diagnostics remain those
of the [original analysis](../README.md). The qPDF has no real-time evolution
and is unaffected by these new tensor-reference grids. All earlier figures
and the dt = 0.1 references remain available.
