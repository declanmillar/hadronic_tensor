# Phoenix hardware quasi-PDF reduction

Eight cards (six packet widths and two matching vacua), 168 circuits and 3,360,000 shots. Each circuit has 20,000 shots. All six packet results use measured h(0) and the matching coupling-point vacuum subtraction.

| Coupling | Width | Hardware first moment | MPS first moment | Norm / MPS norm |
|---|---:|---:|---:|---:|
| prod | 0.75 | 0.4009 +/- 0.0051 | 0.3891 | 0.803 |
| prod | 1.00 | 0.3793 +/- 0.0064 | 0.3728 | 0.776 |
| prod | 1.50 | 0.3942 +/- 0.0085 | 0.3463 | 0.850 |
| relA | 0.75 | 0.3983 +/- 0.0058 | 0.3500 | 0.736 |
| relA | 1.00 | 0.4001 +/- 0.0066 | 0.3296 | 0.795 |
| relA | 1.50 | 0.4008 +/- 0.0092 | 0.2989 | 0.810 |

Errors in this table retain covariance between positive and negative separations measured in the same circuit. They describe shot noise only; hardware systematic errors and MPS uncertainty are not included. The original CLI reductions retain the original independent-amplitude bootstrap errors. Widths at one coupling share a vacuum sample, so their errors are correlated; no extrapolated intercept is fitted.

All normalization integrals are positive, 51 to 97 shot standard errors from zero. Hardware integrals are 15 to 26 percent smaller than the ideal MPS values. Normalized plots therefore hide a material reduction in signal amplitude; inspect the unnormalized figure too.

The production widths 0.75 and 1.00 are within 2.4 shot standard errors of MPS in their first moments; width 1.50 differs by 5.6. All relA widths differ by 8.3 to 11.1 shot standard errors. The hardware relA first moments are nearly flat around 0.400, whereas the MPS reference decreases from 0.350 to 0.299. These differences exceed shot noise, but their physical cause is not established by this reduction.

Conventions: h(m)=A(2m), m=-5,...,5; staggered taste phase (-1)^m; Gaussian sigma_m=5; k0=2*pi/5; normalize and take the moment over x in [-0.5,1.5]. Existing qPDF reduction and reference conventions agree numerically. No tensor mirror calibration or assignment-matrix correction is applied to these prep-only data.

Outputs: reduction.log, command.json, six qpdf_*.npz reductions, six diagnostic_*.npz covariance products, summary.json, and normalized/unnormalized/first-moment figures in PNG and PDF.
