# ibm_phoenix records from Declan, 2026-11-02

- `acceptance_phoenix_rzz_live.json`: live full gate, rzz basis, presets
  prod-bridge relA-core prod-quarter relA-quarter, `--exclude-qubits
  17,62,77`, at commit 5f4c95c (clean tree).  WARN only for the budget
  (193.2 min at the assumed 250 us/shot).  Layout: redundancy 8, 19 spare.
- `acceptance_phoenix_gauss_midcircuit_live.json`: live full gate, CZ basis,
  preset gauss-midcircuit, same exclusions, same commit.  PASS.
- `backend_properties_2026-10-06T20-19-54Z_57f0703f.json`: the calibration
  snapshot returned for the 6 October campaign jobs.
- `backend_properties_2026-11-02T08-13-00Z_894bf57b.json`: the 2 November
  calibration.
- The 6 October per-job QPU usage is in `../2026-10-06/job_usage.csv`
  (instance identifier column removed): 1697 s over 118 jobs and 38.01 M
  shots, 44.6 us/shot on average, 58 us/shot on the deepest (relA stretch).

`scripts/phoenix_snapshot_backend.py` rebuilds an offline phoenix from a
snapshot (FakeNighthawk configuration, same 218 couplers, with rzz); the
embedding it gives for the rzz gate's settings is identical to the live one.

The [3 November planning review](planning_review.md) gives reproducible
62-microsecond cost projections and distinguishes the CZ and RZZ calibration
values on the connections used by the recorded layout.
