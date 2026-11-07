# IBM Chicago campaign — 6 November 2026

Submission records for the Ns=50 campaign on `ibm_chicago`, with insertion
site 24. The directory date is the submission date. Final acceptance and
submission used code revision `7969166fdac64564df2fe465a82d2b4a5a2b20cc`.
Raw results are shared separately; derived analyses remain local.

| Subcampaign | Basis | Completed jobs | PUB results | Shots | QPU seconds |
|---|---|---:|---:|---:|---:|
| `main_rzz` | RZZ | 127 | 817 | 46,380,000 | 1,776 |
| `im_rzz` | RZZ | 56 | 372 | 22,320,000 | 806 |
| `gauss_cz` | CZ | 3 | 17 | 1,020,000 | 105 |
| Total | | 186 | 1,206 | 69,720,000 | 2,687 |

- `main_rzz`: `prod-bridge`, `relA-core`, `prod-quarter`, `relA-quarter`,
  including the later stretch times recorded in the job metadata.
- `im_rzz`: `prod-im`, `relA-im`.
- `gauss_cz`: `gauss-midcircuit`, a separate mid-circuit syndrome pilot.

All three groups requested measurement twirling and XY4 dynamical decoupling.
Gate twirling was disabled for the RZZ groups and enabled for the CZ pilot.
Layouts used 5% readout and 2% two-qubit calibration cuts, without explicit
Phoenix qubit exclusions. Each job records its actual embedding and options.

## Records

- `jobs/<subcampaign>/htq_job_<id>.json`: submission metadata, including the
  physical layout, PUB names, shot counts, readout mappings and options.
  There are 189 records: 186 completed jobs and three cancelled attempts.
  Exclude records with `cancelled: true` when selecting results. The cancelled
  attempts are main job 0 and two attempts at Im job 29; their replacements
  are included in the completed totals. The budget guard cancelled those
  attempts when estimates were unavailable; retries used `--no-strict-guard`.
- `jobs/main_rzz/submit_2026-11-06T23-25-19Z.log`: the saved main submission log.
- `metrics/<subcampaign>/htq_metrics_<id>.json`: retrieved completion status
  and IBM Runtime execution metrics for each completed job.
- `job_usage.csv`: completed-job counts and `usage.quantum_seconds` from those
  metrics. The totals above exclude cancelled attempts.
- `acceptance_*.json`: all nine acceptance records, including earlier failures.
  The final three start at `21-49-59`, `21-50-02` and `21-50-11` UTC; all pass
  and include Runtime input validation at revision `7969166`. They also record
  package versions. Acceptance results describe checks before submission.
- `raw_sha256.txt`: SHA-256 checksums of the 186 separately supplied raw files,
  with paths relative to this directory.

## Using the raw results

Place the supplied files at `raw/<subcampaign>/htq_bits_<id>.npz`, preserving
the three subdirectories. Keep the Gauss pilot separate from the main/Im data
when reducing results: it shares the production card but is a different
experiment. Raw NPZs contain `job_id`, `backend`, `pub_names`, and one binary
shot array per PUB. System columns are in logical-wire order; Gauss PUBs also
carry syndrome columns. The corresponding job JSON records each PUB's width
and logical mapping.

From this directory, verify the transfer with:

```sh
shasum -a 256 -c raw_sha256.txt
```

The existing collection helper generated the metrics and usage CSV. From the
repository root, verify PUB names, shots, bit-array shapes and job identities
against the metadata without contacting IBM or changing files:

```sh
.venv/bin/python scripts/collect_hw_results.py \
  data/hw/runs/chicago/2026-11-06 --check-only
```

Use `.venv/bin/python -m htq_hw analyze` for an independent reduction. The two
cards and their ideal grids, `htq_hw/refs/wing_surrogate_{prod,relA}.npz`, and the four MPS references
`data/w_{prodR,relA}_ns50_j{0,1}_k1.26_s0.75.npz` are already tracked in the
repository. No files under `analysis/`, including `gauss_rounds/` and
`gauss_unfiltered/`, are needed to start from the raw results.
