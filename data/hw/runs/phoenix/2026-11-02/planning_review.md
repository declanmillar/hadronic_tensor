# Phoenix planning review, 3 November 2026

These checks use the saved 2 November calibration and acceptance records,
and the preset manifests at `9ae05ad`. They do not submit jobs.

## Cost projection

At an assumed 62 microseconds per shot, the existing equal-shot planner gives:

| Presets | Shots | Minutes |
|---|---:|---:|
| prod-bridge, relA-core, prod-quarter, relA-quarter | 46,380,000 | 47.926 |
| gauss-midcircuit | 1,020,000 | 1.054 |
| prod-im | 6,660,000 | 6.882 |
| relA-im | 15,660,000 | 16.182 |
| Total | 69,720,000 | 72.044 |

The combined Im tier is **23.064 minutes**, not 16 minutes. The full total
includes 4.650 minutes of relA stretch; omitting it gives **67.394 minutes**.
The arithmetic in `campaign.shots_plan` was already correct; no change to
shot allocation is needed to obtain these figures.

The 62-microsecond rate is extrapolated from October's CZ usage. It is not
a measured rate for these RZZ or dynamic circuits, and the projection does
not add a separate allowance for mid-circuit reset overhead.

Reproduce the table from the repository root:

```sh
.venv/bin/python - <<'PY'
from htq_hw import campaign as C

groups = [
    ("Four RZZ presets", ["prod-bridge", "relA-core", "prod-quarter", "relA-quarter"]),
    ("Gauss", ["gauss-midcircuit"]),
    ("prod-im", ["prod-im"]),
    ("relA-im", ["relA-im"]),
]
total = stretch = 0.0
for label, names in groups:
    specs = C.compose_presets(names)
    # Equal-shot allocation does not use the two-qubit counts.
    plan = C.shots_plan(specs, {s.name: 0 for s in specs}, budget_minutes=180,
                        rep_time_s=62e-6, weighting="equal")
    minutes = plan["minutes"][62e-6]
    print(f"{label}: {plan['total']:,} shots, {minutes:.3f} min")
    total += minutes
    stretch += plan["split"]["stretch"]
print(f"Total: {total:.3f} min; without stretch: {total - stretch:.3f} min")
PY
```

## Gate calibration and physical connections

The [saved RZZ acceptance layout](acceptance_phoenix_rzz_live.json) maps the
gauge-link connections at logical wire pairs (65, 66) and (91, 92) to physical
connections 100–101 and 5–15. The inspected compiled preparation and Trotter
blocks matching this layout contain CZ gates, and no RZZ gates, on those
connections. The [calibration snapshot](backend_properties_2026-11-02T08-13-00Z_894bf57b.json)
reports:

| Physical connection | CZ error | RZZ error |
|---|---:|---:|
| 100–101 | 2.50384% | 7.64765% |
| 5–15 | 0.16744% | 3.30378% |

For those CZ operations, the first connection exceeds a 2% error cut and
the second does not. An RZZ run can contain both gate types, but that does
not mean both are applied on every connection. Checking both calibrations
on every edge remains a conservative policy; it can exclude a connection
because of an operation the recorded circuit does not use there.

The proposed cut policy is retained. Relaxing it by connection would require
checking the actual operations for the new candidate layouts and circuits.
This review does not establish that the old layout's gate assignments persist
after a different embedding or compilation.
