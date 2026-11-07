"""Collect completed htq_hw jobs and recheck their bits without submitting jobs.

Usage: .venv/bin/python scripts/collect_hw_results.py data/hw/runs/chicago/2026-11-06
Add --check-only to verify existing files offline. Usage totals cover completed
jobs only; canceled-attempt metadata is retained but excluded from the CSV.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from htq_hw import campaign as CP


def validate_bits(path, meta):
    names = meta["pub_names"]
    with np.load(path, allow_pickle=False) as bits:
        if set(bits.files) != set(names) | {"job_id", "backend", "pub_names"}:
            raise ValueError(f"{path}: pub keys do not match metadata")
        if bits["pub_names"].tolist() != names:
            raise ValueError(f"{path}: pub order does not match metadata")
        for key in ("job_id", "backend"):
            if bits[key].item() != meta[key]:
                raise ValueError(f"{path}: {key} does not match metadata")
        for name in names:
            info = meta["pubs"][name]
            shape = (int(meta["shots"][name]), int(info.get("n_clbits", meta["n_logical"])))
            arr = bits[name]
            if arr.shape != shape or arr.dtype != np.uint8 or np.any(arr > 1):
                raise ValueError(f"{path}: {name} must be binary uint8 with shape {shape}")
            mapping = info.get("clbit_to_logical")
            if mapping is not None and mapping != list(range(int(meta["n_logical"]))):
                raise ValueError(f"{path}: {name} has a nonidentity logical-bit map")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="verify saved files without API access")
    args = parser.parse_args()
    selected, indices, ids, skipped = [], set(), set(), 0
    paths = sorted((args.run_dir / "jobs").glob("*/htq_job_*.json"))
    if not paths:
        raise ValueError(f"{args.run_dir}: no job metadata found")
    for path in paths:
        meta = json.loads(path.read_text())
        if meta.get("cancelled") is True:
            skipped += 1
            continue
        if meta.get("cancelled") is not False or meta.get("guard") == "pending":
            raise ValueError(f"{path}: pending or ambiguous cancellation state")
        key = (path.parent.name, meta["job_index"])
        if key in indices or meta["job_id"] in ids:
            raise ValueError(f"{path}: duplicate kept campaign/index or job ID")
        if len(set(meta["pub_names"])) != len(meta["pub_names"]):
            raise ValueError(f"{path}: duplicate pub names")
        indices.add(key)
        ids.add(meta["job_id"])
        selected.append((key, meta))
    if not selected:
        raise ValueError("no noncancelled jobs to collect")

    service, rows = None, []
    for (campaign, index), meta in sorted(selected, key=lambda item: item[0]):
        jid = meta["job_id"]
        raw = args.run_dir / "raw" / campaign / f"htq_bits_{jid}.npz"
        metrics_path = args.run_dir / "metrics" / campaign / f"htq_metrics_{jid}.json"
        if raw.exists():
            validate_bits(raw, meta)  # Never replace a downloaded result that fails validation.
        if args.check_only and (not raw.exists() or not metrics_path.exists()):
            raise ValueError(f"{campaign}/{index}: missing bits or metrics; collect first")
        if not args.check_only and (not raw.exists() or not metrics_path.exists()):
            if service is None:
                from qiskit_ibm_runtime import QiskitRuntimeService
                service = QiskitRuntimeService()
            job = service.job(jid)
            status = job.status()
            status = getattr(status, "name", str(status)).split(".")[-1].upper()
            if status != "DONE":
                raise ValueError(f"{campaign}/{index} ({jid}): status {status}, expected DONE")
            if job.job_id() != jid or job.backend().name != meta["backend"]:
                raise ValueError(f"{jid}: cloud job identity/backend does not match metadata")
            if not raw.exists():
                CP.fetch(job, meta, str(raw.parent))
                validate_bits(raw, meta)
            if not metrics_path.exists():
                snapshot = {"job_id": jid, "backend": meta["backend"], "status": status,
                            "metrics": job.metrics()}
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("x") as handle:
                    json.dump(snapshot, handle, indent=2, default=str)
                    handle.write("\n")
        snapshot = json.loads(metrics_path.read_text())
        if any(snapshot.get(k) != meta[k] for k in ("job_id", "backend")) or snapshot.get("status") != "DONE":
            raise ValueError(f"{metrics_path}: job identity/backend/status does not match")
        seconds = ((snapshot.get("metrics") or {}).get("usage") or {}).get("quantum_seconds")
        if seconds is not None:
            seconds = float(seconds)
            if not np.isfinite(seconds) or seconds < 0:
                raise ValueError(f"{metrics_path}: invalid measured QPU usage")
        rows.append(dict(campaign=campaign, job_index=index, job_id=jid, backend=meta["backend"],
                         status="DONE", pubs=len(meta["pub_names"]), shots=sum(meta["shots"].values()),
                         qpu_seconds=seconds))
        print(f"{campaign} job {index}: {len(meta['pub_names'])} pubs checked", flush=True)

    if not args.check_only:
        with (args.run_dir / "job_usage.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)  # None is an empty CSV field, never an assumed zero.
    print(f"Completed jobs only; {skipped} canceled attempts excluded from bits and usage CSV.")
    for campaign in sorted({row["campaign"] for row in rows}):
        group = [row for row in rows if row["campaign"] == campaign]
        known = [row["qpu_seconds"] for row in group if row["qpu_seconds"] is not None]
        usage = f"{sum(known):.3f} s known" if known else "unknown"
        missing = len(group) - len(known)
        print(f"{campaign}: {len(group)} jobs, {sum(r['pubs'] for r in group)} pubs, "
              f"{sum(r['shots'] for r in group):,} shots; QPU usage {usage} ({missing} missing)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"collection failed: {exc}") from exc
