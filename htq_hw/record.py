"""Machine-readable records of what was validated, and when.

Until now the only evidence that `check` had ever passed was prose in a notes
file.  A deliverable handed to someone else needs the evidence to travel with
it: which environment, which device, which embedding, which grids (by hash),
and what the numbers were.  `bundle` refuses to build without a passing
record, so these are gates rather than logs.
"""

import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _run(*cmd) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                             cwd=str(pathlib.Path(__file__).resolve().parent.parent))
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def git_stamp() -> dict:
    """Commit, tag and dirty flag of the tree the package is being built from."""
    commit = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain", "--", "htq_hw")
    return {"commit": commit, "tag": _run("git", "describe", "--tags", "--always"),
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "dirty_files": (status or "").splitlines()[:20]}


def env_stamp() -> dict:
    from . import PINNED, __version__, check_versions
    mods = {}
    for mod in PINNED:
        try:
            mods[mod] = __import__(mod).__version__
        except Exception:
            mods[mod] = None
    bad = check_versions()
    return {"htq_hw": __version__, "python": platform.python_version(),
            "platform": platform.platform(), "modules": mods, "pinned": PINNED,
            "pinned_ok": not bad, "pin_mismatches": bad}


def backend_stamp(be) -> dict:
    """Everything about the target that a result depends on, including whether
    it was actually a real device."""
    from . import target as T
    out = {"name": getattr(be, "name", str(be)), "label": T.backend_label(be),
           "is_real": T.is_real_backend(be),
           "standin_for": getattr(be, "_htq_standin_for", None),
           "twin_of": getattr(be, "_htq_twin_of", None)}
    for attr, fn in (("num_qubits", lambda: be.num_qubits),
                     ("n_edges", lambda: len(be.coupling_map.get_edges()) // 2),
                     ("basis_gates", lambda: sorted(be.operation_names)),
                     ("instance", lambda: str(be.instance)),
                     ("last_update_date", lambda: str(be.properties().last_update_date))):
        try:
            out[attr] = fn()
        except Exception:
            pass
    return out


def file_hashes(root, suffixes=(".py", ".json", ".npz", ".md", ".txt")) -> dict:
    """sha256 of every shipped file, so a bundle can be checked against the
    record it claims to satisfy."""
    root = pathlib.Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def write_record(path, kind: str, payload: dict, status: str) -> str:
    rec = {"kind": kind, "status": status,
           "written": datetime.datetime.now().isoformat(timespec="seconds"),
           "env": env_stamp(), "git": git_stamp(), **payload}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    return path


def load_record(path) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize(rec: dict) -> str:
    """One page a human can read before shipping."""
    g = rec.get("git", {})
    prov = (f"git {g.get('tag') or g.get('commit', '')[:8]}"
            + (" (DIRTY)" if g.get("dirty") else "")) if g.get("commit") else "no git checkout"
    L = [f"{rec['kind']}: {rec['status']}   ({rec.get('written')})",
         f"  htq_hw {rec['env']['htq_hw']}  {prov}",
         f"  environment pinned: {rec['env']['pinned_ok']}"
         + ("" if rec["env"]["pinned_ok"] else f"  {rec['env']['pin_mismatches']}")]
    if "target" in rec:
        t = rec["target"]
        L.append(f"  target {t.get('label')}  real={t.get('is_real')}  "
                 f"{t.get('num_qubits')}q  {t.get('n_edges')} edges")
    if "embedding" in rec:
        e = rec["embedding"]
        L.append(f"  embedding {e.get('kind')}  redundancy {e.get('redundancy')}  "
                 f"spare {e.get('spare_qubits')}")
    for step in rec.get("steps", []):
        L.append(f"  [{step['status']:4}] {step['name']}: {step.get('detail', '')}")
    if rec.get("failures"):
        L.append("  FAILURES:")
        L += [f"    - {f}" for f in rec["failures"]]
    if rec.get("warnings"):
        L.append("  warnings:")
        L += [f"    - {w}" for w in rec["warnings"]]
    return "\n".join(L)
