"""Separate hardware and MPS runs and protect existing output files."""

from datetime import datetime, timezone
from pathlib import Path
import re


def output_directory(campaign, explicit=None, *, kind="hw"):
    """Create a dated run directory, or use an explicitly chosen directory.

    The UTC timestamp records run creation, not hardware acquisition time.
    """
    if explicit is not None:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=True)
        return path
    if kind not in ("hw", "mps"):
        raise ValueError("Output kind must be hw or mps")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", campaign).strip(".-") or kind
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = Path("data") / kind / "runs" / label / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_output_prefix(prefix):
    """Refuse to replace any existing figure, array or sidecar for a prefix."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in prefix.parent.iterdir()
                      if p.name == prefix.name or
                      p.name.startswith((prefix.name + ".", prefix.name + "_")))
    if existing:
        raise FileExistsError(
            f"Output already exists: {existing[0]}. Choose a new output directory or prefix.")
    return str(prefix)


def output_file(path, *, resume=False):
    """Check a production NPZ path once, before incremental writes begin.

    Resume must name an existing file explicitly. The caller validates its
    contents and owns subsequent checkpoint writes within that run.
    """
    if path is None:
        raise ValueError("--resume requires an explicit --out checkpoint.npz")
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Output file must have a .npz extension")
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    else:
        check_output_prefix(path.with_suffix(""))
    return str(path)
