"""New runs must not replace archives; only explicit resumes reuse files."""

from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.run_outputs import check_output_prefix, output_directory, output_file


@pytest.mark.parametrize("kind", ["hw", "mps"])
def test_default_runs_are_separate_and_stay_under_run_root(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    first = output_directory("../prodR/ibm_phoenix", kind=kind)
    second = output_directory("../prodR/ibm_phoenix", kind=kind)
    assert first != second
    assert first.parent == second.parent == Path(f"data/{kind}/runs/prodR-ibm_phoenix")
    assert first.is_dir() and second.is_dir()
    assert first.name.endswith("Z")


@pytest.mark.parametrize("suffix", [".npz", ".pdf", ".png", ".log",
                                    "_command.json", "_odd.pdf", "_cuts.png"])
def test_existing_output_blocks_reuse_without_modifying_files(tmp_path, suffix):
    path = tmp_path / ("tensor" + suffix)
    path.write_bytes(b"previous campaign result")
    with pytest.raises(FileExistsError, match="Choose a new output"):
        check_output_prefix(tmp_path / "tensor")
    assert path.read_bytes() == b"previous campaign result"


def test_distinct_products_can_share_an_explicit_run_directory(tmp_path):
    out = output_directory("kingston", tmp_path / "run")
    (out / "hw_w00_coarse_t3.npz").write_bytes(b"coarse result")
    assert check_output_prefix(out / "hw_t3_figB") == str(out / "hw_t3_figB")
    assert output_directory("kingston", out) == out


def test_resume_requires_an_existing_explicit_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="explicit --out"):
        output_file(None, resume=True)
    path = tmp_path / "grid.npz"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        output_file(path, resume=True)
    path.write_bytes(b"checkpoint")
    with pytest.raises(FileExistsError):
        output_file(path)
    assert output_file(path, resume=True) == str(path)
    assert path.read_bytes() == b"checkpoint"


def test_mps_resume_keeps_finished_rows_and_rejects_mismatches(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    import prep_common
    from htensor import Z2Lattice, backends

    monkeypatch.setattr(prep_common, "build_prep",
                        lambda args: (Z2Lattice(args.ns), None, {"label": "vac"}))
    monkeypatch.setattr(prep_common, "stored_prep", lambda *a: (None, None, 0.))
    calls = []
    interrupt = True

    def correlator(lat, prep, insert, probes, m0, g2, eta, times, **kwargs):
        calls.append(times[0])
        if interrupt and times[0] > 0:
            raise RuntimeError("simulated interruption")
        row = np.full((1, len(probes)), 1 + times[0], complex)
        return SimpleNamespace(correlator=row, probe_expect=row / 2, insert_expect=.25)

    monkeypatch.setattr(backends, "hadamard_correlator_aer", correlator)
    out = tmp_path / "grid.npz"
    argv = ["run_w_ns.py", "--state", "vac", "--ns", "4", "--center", "0",
            "--probes", "j0", "--tmax", "0.5", "--out", str(out)]

    def execute(extra=()):
        monkeypatch.setattr(sys, "argv", argv + list(extra))
        runpy.run_path(str(root / "scripts/run_w_ns.py"), run_name="__main__")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        execute()
    with np.load(out) as z:
        first = z["corr_vac"][0].copy()
        assert z["n_done_vac"] == 1
    saved = out.read_bytes()
    with pytest.raises(FileExistsError):
        execute()
    for extra in (("--dt-target", ".25"), ("--tmax", "1"), ("--insert", "j1")):
        with pytest.raises(ValueError, match="does not match|do not match"):
            execute(("--resume", *extra))
        assert out.read_bytes() == saved
    calls.clear()
    interrupt = False
    execute(("--resume",))
    assert calls == [.5]
    with np.load(out) as z:
        np.testing.assert_array_equal(z["corr_vac"][0], first)
        assert z["n_done_vac"] == 2
