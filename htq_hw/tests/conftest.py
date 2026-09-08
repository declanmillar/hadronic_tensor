"""Shared fixtures.  The suite must run on a stranger's machine, offline:
no absolute paths from this workstation, and no network."""
import pathlib

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: offline 101-qubit transpiles (minutes)")


@pytest.fixture(scope="session")
def scratch(tmp_path_factory) -> pathlib.Path:
    """Session-scoped scratch directory for test artefacts (bits, slices,
    caches).  Replaces the hardcoded developer paths that shipped in the
    bundle and would not exist anywhere else."""
    return tmp_path_factory.mktemp("htq_tests")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if a test reaches for the IBM platform.  The package is
    handed to people who will run it against a real account; a test that
    silently contacted the service could submit or leak credentials."""
    try:
        import qiskit_ibm_runtime as R
    except Exception:
        return
    def _blocked(*a, **k):
        raise RuntimeError("tests must run offline: QiskitRuntimeService was constructed")
    monkeypatch.setattr(R.QiskitRuntimeService, "__init__", _blocked, raising=False)
