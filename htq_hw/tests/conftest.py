import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: offline 101-qubit transpiles (minutes)")
