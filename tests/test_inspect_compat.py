"""The package must import against upstream inspect-ai (fork-only symbols degrade)."""

import importlib

import pytest


def test_import_inspect_swe_works() -> None:
    # Collection itself exercises this in an upstream-inspect-ai venv; the
    # explicit import documents the contract.
    import inspect_swe  # noqa: F401


def test_bridge_request_headers_falls_back_when_fork_symbol_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_ai.model

    monkeypatch.delattr(inspect_ai.model, "BRIDGE_REQUEST_HEADERS", raising=False)
    import inspect_swe._util.inspect_compat as compat

    reloaded = importlib.reload(compat)
    try:
        assert reloaded.BRIDGE_REQUEST_HEADERS == "bridge_request_headers"
    finally:
        monkeypatch.undo()
        importlib.reload(compat)


def test_bridge_request_headers_matches_fork_value_when_present() -> None:
    import inspect_ai.model

    fork_value = getattr(inspect_ai.model, "BRIDGE_REQUEST_HEADERS", None)
    if fork_value is None:
        pytest.skip("upstream inspect-ai: fork symbol absent")
    from inspect_swe._util.inspect_compat import BRIDGE_REQUEST_HEADERS

    assert BRIDGE_REQUEST_HEADERS == fork_value
