"""The package must import against upstream inspect-ai (fork-only symbols degrade)."""

import importlib

import pytest


def test_import_inspect_swe_works() -> None:
    # The dev venv pairs with the fork inspect_ai; the upstream pairing is
    # exercised by the fallback test below and by the release's install probes.
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


def _bridge_with_headers(*, model_event_metadata_headers: object = None) -> None:
    raise AssertionError("signature stand-in; never called")


def _bridge_without_headers(*, port: int = 13131) -> None:
    raise AssertionError("signature stand-in; never called")


def test_header_args_name_the_headers_when_the_bridge_records_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_swe._util.inspect_compat as compat

    monkeypatch.setattr(compat, "sandbox_agent_bridge", _bridge_with_headers)
    assert compat.bridge_request_header_args(["x-a", "x-b"]) == {
        "model_event_metadata_headers": ("x-a", "x-b")
    }


def test_header_args_are_empty_when_the_bridge_cannot_record_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect_swe._util.inspect_compat as compat

    monkeypatch.setattr(compat, "sandbox_agent_bridge", _bridge_without_headers)
    assert compat.bridge_request_header_args(["x-a"]) == {}
