import pytest
from inspect_swe._codex_cli.config import (
    codex_cli_config_overrides,
    codex_config_options,
    resolve_codex_deprecated_args,
    resolve_codex_web_search,
)
from inspect_swe._util.toml import to_toml


def test_codex_config_defaults() -> None:
    config = codex_config_options("live", True)

    assert config["web_search"] == "live"
    assert config["features.goals"] is True
    toml = to_toml(config)
    assert 'web_search = "live"' in toml
    assert "features.goals = true" in toml


@pytest.mark.parametrize("web_search", ["live", "cached", "disabled"])
def test_resolve_codex_web_search_modes(web_search: str) -> None:
    assert resolve_codex_web_search(web_search) == web_search


def test_resolve_codex_web_search_invalid_mode() -> None:
    with pytest.raises(ValueError, match="web_search must be one of"):
        resolve_codex_web_search("offline")


def test_deprecated_disallowed_tools_disable_web_search() -> None:
    disallowed_tools = resolve_codex_deprecated_args(
        {"disallowed_tools": ["web_search"]}
    )

    assert resolve_codex_web_search("live", disallowed_tools) == "disabled"


def test_deprecated_disallowed_tools_reject_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unsupported Codex disallowed_tools"):
        resolve_codex_deprecated_args({"disallowed_tools": ["bash"]})


def test_deprecated_args_reject_unexpected_keyword() -> None:
    with pytest.raises(TypeError, match="Unexpected keyword argument"):
        resolve_codex_deprecated_args({"unexpected": True})


def test_codex_cli_config_overrides_format_values_for_cli() -> None:
    assert codex_cli_config_overrides("cached", False) == {
        "web_search": '"cached"',
        "features.goals": "false",
    }


def test_bridged_mcp_servers_get_a_generous_startup_timeout() -> None:
    """Codex must not give up on a slow-starting MCP server and run toolless.

    Codex awaits its MCP tool list at session start, but the wait is bounded by a
    per-server `startup_timeout_sec`. Nothing set it, so codex used its built-in
    default; when a sandboxed server took longer, codex proceeded with the server
    FAILED and the agent had no environment tools. It then did nothing, and the
    empty trajectory was SCORED with no error to retry on -- measured at 2.4% of
    250 samples at 150-way concurrency, and 87/4130 across a production
    collection.

    Asserts the value is large enough to cover a slow sandbox boot; a small
    default is exactly the bug.
    """
    from inspect_swe._codex_cli.codex_cli import MCP_STARTUP_TIMEOUT_SEC

    assert MCP_STARTUP_TIMEOUT_SEC >= 120, (
        "startup timeout must exceed realistic sandboxed MCP startup time, or "
        "codex silently runs the agent with no environment tools"
    )
