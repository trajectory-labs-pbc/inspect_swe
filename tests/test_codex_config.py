import pytest
from inspect_swe._codex_cli.config import (
    CodexApprovalPolicy,
    CodexSandboxMode,
    codex_cli_config_overrides,
    codex_config_options,
    codex_mcp_server_toml,
    codex_sandbox_args,
    resolve_codex_approval_policy,
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


def test_codex_mcp_server_toml_sets_approve_when_never() -> None:
    from inspect_swe._codex_cli.config import MCP_STARTUP_TIMEOUT_SEC

    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    result = codex_mcp_server_toml(dump, "never")
    # startup_timeout_sec is expected in EVERY table: codex's own default is
    # short enough that a slow sandboxed server is abandoned and the agent runs
    # with no tools, silently scored. See codex_mcp_server_toml.
    assert result == {
        "type": "http",
        "url": "http://localhost:8901/mcp/taiga-mcp",
        "default_tools_approval_mode": "approve",
        "startup_timeout_sec": MCP_STARTUP_TIMEOUT_SEC,
    }


def test_codex_mcp_server_toml_leaves_other_policies_untouched() -> None:
    """A prompting policy keeps the per-server approval default.

    The startup timeout is NOT policy-dependent -- a slow server is abandoned
    regardless of how tool calls get approved -- so it is expected here too.
    """
    from inspect_swe._codex_cli.config import MCP_STARTUP_TIMEOUT_SEC

    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    for policy in ("untrusted", "on-request"):
        assert codex_mcp_server_toml(dump, policy) == {
            **dump,
            "startup_timeout_sec": MCP_STARTUP_TIMEOUT_SEC,
        }


def test_codex_mcp_server_toml_does_not_mutate_input() -> None:
    dump = {"type": "http", "url": "http://localhost:8901/mcp/taiga-mcp"}
    codex_mcp_server_toml(dump, "never")
    assert "default_tools_approval_mode" not in dump


@pytest.mark.parametrize(
    ("sandbox_mode", "approval_policy", "network_access", "expected"),
    [
        (
            "danger-full-access",
            "never",
            True,
            ["--dangerously-bypass-approvals-and-sandbox"],
        ),
        (
            "read-only",
            "never",
            True,
            ["--sandbox", "read-only", "-c", "approval_policy=never"],
        ),
        (
            "workspace-write",
            "on-request",
            False,
            [
                "--sandbox",
                "workspace-write",
                "-c",
                "approval_policy=on-request",
                "-c",
                "sandbox_workspace_write.network_access=false",
            ],
        ),
    ],
)
def test_codex_sandbox_args(
    sandbox_mode: CodexSandboxMode,
    approval_policy: CodexApprovalPolicy,
    network_access: bool,
    expected: list[str],
) -> None:
    assert codex_sandbox_args(sandbox_mode, approval_policy, network_access) == expected


def test_config_override_resolves_effective_approval_policy() -> None:
    assert (
        resolve_codex_approval_policy("on-request", {"approval_policy": "never"})
        == "never"
    )


def test_config_override_rejects_unknown_approval_policy() -> None:
    with pytest.raises(ValueError, match="approval_policy"):
        resolve_codex_approval_policy("never", {"approval_policy": "always"})


def test_bridged_mcp_servers_get_a_generous_startup_timeout() -> None:
    """Codex must not give up on a slow-starting MCP server and run toolless.

    Codex awaits its MCP tool list at session start, but the wait is bounded by a
    per-server `startup_timeout_sec`. Nothing set it, so codex used its built-in
    default; when a sandboxed server took longer, codex proceeded with the server
    FAILED and the agent had no environment tools. It then did nothing and the
    empty trajectory was SCORED with no error to retry on -- 2.40% of 250 samples
    at 150-way concurrency, and 87/4130 across a production collection.
    """
    from inspect_swe._codex_cli.config import (
        MCP_STARTUP_TIMEOUT_SEC,
        codex_mcp_server_toml,
    )

    assert MCP_STARTUP_TIMEOUT_SEC >= 120, (
        "startup timeout must exceed realistic sandboxed MCP startup time, or "
        "codex silently runs the agent with no environment tools"
    )
    table = codex_mcp_server_toml({"url": "http://localhost:1/mcp/x"}, "never")
    assert table["startup_timeout_sec"] == MCP_STARTUP_TIMEOUT_SEC
    # an explicit author value must win
    explicit = codex_mcp_server_toml(
        {"url": "http://localhost:1/mcp/x", "startup_timeout_sec": 7}, "never"
    )
    assert explicit["startup_timeout_sec"] == 7
