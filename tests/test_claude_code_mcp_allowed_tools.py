"""Regression tests for the ``resolve_mcp_servers`` allowed-tools handling.

Two independent behaviors are covered:

1. The wildcard glob form. Claude Code's ``--allowed-tools`` parser only
   recognizes a wildcard rule in the exact form ``mcp__<server>__*`` (double
   underscore before the glob). A single-underscore variant
   (``mcp__<server>_*``) is silently ignored (logged as a warning, not an
   error), so under a permission mode that actually consults the allow-list
   (e.g. ``--permission-mode auto`` -- as opposed to
   ``--dangerously-skip-permissions``, which bypasses the allow-list check
   entirely) every bridged MCP tool call is denied. Confirmed empirically
   against the real ``claude`` CLI (2.1.205 and 2.1.212) in a fresh sandbox
   home: the single-underscore rule is ignored and tool calls report "you
   haven't granted it yet"; the double-underscore rule is honored and the same
   tool call succeeds.

2. The ``allowlist_mcp_tools`` toggle. An allow rule resolves before Claude
   Code's first-party classifier, so allow-listing bridged MCP tools would
   bypass the classifier entirely. Disabling the allow-list must still leave
   the servers registered via ``--mcp-config`` so their tools remain callable.
"""

from inspect_ai.tool import MCPServerConfig
from inspect_swe._claude_code.claude_code import resolve_mcp_servers


def test_all_tools_wildcard_uses_double_underscore_before_glob() -> None:
    server = MCPServerConfig(type="http", name="taiga-mcp", tools="all")
    _mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])
    assert allowed_tools == ["mcp__taiga-mcp__*"]


def test_default_registers_and_allowlists_explicit_mcp_tools() -> None:
    server = MCPServerConfig(
        type="http", name="taiga-mcp", tools=["list_files", "read_file"]
    )

    mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])

    assert mcp_config_cmds[0] == "--mcp-config"
    assert allowed_tools == [
        "mcp__taiga-mcp__list_files",
        "mcp__taiga-mcp__read_file",
    ]


def test_disabling_mcp_allowlist_keeps_servers_registered() -> None:
    server = MCPServerConfig(
        type="http", name="taiga-mcp", tools=["list_files", "read_file"]
    )

    mcp_config_cmds, allowed_tools = resolve_mcp_servers(
        [server], allowlist_mcp_tools=False
    )

    assert mcp_config_cmds[0] == "--mcp-config"
    assert allowed_tools == []
