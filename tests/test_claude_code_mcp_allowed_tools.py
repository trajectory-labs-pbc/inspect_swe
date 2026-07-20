"""Regression test for the ``resolve_mcp_servers`` allowed-tools glob.

Claude Code's ``--allowed-tools`` parser only recognizes a wildcard rule in
the exact form ``mcp__<server>__*`` (double underscore before the glob). A
single-underscore variant (``mcp__<server>_*``) is silently ignored (logged
as a warning, not an error), so under a permission mode that actually
consults the allow-list (e.g. ``--permission-mode auto`` -- as opposed to
``--dangerously-skip-permissions``, which bypasses the allow-list check
entirely) every bridged MCP tool call is denied. Confirmed empirically
against the real ``claude`` CLI (2.1.205 and 2.1.212) in a fresh sandbox
home: the single-underscore rule is ignored and tool calls report "you
haven't granted it yet"; the double-underscore rule is honored and the same
tool call succeeds.
"""

import inspect

from inspect_ai.tool import MCPServerConfig
from inspect_swe import claude_code
from inspect_swe._claude_code.claude_code import resolve_mcp_servers


def test_all_tools_wildcard_uses_double_underscore_before_glob() -> None:
    server = MCPServerConfig(type="http", name="taiga-mcp", tools="all")
    _mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])
    assert allowed_tools == ["mcp__taiga-mcp__*"]


def test_explicit_tool_list_is_unaffected() -> None:
    server = MCPServerConfig(
        type="http", name="taiga-mcp", tools=["list_files", "read_file"]
    )
    _mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])
    assert allowed_tools == [
        "mcp__taiga-mcp__list_files",
        "mcp__taiga-mcp__read_file",
    ]


def test_default_emits_mcp_config_and_allowlist() -> None:
    """Default (allowlist_mcp_tools=True) is byte-for-byte unchanged.

    It emits both the --mcp-config server spec and the allow-list entries.
    """
    server = MCPServerConfig(type="http", name="taiga-mcp", tools="all")
    mcp_config_cmds, allowed_tools = resolve_mcp_servers([server])
    assert mcp_config_cmds[0] == "--mcp-config"
    assert len(mcp_config_cmds) == 2
    assert allowed_tools == ["mcp__taiga-mcp__*"]


def test_suppressed_allowlist_emits_mcp_config_but_no_allowed_tools() -> None:
    """Suppressing the allow-list still registers the servers via --mcp-config.

    The bridged tools remain invocable, but no allow-list entries are emitted,
    so under a classifier-backed permission mode (e.g. --permission-mode auto)
    every bridged MCP call is adjudicated by the classifier rather than
    pre-approved by an allow rule.
    """
    server = MCPServerConfig(type="http", name="taiga-mcp", tools="all")
    mcp_config_cmds, allowed_tools = resolve_mcp_servers(
        [server], allowlist_mcp_tools=False
    )
    assert mcp_config_cmds[0] == "--mcp-config"
    assert len(mcp_config_cmds) == 2
    assert allowed_tools == []


def test_suppressed_allowlist_with_explicit_tool_list_emits_no_allowed_tools() -> None:
    server = MCPServerConfig(
        type="http", name="taiga-mcp", tools=["list_files", "read_file"]
    )
    mcp_config_cmds, allowed_tools = resolve_mcp_servers(
        [server], allowlist_mcp_tools=False
    )
    assert mcp_config_cmds[0] == "--mcp-config"
    assert allowed_tools == []


def test_claude_code_exposes_allowlist_mcp_tools_param() -> None:
    """The claude_code() agent exposes allowlist_mcp_tools, defaulting True.

    The True default keeps existing full/auto callers unaffected.
    """
    param = inspect.signature(claude_code).parameters.get("allowlist_mcp_tools")
    assert param is not None
    assert param.default is True
