from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from inspect_swe._util.agentbinary import (
    AgentBinarySource,
    AgentBinaryVersion,
    AgentBinaryVersionMismatchError,
    ensure_agent_binary_installed,
)


@pytest.mark.anyio
async def test_install_rejects_binary_with_wrong_reported_version(tmp_path: Path) -> None:
    async def resolve_version(*_: object) -> AgentBinaryVersion:
        return AgentBinaryVersion("1.2.3", "checksum", "https://example.invalid/binary")

    async def reported_version(*_: object) -> str | None:
        return "1.2.2"

    source = AgentBinarySource(
        agent="test agent",
        binary="test-agent",
        resolve_version=resolve_version,
        cached_binary_path=lambda version, platform: tmp_path / f"{version}-{platform}",
        list_cached_binaries=lambda: [],
        post_download=None,
        post_install=None,
        reported_version=reported_version,
    )
    sandbox = SimpleNamespace(write_file=AsyncMock())

    with (
        patch(
            "inspect_swe._util.agentbinary.detect_sandbox_platform",
            AsyncMock(return_value="linux-x64"),
        ),
        patch(
            "inspect_swe._util.agentbinary.download_agent_binary_async",
            AsyncMock(return_value=(b"binary", "1.2.3")),
        ),
        patch("inspect_swe._util.agentbinary.sandbox_exec", AsyncMock()),
        pytest.raises(AgentBinaryVersionMismatchError, match="expected 1.2.3, got 1.2.2"),
    ):
        await ensure_agent_binary_installed(source, version="stable", sandbox=sandbox)
