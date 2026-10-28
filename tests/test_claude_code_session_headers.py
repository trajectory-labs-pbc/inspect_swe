"""Claude Code's session headers are recorded on the bridge's model events.

No sandbox, no Docker, no API keys: the agent runs only until it opens its
model bridge, and the test checks the options it opened it with.
"""

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import anyio
import pytest
from inspect_ai.agent import AgentState, SandboxAgentBridge, sandbox_agent_bridge
from inspect_ai.model import ChatMessageUser
from inspect_ai.util import Store
from inspect_swe import claude_code
from inspect_swe._claude_code import claude_code as claude_code_module


class _BridgeReached(Exception):
    pass


def _bridge_options() -> dict[str, Any]:
    options: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_bridge(state: AgentState, **kwargs: Any) -> AsyncIterator[None]:
        options.update(kwargs)
        raise _BridgeReached()
        yield

    @asynccontextmanager
    async def no_checkpointer() -> AsyncIterator[None]:
        yield None

    agent = claude_code(model="mockllm/model")
    with (
        patch.object(claude_code_module, "sandbox_agent_bridge", fake_bridge),
        patch.object(claude_code_module, "checkpointer", no_checkpointer),
        patch.object(claude_code_module, "store", return_value=Store()),
        pytest.raises(_BridgeReached),
    ):
        anyio.run(agent, AgentState(messages=[ChatMessageUser(content="Solve it.")]))
    return options


def test_claude_code_asks_the_bridge_to_record_its_session_headers() -> None:
    options = _bridge_options()
    assert options["model_event_metadata_headers"] == (
        "x-claude-code-session-id",
        "x-claude-code-agent-id",
        "x-claude-code-parent-agent-id",
    )
    # the installed inspect_ai accepts every option claude_code passes, and
    # accepts these names (it refuses names that may carry a credential)
    inspect.signature(sandbox_agent_bridge).bind(None, **options)
    bridge = SandboxAgentBridge(
        state=AgentState(messages=[]),
        filter=None,
        retry_refusals=None,
        compaction=None,
        port=13131,
        model=None,
        model_event_metadata_headers=options["model_event_metadata_headers"],
    )
    assert bridge.model_event_metadata_headers == frozenset(
        options["model_event_metadata_headers"]
    )
