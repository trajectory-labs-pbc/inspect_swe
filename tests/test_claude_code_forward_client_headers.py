"""``forward_client_headers`` reaches the model bridge unchanged.

The model bridge drops client header values it was not told to forward, so a
caller that needs Claude Code's own feature flags (``anthropic-beta`` values)
to reach the provider lists them on ``claude_code()``, which passes them to
``sandbox_agent_bridge``.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import anyio
import pytest
from inspect_ai.agent import AgentState
from inspect_swe._claude_code import claude_code as claude_code_module


class _BridgeReached(Exception):
    def __init__(self, kwargs: dict[str, Any]) -> None:
        super().__init__("sandbox_agent_bridge reached")
        self.kwargs = kwargs


def _bridge_kwargs(
    monkeypatch: pytest.MonkeyPatch, **agent_args: Any
) -> dict[str, Any]:
    """Run the agent up to its bridge call and return the call's keyword args."""

    @asynccontextmanager
    async def fake_bridge(*args: Any, **kwargs: Any) -> AsyncIterator[None]:
        raise _BridgeReached(kwargs)
        yield

    @asynccontextmanager
    async def fake_checkpointer() -> AsyncIterator[None]:
        yield

    class _Store(dict[str, Any]):
        def set(self, key: str, value: Any) -> None:
            self[key] = value

    the_store = _Store()
    monkeypatch.setattr(claude_code_module, "sandbox_agent_bridge", fake_bridge)
    monkeypatch.setattr(claude_code_module, "checkpointer", fake_checkpointer)
    monkeypatch.setattr(claude_code_module, "store", lambda: the_store)

    agent = claude_code_module.claude_code(model="mockllm/model", **agent_args)
    with pytest.raises(_BridgeReached) as reached:
        anyio.run(agent, AgentState(messages=[]))
    return reached.value.kwargs


def test_forward_client_headers_passed_to_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = {
        "anthropic-beta": ["claude-code-20250219", "interleaved-thinking-2025-05-14"]
    }
    kwargs = _bridge_kwargs(monkeypatch, forward_client_headers=headers)
    assert kwargs["forward_client_headers"] == headers


def test_forward_client_headers_omitted_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _bridge_kwargs(monkeypatch)
    assert "forward_client_headers" not in kwargs
