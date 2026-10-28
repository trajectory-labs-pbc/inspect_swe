"""Compatibility with upstream inspect-ai installs.

This fork of inspect_swe normally pairs with the trajectory-labs inspect_ai
fork, but nothing pins that pairing for plain installs: this package's own
dependency floor admits upstream inspect-ai from PyPI, and downstream
projects deliberately leave inspect-ai unpinned. Importing inspect_swe must
therefore work against an upstream inspect-ai; fork-only integration
degrades instead of breaking the import.

``BRIDGE_REQUEST_HEADERS`` is the fork's model-metadata key for bridge
request headers (fork inspect_ai ``model/_model.py``). Upstream inspect-ai
never defines the symbol and its bridge never sets the metadata key, so the
literal fallback keeps ``metadata.get(BRIDGE_REQUEST_HEADERS)`` returning
``None`` there -- headers-dependent behavior switches off, nothing else
changes. The value must match the fork's definition; a unit test asserts
that whenever the fork symbol is present.

``model_event_metadata_headers`` is likewise a fork-only parameter of
``sandbox_agent_bridge``: ``bridge_request_header_args`` asks for it only when
the installed bridge accepts it, so upstream simply records no headers.
"""

import inspect
from typing import Any, Sequence

import inspect_ai.model as _inspect_ai_model
from inspect_ai.agent import sandbox_agent_bridge

MODEL_EVENT_METADATA_HEADERS_PARAM = "model_event_metadata_headers"

BRIDGE_REQUEST_HEADERS: str = getattr(
    _inspect_ai_model, "BRIDGE_REQUEST_HEADERS", "bridge_request_headers"
)


def bridge_request_header_args(headers: Sequence[str]) -> dict[str, Any]:
    """Keyword arguments asking the bridge to record ``headers`` on model events.

    Spread the result into the agent's ``sandbox_agent_bridge(...)`` call. The
    recorded headers land in each ``ModelEvent``'s metadata under
    ``BRIDGE_REQUEST_HEADERS``.

    Args:
        headers: The non-sensitive request header names to record.

    Returns:
        ``model_event_metadata_headers`` alone when the installed bridge accepts
        it; otherwise an empty mapping, and no headers are recorded.
    """
    if (
        MODEL_EVENT_METADATA_HEADERS_PARAM
        not in inspect.signature(sandbox_agent_bridge).parameters
    ):
        return {}
    return {MODEL_EVENT_METADATA_HEADERS_PARAM: tuple(headers)}
