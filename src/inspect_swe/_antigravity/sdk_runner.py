from __future__ import annotations

import argparse
import json
import posixpath
from pathlib import Path
from typing import Any, ClassVar, Final, Literal, TypedDict

import anyio
from google.antigravity import Agent, LocalAgentConfig, types
from google.antigravity.hooks import policy
from pydantic import BaseModel, ConfigDict

_BRIDGE_ENDPOINT_MODEL: Final = "gemini-3.6-flash"


class RunnerPayload(TypedDict):
    prompt: str
    system_instructions: str
    bridge_base_url: str
    mcp_name: Literal["taiga-mcp"]
    mcp_url: str
    app_data_dir: str
    save_dir: str


class _RunnerPayloadModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    prompt: str
    system_instructions: str
    bridge_base_url: str
    mcp_name: Literal["taiga-mcp"]
    mcp_url: str
    app_data_dir: str
    save_dir: str


def load_payload(config_path: Path) -> RunnerPayload:
    """Parse the host-created request into the runner's trusted payload."""
    parsed = _RunnerPayloadModel.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )
    return {
        "prompt": parsed.prompt,
        "system_instructions": parsed.system_instructions,
        "bridge_base_url": parsed.bridge_base_url,
        "mcp_name": parsed.mcp_name,
        "mcp_url": parsed.mcp_url,
        "app_data_dir": parsed.app_data_dir,
        "save_dir": parsed.save_dir,
    }


def build_config(payload: RunnerPayload) -> LocalAgentConfig:
    """Create the confined native-Gemini SDK configuration for one bridged sample.

    The SDK's native localharness backend speaks the Gemini generateContent wire
    directly to the Inspect loopback bridge (``base_url``), so there is no OpenAI
    translation seam. This mirrors inspect_swe's gemini_cli, which points a Google
    agent at the bridge via GOOGLE_GEMINI_BASE_URL plus a placeholder api key.
    localharness dispatches taiga-mcp tools through its ``call_mcp_tool`` wrapper
    (with ServerName/ToolName in the args), so the policy must allow that dispatcher
    for the configured server. Tool RESULTS come back as functionResponse parts inside
    model-role turns, which the host bridge converter re-roles into tool messages.
    """
    mcp_server = types.McpStreamableHttpServer(
        name=payload["mcp_name"],
        url=payload["mcp_url"],
    )

    def _targets_configured_server(args: dict[str, Any]) -> bool:
        return args.get("ServerName") == payload["mcp_name"]

    # view_file exists ONLY to read back localharness's offloaded tool results,
    # which live under the agent's own app_data_dir (/home/model/.antigravity).
    # Confine it there so it cannot read task files, the workspace, or grading
    # ground truth elsewhere in the sandbox. Paths are normalized first so ``..``
    # traversal cannot escape the prefix.
    _app_data_dir = posixpath.normpath(payload["app_data_dir"])

    def _within_app_data_dir(args: dict[str, Any]) -> bool:
        raw = args.get("AbsolutePath")
        if not isinstance(raw, str) or not raw:
            return False
        resolved = posixpath.normpath(raw)
        return resolved == _app_data_dir or resolved.startswith(_app_data_dir + "/")

    # Setting base_url makes GeminiAPIEndpoint.validate_endpoint() short-circuit
    # the real-key requirement; the placeholder api_key rides the localharness
    # proto but the bridge never checks it (sandbox egress is network_mode:none).
    bridge_model = types.ModelTarget(
        name=_BRIDGE_ENDPOINT_MODEL,
        types=[types.ModelType.TEXT, types.ModelType.IMAGE],
        endpoint=types.GeminiAPIEndpoint(
            base_url=payload["bridge_base_url"],
            api_key="inspect-bridge",
        ),
    )
    return LocalAgentConfig(
        models=[bridge_model],
        system_instructions=payload["system_instructions"],
        capabilities=types.CapabilitiesConfig(
            # VIEW_FILE is the one builtin the model needs to read back a large MCP
            # tool result that localharness has offloaded to a file (see
            # _ALLOWED_TOOL_NAMES in antigravity.py). With an empty allowlist the
            # offloaded payload is unreadable and any tool observation above
            # localharness's internal cap is silently lost to the model.
            enabled_tools=[types.BuiltinTools.VIEW_FILE],
            enable_subagents=False,
        ),
        mcp_servers=[mcp_server],
        policies=[
            policy.deny_all(),
            policy.allow("call_mcp_tool", when=_targets_configured_server),
            policy.allow(types.BuiltinTools.VIEW_FILE.value, when=_within_app_data_dir),
            *policy.allow(mcp_server),
        ],
        workspaces=[],
        app_data_dir=payload["app_data_dir"],
        save_dir=payload["save_dir"],
    )


async def run(payload: RunnerPayload) -> None:
    """Run one fresh SDK Agent and consume its full streaming response."""
    async with Agent(build_config(payload)) as sdk_agent:
        response = await sdk_agent.chat(payload["prompt"])
        final_text = await response.text()
        print(
            json.dumps(
                {
                    "final_text": final_text,
                    "steps": len(sdk_agent.conversation.history),
                    "turn_count": sdk_agent.conversation.turn_count,
                },
                sort_keys=True,
            )
        )


def main() -> None:
    """Load one host-created request and execute it with AnyIO."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    anyio.run(run, load_payload(Path(arguments.config)))


if __name__ == "__main__":
    main()
