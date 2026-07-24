"""Provision the ``google-antigravity`` SDK into the sandbox at runtime.

Mirrors ``_gemini_cli/agentbinary.py`` (which provisions node + the Gemini CLI):
inspect_swe agents run against arbitrary sandbox images, so the agent runtime
must be ensured present. Here that runtime is the ``google-antigravity`` Python
SDK, which bundles the ``localharness`` engine that actually runs the agent.

If the SDK is already importable (e.g. baked into the image, as in agent-c's
images), provisioning is skipped. Otherwise it is installed with ``pip`` (which
requires the sandbox to have egress; no-egress images must bake the SDK in).
"""

from __future__ import annotations

from inspect_ai.util import SandboxEnvironment, concurrency

# Candidate interpreters, most-specific first. agent-c images ship the SDK in
# ``/opt/venv``; generic images fall back to a system python.
_CANDIDATE_PYTHONS: tuple[str, ...] = (
    "/opt/venv/bin/python",
    "python3",
    "python",
)
_IMPORT_CHECK = "import google.antigravity"


async def ensure_antigravity_sdk(
    sandbox: SandboxEnvironment,
    user: str | None = None,
) -> str:
    """Return the path to a sandbox python that can import ``google.antigravity``.

    Skips work when the SDK is already present; otherwise provisions it.
    """
    python = await _python_with_sdk(sandbox, user)
    if python is not None:
        return python
    return await _provision_antigravity_sdk(sandbox, user)


async def _python_with_sdk(
    sandbox: SandboxEnvironment,
    user: str | None,
) -> str | None:
    for python in _CANDIDATE_PYTHONS:
        result = await sandbox.exec([python, "-c", _IMPORT_CHECK], user=user)
        if result.success:
            return python
    return None


async def _provision_antigravity_sdk(
    sandbox: SandboxEnvironment,
    user: str | None,
) -> str:
    """Install ``google-antigravity`` into a system python (requires egress)."""
    async with concurrency("antigravity-sdk-install", 1, visible=False):
        # Re-check under the lock in case a concurrent sample installed it.
        python = await _python_with_sdk(sandbox, user)
        if python is not None:
            return python

        base_python = await _base_python(sandbox, user)
        install = await sandbox.exec(
            [base_python, "-m", "pip", "install", "--user", "google-antigravity"],
            user=user,
        )
        if not install.success:
            raise RuntimeError(
                "Failed to provision the google-antigravity SDK into the sandbox. "
                "Bake it into the image for no-egress sandboxes, or ensure the "
                f"sandbox has network egress. stderr:\n{install.stderr.strip()}"
            )
        verify = await sandbox.exec([base_python, "-c", _IMPORT_CHECK], user=user)
        if not verify.success:
            raise RuntimeError(
                "google-antigravity installed but is not importable: "
                f"{verify.stderr.strip()}"
            )
        return base_python


async def _base_python(sandbox: SandboxEnvironment, user: str | None) -> str:
    for python in ("python3", "python"):
        result = await sandbox.exec([python, "--version"], user=user)
        if result.success:
            return python
    raise RuntimeError(
        "No python interpreter found in the sandbox to install google-antigravity."
    )
