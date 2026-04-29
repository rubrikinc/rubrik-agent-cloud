"""Framework-agnostic client for the Rubrik webhook policy service.

Handles webhook communication, OpenAI format translation, and policy result
parsing. Framework adapters (ADK, LangChain, etc.) use this client to evaluate
tool calls against policies.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """Framework-agnostic representation of a tool call."""

    name: str
    arguments: str  # JSON string
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")


@dataclass(frozen=True)
class PolicyResult:
    """Result from the webhook policy evaluation.

    If all tools are allowed, ``blocked`` is empty and ``explanation`` is empty.
    """

    allowed: tuple[ToolCall, ...]
    blocked: tuple[ToolCall, ...]
    explanation: str


class RubrikPolicyClient:
    """Framework-agnostic client for the Rubrik webhook policy service.

    Requires a ``webhook_url``. Use :meth:`from_env` to construct from
    environment variables.
    """

    _WEBHOOK_URL_ENV = "RUBRIK_WEBHOOK_URL"
    _API_KEY_ENV = "RUBRIK_API_KEY"
    _FAIL_OPEN_ENV = "RUBRIK_POLICY_FAIL_OPEN"

    _CONNECT_TIMEOUT = 2.0
    _TOTAL_TIMEOUT = 5.0

    def __init__(
        self,
        webhook_url: str,
        api_key: Optional[str] = None,
        fail_open: bool = True,
    ) -> None:
        if not webhook_url:
            raise ValueError(
                "webhook_url is required. Set RUBRIK_WEBHOOK_URL or pass explicitly. "
                "If you don't need policy enforcement, don't construct this client."
            )

        base = webhook_url.removesuffix("/").removesuffix("/v1")
        self._webhook_url = base
        self._tool_blocking_endpoint = f"{base}/v1/after_completion/openai/v1"
        self.fail_open = fail_open
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._client: Optional[httpx.AsyncClient] = None
        log.info(
            "RubrikPolicyClient enabled (webhook=%s, fail_open=%s)",
            self._webhook_url,
            self.fail_open,
        )

    @classmethod
    def from_env(cls) -> RubrikPolicyClient:
        """Construct from environment variables.

        Reads ``RUBRIK_WEBHOOK_URL``, ``RUBRIK_API_KEY``, and
        ``RUBRIK_POLICY_FAIL_OPEN`` from the environment.

        Raises ``RuntimeError`` if ``RUBRIK_WEBHOOK_URL`` is not set.
        """
        webhook_url = os.getenv(cls._WEBHOOK_URL_ENV, "")
        if not webhook_url:
            raise RuntimeError(
                f"{cls._WEBHOOK_URL_ENV} must be set. "
                "The rubrik-agent-cloud-policy-plugin is installed but has no webhook to connect to."
            )
        api_key = os.getenv(cls._API_KEY_ENV)
        fail_open = os.getenv(cls._FAIL_OPEN_ENV, "true").lower() != "false"
        return cls(webhook_url=webhook_url, api_key=api_key, fail_open=fail_open)

    def __reduce__(self):
        """Re-run from_env on unpickle so env vars are read in the target environment.

        Agent Engine uses cloudpickle to serialize plugins locally, then
        deserializes on the remote container where env vars like
        RUBRIK_WEBHOOK_URL are set.  Raises RuntimeError at unpickle time
        if the env var is missing — a loud signal that the deployment is
        misconfigured.
        """
        return (self.from_env, ())

    # -- public API -----------------------------------------------------------

    async def evaluate_tool_calls(self, tool_calls: list[ToolCall]) -> PolicyResult:
        """Evaluate tool calls against Rubrik policies via the webhook.

        Raises on HTTP or parsing errors — the caller decides fail-open
        vs. fail-closed behavior.
        """
        payload = _to_openai_payload(tool_calls)
        client = self._get_client()

        resp = await client.post(
            self._tool_blocking_endpoint, json=payload, headers=self._headers
        )
        resp.raise_for_status()

        return _parse_webhook_response(resp.json(), tool_calls)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # -- internals ------------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._TOTAL_TIMEOUT, connect=self._CONNECT_TIMEOUT),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client


# -- format translation (module-level, stateless) ----------------------------


def _to_openai_payload(tool_calls: list[ToolCall]) -> dict:
    """Build an OpenAI chat completion dict from a list of ToolCalls."""
    openai_tool_calls = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": tc.arguments,
            },
        }
        for tc in tool_calls
    ]
    return {
        "id": f"rubrik-policy-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": "unknown",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": openai_tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
    }


def _parse_webhook_response(
    response: dict, original_tool_calls: list[ToolCall]
) -> PolicyResult:
    """Parse the webhook response to determine allowed/blocked tools.

    The webhook returns the *allowed* tool calls in
    ``choices[0].message.tool_calls`` and a blocking explanation in
    ``choices[0].message.content``.  Allowed tools are matched by their
    ``id`` field.
    """
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("Webhook returned empty choices")

    message = choices[0].get("message", {})
    returned_tool_calls = message.get("tool_calls") or []
    explanation = message.get("content") or ""

    allowed_ids = {tc["id"] for tc in returned_tool_calls if tc.get("id")}

    allowed = tuple(tc for tc in original_tool_calls if tc.id in allowed_ids)
    blocked = tuple(tc for tc in original_tool_calls if tc.id not in allowed_ids)

    return PolicyResult(allowed=allowed, blocked=blocked, explanation=explanation)
