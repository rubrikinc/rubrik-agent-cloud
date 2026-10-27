"""Framework-agnostic client for the Rubrik webhook policy service.

Handles webhook communication, OpenAI format translation, and policy result
parsing. Framework adapters (ADK, LangChain, etc.) use this client to evaluate
model responses, prompts, and tool results against policies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

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
    """Result from the webhook policy evaluation of a model response.

    If nothing was blocked, ``blocked`` is empty, ``response_blocked`` is
    false, and ``explanation`` is empty. When ``response_blocked`` is true the
    whole response was blocked: ``allowed`` is empty and ``explanation`` is the
    text that replaces the response.
    """

    allowed: tuple[ToolCall, ...]
    blocked: tuple[ToolCall, ...]
    explanation: str
    response_blocked: bool = False


@dataclass(frozen=True)
class PolicyContext:
    """Request context sent with each evaluation.

    ``messages`` is the conversation in OpenAI chat format, ending with the
    turn being evaluated. ``tools`` lists the agent's declared tools in OpenAI
    format. ``agent_id`` identifies the agent to Agent Cloud so that policies
    scoped to specific agents apply. ``correlation_id`` groups the checks of
    one agent run (the ADK invocation ID): ``before_prompt`` sends it as
    ``correlation_key``, and ``after_completion`` uses it as the prefix of
    the response ``id``, which is unique per check.
    """

    messages: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    model: str = ""
    agent_id: str = ""
    correlation_id: str = ""


class RubrikPolicyClient:
    """Framework-agnostic client for the Rubrik webhook policy service.

    Requires a ``webhook_url``. Use :meth:`from_env` to construct from
    environment variables.
    """

    _WEBHOOK_URL_ENV = "RUBRIK_WEBHOOK_URL"
    _API_KEY_ENV = "RUBRIK_API_KEY"
    _FAIL_OPEN_ENV = "RUBRIK_POLICY_FAIL_OPEN"
    _AGENT_ID_ENV = "RUBRIK_AGENT_ID"
    # Set by Vertex AI Agent Runtime to the reasoning engine ID, which is the
    # agent ID that the Google Cloud Gemini Enterprise connection records.
    _AGENT_ENGINE_ID_ENV = "GOOGLE_CLOUD_AGENT_ENGINE_ID"

    _CONNECT_TIMEOUT = 2.0
    _TOTAL_TIMEOUT = 5.0
    # The webhook rejects request bodies over 15 MiB with 413. Older messages
    # are dropped to stay under this, with headroom for the response and tools.
    _MAX_BODY_BYTES = 12 * 1024 * 1024

    def __init__(
        self,
        webhook_url: str,
        api_key: Optional[str] = None,
        fail_open: bool = True,
        agent_id: str = "",
    ) -> None:
        if not webhook_url:
            raise ValueError(
                "webhook_url is required. Set RUBRIK_WEBHOOK_URL or pass explicitly. "
                "If you don't need policy enforcement, don't construct this client."
            )

        base = webhook_url.removesuffix("/").removesuffix("/v1")
        self._webhook_url = base
        self._tool_blocking_endpoint = f"{base}/v1/after_completion/openai/v1"
        self._prompt_endpoint = f"{base}/v1/before_prompt/openai/v1"
        self.fail_open = fail_open
        self.agent_id = agent_id
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._clients: dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}
        self._clients_lock = threading.Lock()
        log.info(
            "RubrikPolicyClient enabled (webhook=%s, fail_open=%s, agent_id=%s)",
            self._webhook_url,
            self.fail_open,
            self.agent_id or "<none>",
        )

    @classmethod
    def from_env(cls) -> RubrikPolicyClient:
        """Construct from environment variables.

        Reads ``RUBRIK_WEBHOOK_URL``, ``RUBRIK_API_KEY``,
        ``RUBRIK_POLICY_FAIL_OPEN``, and ``RUBRIK_AGENT_ID`` from the
        environment. ``RUBRIK_AGENT_ID`` defaults to
        ``GOOGLE_CLOUD_AGENT_ENGINE_ID``.

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
        agent_id = os.getenv(cls._AGENT_ID_ENV) or os.getenv(cls._AGENT_ENGINE_ID_ENV, "")
        return cls(
            webhook_url=webhook_url,
            api_key=api_key,
            fail_open=fail_open,
            agent_id=agent_id,
        )

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

    async def evaluate_response(
        self,
        tool_calls: list[ToolCall],
        text: str = "",
        context: Optional[PolicyContext] = None,
    ) -> PolicyResult:
        """Evaluate a model response (tool calls and text) against policies.

        Sends the ``{request, response}`` envelope to the after_completion
        webhook. Raises on HTTP or parsing errors — the caller decides
        fail-open vs. fail-closed behavior.
        """
        context = context or PolicyContext()
        response = _to_openai_payload(tool_calls, text, _response_id(context.correlation_id))
        envelope = _fit_body(
            lambda request: {"request": request, "response": response},
            _to_openai_request(context, self.agent_id),
            self._MAX_BODY_BYTES,
        )
        data = await self._post(self._tool_blocking_endpoint, envelope)
        return _parse_webhook_response(data, tool_calls, text)

    async def evaluate_tool_calls(
        self,
        tool_calls: list[ToolCall],
        context: Optional[PolicyContext] = None,
    ) -> PolicyResult:
        """Evaluate tool calls against policies. See :meth:`evaluate_response`."""
        return await self.evaluate_response(tool_calls, context=context)

    async def evaluate_prompt(self, context: PolicyContext) -> Optional[str]:
        """Evaluate the last message of ``context.messages`` against policies.

        The last message must be a user prompt (role ``user``) or a tool
        result (role ``tool``); the webhook passes any other turn through.
        Returns the refusal text when the message is blocked, else ``None``.
        Raises on HTTP or parsing errors.
        """
        body = _to_openai_request(context, self.agent_id)
        if context.correlation_id:
            body["correlation_key"] = context.correlation_id
        body = _fit_body(lambda request: request, body, self._MAX_BODY_BYTES)
        data = await self._post(self._prompt_endpoint, body)
        return _parse_prompt_response(data)

    async def close(self) -> None:
        """Close the HTTP client of the running event loop."""
        with self._clients_lock:
            client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()

    # -- internals ------------------------------------------------------------

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = await self._get_client().post(endpoint, json=payload, headers=self._headers)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"Webhook returned {type(data).__name__}, expected an object")
        return data

    def _get_client(self) -> httpx.AsyncClient:
        # An httpx.AsyncClient is bound to the event loop it first ran on.
        # ADK's synchronous Runner.run (used by AdkApp.stream_query) runs each
        # call on a new event loop in its own thread, and calls can overlap,
        # so keep one client per loop. Clients of closed loops can no longer
        # be closed with aclose(); dropping them releases their connections.
        loop = asyncio.get_running_loop()
        with self._clients_lock:
            client = self._clients.get(loop)
            if client is None:
                for stale in [l for l in self._clients if l.is_closed()]:
                    del self._clients[stale]
                client = self._clients[loop] = httpx.AsyncClient(
                    timeout=httpx.Timeout(self._TOTAL_TIMEOUT, connect=self._CONNECT_TIMEOUT),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
        return client


# -- format translation (module-level, stateless) ----------------------------


def _to_openai_request(context: PolicyContext, default_agent_id: str = "") -> dict[str, Any]:
    """Build an OpenAI chat completion request dict from a PolicyContext."""
    request: dict[str, Any] = {
        "model": context.model,
        "messages": list(context.messages),
    }
    if context.tools:
        request["tools"] = list(context.tools)
    agent_id = context.agent_id or default_agent_id
    if agent_id:
        request["metadata"] = {"rac_agent_id": agent_id}
    return request


def _response_id(correlation_id: str) -> str:
    """A response ``id`` unique per check, prefixed by the correlation ID.

    The webhook keys the forensic records of a check by this ``id``, so
    checks of different model responses in one run must not share it.
    """
    suffix = uuid.uuid4().hex[:12]
    return f"{correlation_id}-{suffix}" if correlation_id else f"rubrik-policy-{suffix}"


def _fit_body(
    build: Callable[[dict[str, Any]], dict[str, Any]],
    request: dict[str, Any],
    max_bytes: int,
) -> dict[str, Any]:
    """Drop the oldest conversation messages until the body fits ``max_bytes``.

    ``build`` maps the OpenAI request to the full body. Leading system
    messages and the last message (the turn being evaluated) are kept, and
    tool results are not left without the assistant turn that called them.
    """
    body = build(request)
    size = _json_size(body)
    if size <= max_bytes:
        return body

    messages = request["messages"]
    head = 0
    while head < len(messages) - 1 and messages[head].get("role") in ("system", "developer"):
        head += 1
    sizes = [_json_size(m) + 2 for m in messages]  # +2 for the ", " separator
    start = head
    while start < len(messages) - 1 and size > max_bytes:
        size -= sizes[start]
        start += 1
    while start < len(messages) - 1 and messages[start].get("role") == "tool":
        start += 1
    log.warning(
        "Conversation exceeds the webhook body limit; dropped the %d oldest messages",
        start - head,
    )
    return build({**request, "messages": messages[:head] + messages[start:]})


def _json_size(value: Any) -> int:
    return len(json.dumps(value, default=str).encode())


def _to_openai_payload(
    tool_calls: list[ToolCall], text: str = "", response_id: str = ""
) -> dict[str, Any]:
    """Build an OpenAI chat completion response dict."""
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = [
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
        "id": response_id or f"rubrik-policy-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": "unknown",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }


def _parse_webhook_response(
    response: dict[str, Any],
    original_tool_calls: list[ToolCall],
    sent_text: str = "",
) -> PolicyResult:
    """Parse the after_completion response to determine what was blocked.

    The webhook returns the *allowed* tool calls in
    ``choices[0].message.tool_calls``, matched to the originals by ``id``.
    It signals a blocked tool call by appending an explanation to the
    content, separated from any original text by a blank line. It signals a
    blocked response by replacing the content entirely.

    Raises ``ValueError`` if the response does not have that shape.
    """
    message = _first_message(response)
    if message is None:
        raise ValueError("Webhook returned empty choices")
    content = _content(message)
    returned_tool_calls = message.get("tool_calls") or []
    if not isinstance(returned_tool_calls, list) or not all(
        isinstance(tc, dict) for tc in returned_tool_calls
    ):
        raise ValueError("Webhook returned malformed tool_calls")

    appended = bool(sent_text) and content.startswith(f"{sent_text}\n\n")
    if sent_text and content != sent_text and not appended:
        return PolicyResult(
            allowed=(),
            blocked=tuple(original_tool_calls),
            explanation=content or "Response blocked by Rubrik Agent Cloud policy.",
            response_blocked=True,
        )

    # Counter, not set: if a returned id matches two originals, only as many
    # originals as were returned count as allowed.
    remaining = Counter(
        tc["id"] for tc in returned_tool_calls if isinstance(tc.get("id"), str)
    )
    allowed: list[ToolCall] = []
    blocked: list[ToolCall] = []
    for tc in original_tool_calls:
        if remaining[tc.id] > 0:
            remaining[tc.id] -= 1
            allowed.append(tc)
        else:
            blocked.append(tc)

    explanation = ""
    if blocked:
        explanation = content[len(sent_text):].lstrip("\n") if appended else content
    return PolicyResult(allowed=tuple(allowed), blocked=tuple(blocked), explanation=explanation)


def _parse_prompt_response(response: dict[str, Any]) -> Optional[str]:
    """Parse the before_prompt response.

    The webhook returns ``{}`` to let the message through, or a synthetic
    chat completion whose ``choices[0].message.content`` is the refusal.

    Raises ``ValueError`` if the response does not have that shape.
    """
    message = _first_message(response)
    if message is None:
        return None
    return _content(message) or "Message blocked by Rubrik Agent Cloud policy."


def _first_message(response: dict[str, Any]) -> Optional[dict[str, Any]]:
    """``choices[0].message``, or ``None`` when there are no choices."""
    choices = response.get("choices")
    if not choices:
        return None
    if not isinstance(choices, list) or not isinstance(choices[0], dict):
        raise ValueError("Webhook returned malformed choices")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise ValueError("Webhook returned a malformed message")
    return message


def _content(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("Webhook returned non-string content")
    return content
