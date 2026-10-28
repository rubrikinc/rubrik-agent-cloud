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
    # The webhook rejects request bodies over 15 MiB with 413.
    _MAX_BODY_BYTES = 15 * 1024 * 1024

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
        body = _fit_body(
            lambda request: {"request": request, "response": response},
            _to_openai_request(context, self.agent_id),
            self._MAX_BODY_BYTES,
        )
        data = await self._post(self._tool_blocking_endpoint, body)
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
        request = _to_openai_request(context, self.agent_id)
        if context.correlation_id:
            request["correlation_key"] = context.correlation_id
        body = _fit_body(lambda r: r, request, self._MAX_BODY_BYTES)
        data = await self._post(self._prompt_endpoint, body)
        return _parse_prompt_response(data)

    async def close(self) -> None:
        """Close the HTTP client of the running event loop."""
        with self._clients_lock:
            client = self._clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.aclose()

    # -- internals ------------------------------------------------------------

    async def _post(self, endpoint: str, body: bytes) -> dict[str, Any]:
        resp = await self._get_client().post(endpoint, content=body, headers=self._headers)
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
) -> bytes:
    """Serialize the body, dropping the oldest messages to fit ``max_bytes``.

    ``build`` maps the OpenAI request to the full body, so the size counts
    everything sent: the system prompt, the tools, the response under
    evaluation, and the last message.  Leading system messages and the last
    message (the turn being evaluated) are always kept, and a tool result is
    not kept without the assistant turn that called it.  Raises
    ``ValueError`` if the body does not fit even with every other message
    dropped.
    """
    body = _encode(build(request))
    if len(body) <= max_bytes:
        return body

    messages = request["messages"]
    head = 0
    while head < len(messages) - 1 and messages[head].get("role") in ("system", "developer"):
        head += 1
    # Each dropped message removes its bytes and one "," separator.
    over = len(body) - max_bytes
    start = head
    while start < len(messages) - 1 and over > 0:
        over -= len(_encode(messages[start])) + 1
        start += 1
    while start < len(messages) - 1 and messages[start].get("role") == "tool":
        start += 1

    body = _encode(build({**request, "messages": messages[:head] + messages[start:]}))
    if len(body) > max_bytes:
        raise ValueError(
            f"Policy check body is {len(body)} bytes with only the system prompt, "
            f"tools, and the turn being evaluated; the webhook limit is {max_bytes}"
        )
    log.warning(
        "Conversation exceeds the webhook body limit; dropped the %d oldest messages",
        start - head,
    )
    return body


def _encode(value: Any) -> bytes:
    """Serialize as JSON, compactly, as the request body."""
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str
    ).encode()


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


# Fixed prefixes of the webhook's block explanations (predibase
# policy.RubrikDecisionReason and buildExplanationMessage).  A Reference ID and,
# for verbose requests, the policy's reasoning may follow.
_RESPONSE_BLOCKED = "Response blocked by Rubrik Agent Cloud"
_TOOLS_BLOCKED = "The following tool calls were blocked by Rubrik Agent Cloud:"


def _parse_webhook_response(
    response: dict[str, Any],
    original_tool_calls: list[ToolCall],
    sent_text: str = "",
) -> PolicyResult:
    """Parse the after_completion response to determine what was blocked.

    The webhook returns the *allowed* tool calls in
    ``choices[0].message.tool_calls``, matched to the originals by ``id``;
    an original that is not returned was blocked.  It explains blocked tool
    calls by appending an explanation that starts with ``The following tool
    calls were blocked by Rubrik Agent Cloud:``, after the original text and
    a blank line.  It blocks the whole response by replacing the content
    with an explanation that starts with ``Response blocked by Rubrik Agent
    Cloud``.  Any other change to the content is also treated as a blocked
    response, so that a change in the webhook's wording blocks too much
    rather than too little.

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

    # The appended-explanation form is checked first: model text that
    # happens to start with the response-block marker is still model text.
    prefix = f"{sent_text}\n\n" if sent_text else ""
    if content == sent_text:
        explanation = ""
    elif content.startswith(prefix) and content[len(prefix):].startswith(_TOOLS_BLOCKED):
        explanation = content[len(prefix):]
    else:
        if not content.startswith(_RESPONSE_BLOCKED):
            log.warning(
                "Webhook changed the response without a recognized explanation; "
                "treating the response as blocked"
            )
        return PolicyResult(
            allowed=(),
            blocked=tuple(original_tool_calls),
            explanation=content or "Response blocked by Rubrik Agent Cloud policy.",
            response_blocked=True,
        )
    if blocked and not explanation:
        explanation = "Tool calls blocked by Rubrik Agent Cloud policy."
    return PolicyResult(allowed=tuple(allowed), blocked=tuple(blocked), explanation=explanation)


def _parse_prompt_response(response: dict[str, Any]) -> Optional[str]:
    """Parse the before_prompt response.

    The webhook returns ``{}`` to let the message through, or a synthetic
    chat completion with ``finish_reason`` ``content_filter`` (OpenAI's
    moderation signal) whose ``choices[0].message.content`` is the refusal.
    A response with choices but another ``finish_reason`` is still treated
    as a refusal, so that a webhook change blocks too much rather than too
    little.

    Raises ``ValueError`` if the response does not have that shape.
    """
    message = _first_message(response)
    if message is None:
        return None
    if response["choices"][0].get("finish_reason") != "content_filter":
        log.warning("Webhook returned a refusal without finish_reason content_filter")
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
