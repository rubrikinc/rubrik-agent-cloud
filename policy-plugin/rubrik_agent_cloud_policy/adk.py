"""ADK adapter for Rubrik Agent Cloud policy enforcement.

Thin wrapper around :class:`RubrikPolicyClient` that implements Google ADK's
:class:`BasePlugin` callbacks.  All policy logic lives in the core client;
this module only handles Gemini <-> ToolCall conversion.

Requires the ``adk`` extra::

    pip install rubrik-agent-cloud-policy-plugin[adk]
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import httpx

try:
    from google.adk.plugins.base_plugin import BasePlugin
    from google.genai import types
except ImportError as exc:
    raise ImportError(
        "google-adk is required for the ADK adapter. "
        "Install with: pip install rubrik-agent-cloud-policy-plugin[adk]"
    ) from exc

from rubrik_agent_cloud_policy.client import PolicyResult, RubrikPolicyClient, ToolCall

log = logging.getLogger(__name__)


class RubrikPolicyPlugin(BasePlugin):
    """ADK plugin for Rubrik Agent Cloud tool-blocking policy enforcement.

    Uses :class:`RubrikPolicyClient` to evaluate tool calls against the Rubrik
    webhook service.  Configuration is read from environment variables.

    **Primary enforcement** via ``after_model_callback``: intercept the LLM
    response, validate tool calls with the webhook, strip blocked calls.

    **Safety-net** via ``before_tool_callback``: catch tools that bypassed
    ``after_model_callback``.  This happens when another plugin's
    ``before_model_callback`` returns a cached response, which causes ADK's
    ``_call_llm_async`` to skip ``after_model_callback`` entirely
    (see ``base_llm_flow.py`` lines 731-735).

    Construction defers client initialization until first use so the plugin
    can be created locally (for cloudpickle) without ``RUBRIK_WEBHOOK_URL``.
    ``__reduce__`` delegates to ``RubrikPolicyClient.from_env()`` which
    raises at unpickle time if the env var is missing.
    """

    PLUGIN_NAME = "rubrik_agent_cloud_policy"

    def __init__(self, name: str = PLUGIN_NAME, **kwargs: Any) -> None:
        super().__init__(name=name)
        self._kwargs = kwargs
        # Defer client creation — may not have env vars at pickle time.
        # _get_policy_client() initializes on first callback invocation.
        self._policy_client: Optional[RubrikPolicyClient] = None

    def _get_policy_client(self) -> RubrikPolicyClient:
        if self._policy_client is None:
            if self._kwargs:
                self._policy_client = RubrikPolicyClient(**self._kwargs)
            else:
                self._policy_client = RubrikPolicyClient.from_env()
        return self._policy_client

    async def close(self) -> None:
        if self._policy_client:
            await self._policy_client.close()
        log.info("[%s] Shut down", self.PLUGIN_NAME)

    # -- callbacks ------------------------------------------------------------

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> Any:
        """Intercept LLM response and strip blocked tool calls."""
        if not llm_response or not llm_response.content:
            return None
        if not llm_response.content.parts:
            return None

        fc_parts = [
            (i, part)
            for i, part in enumerate(llm_response.content.parts)
            if part.function_call is not None
        ]
        if not fc_parts:
            return None

        try:
            client = self._get_policy_client()
            tool_calls, index_by_id = _gemini_parts_to_tool_calls(fc_parts)
            result = await client.evaluate_tool_calls(tool_calls)

            if not result.blocked:
                return None

            log.info(
                "[%s] Blocked tools: %s",
                self.PLUGIN_NAME,
                [tc.name for tc in result.blocked],
            )
            return _apply_policy_result(llm_response, index_by_id, result)

        except (httpx.HTTPError, httpx.StreamError, httpx.InvalidURL, ValueError, KeyError):
            log.exception(
                "[%s] Webhook call failed in after_model_callback", self.PLUGIN_NAME
            )
            return _handle_error(
                llm_response, self._get_policy_client().fail_open
            )

    async def before_tool_callback(
        self, *, tool: Any, tool_args: Any, tool_context: Any
    ) -> Optional[dict[str, str]]:
        """Safety net — evaluates each tool individually before execution.

        Catches tools that bypassed ``after_model_callback`` — this happens
        when another plugin's ``before_model_callback`` returns a cached LLM
        response, causing ADK's ``_call_llm_async`` to skip the entire
        ``after_model_callback`` chain (see ``base_llm_flow.py:731-735``).
        """
        try:
            client = self._get_policy_client()
            tc = ToolCall(
                name=tool.name,
                arguments=json.dumps(tool_args if tool_args else {}),
            )
            result = await client.evaluate_tool_calls([tc])

            if not result.blocked:
                return None

            log.info("[%s] Safety net blocked: %s", self.PLUGIN_NAME, tool.name)
            return {
                "error": result.explanation
                or f"Tool '{tool.name}' blocked by Rubrik Agent Cloud policy"
            }

        except (httpx.HTTPError, httpx.StreamError, httpx.InvalidURL, ValueError, KeyError):
            log.exception(
                "[%s] Webhook call failed in before_tool_callback", self.PLUGIN_NAME
            )
            if self._get_policy_client().fail_open:
                return None
            return {"error": "Tool execution blocked: policy check unavailable"}


# -- Gemini <-> ToolCall helpers (module-level, stateless) ----------------------


def _gemini_parts_to_tool_calls(
    fc_parts: list[tuple[int, Any]],
) -> tuple[list[ToolCall], dict[str, int]]:
    """Convert indexed Gemini ``Part`` objects to framework-agnostic ToolCalls.

    Returns:
        A tuple of (tool_calls, index_by_id) where index_by_id maps each
        ToolCall.id to its original index in llm_response.content.parts.
    """
    tool_calls = []
    index_by_id: dict[str, int] = {}
    for i, part in fc_parts:
        tc = ToolCall(
            name=part.function_call.name,
            arguments=json.dumps(
                dict(part.function_call.args) if part.function_call.args else {}
            ),
        )
        tool_calls.append(tc)
        index_by_id[tc.id] = i
    return tool_calls, index_by_id


def _apply_policy_result(
    llm_response: Any,
    index_by_id: dict[str, int],
    result: PolicyResult,
) -> Any:
    """Remove blocked function_call parts and inject explanation text."""
    blocked_indices = {
        index_by_id[tc.id] for tc in result.blocked if tc.id in index_by_id
    }

    new_parts = [
        part
        for i, part in enumerate(llm_response.content.parts)
        if i not in blocked_indices
    ]
    if result.explanation:
        new_parts.append(types.Part(text=result.explanation))

    llm_response.content = types.Content(
        role=llm_response.content.role, parts=new_parts
    )
    return llm_response


def _handle_error(llm_response: Any, fail_open: bool) -> Any:
    """Handle webhook errors: fail-open passes through, fail-closed strips all tools."""
    if fail_open:
        log.warning("Webhook unreachable — failing open, all tools allowed")
        return None

    new_parts = [
        p for p in llm_response.content.parts if p.function_call is None
    ]
    new_parts.append(
        types.Part(text="Tool execution suspended: policy check unavailable")
    )
    llm_response.content = types.Content(
        role=llm_response.content.role, parts=new_parts
    )
    return llm_response
