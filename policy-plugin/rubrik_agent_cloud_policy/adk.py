"""ADK adapter for Rubrik Agent Cloud policy enforcement.

Thin wrapper around :class:`RubrikPolicyClient` that implements Google ADK's
:class:`BasePlugin` callbacks.  All policy logic lives in the core client;
this module only handles Gemini <-> OpenAI conversion.

Requires the ``adk`` extra::

    pip install rubrik-agent-cloud-policy-plugin[adk]
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, replace
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

from rubrik_agent_cloud_policy.client import (
    PolicyContext,
    PolicyResult,
    RubrikPolicyClient,
    ToolCall,
)

log = logging.getLogger(__name__)

_WEBHOOK_ERRORS = (
    httpx.HTTPError,
    httpx.StreamError,
    httpx.InvalidURL,
    ValueError,
    KeyError,
    TypeError,
)

_CHECK_UNAVAILABLE = "policy check unavailable"
_REDACTED_PROMPT = "[Prompt blocked by Rubrik Agent Cloud policy]"

# Bounds the per-invocation bookkeeping if after_run_callback never runs for
# some invocations (for example, when a run raises).
_MAX_TRACKED_INVOCATIONS = 256


@dataclass
class _InvocationState:
    """Per-invocation bookkeeping shared across callbacks."""

    # Latest model request per agent, used as the context of later checks.
    requests: dict[str, PolicyContext] = field(default_factory=dict)
    # Signatures of tool calls that after_model_callback already allowed.
    approved_calls: Counter = field(default_factory=Counter)
    # Function call IDs of tools that before_tool_callback blocked.
    blocked_call_ids: set = field(default_factory=set)
    # Refusal for a blocked user prompt, returned by before_run_callback.
    prompt_refusal: Optional[str] = None


class RubrikPolicyPlugin(BasePlugin):
    """ADK plugin for Rubrik Agent Cloud policy enforcement.

    Uses :class:`RubrikPolicyClient` to evaluate agent activity against the
    Rubrik webhook service.  Configuration is read from environment variables.
    Each check sends the conversation so far, the tools declared to the
    model, and the agent ID.

    - ``on_user_message_callback`` and ``before_run_callback``: check a new
      user prompt before it is saved to the session.  A blocked prompt is
      saved as a placeholder, and the run ends with the refusal before the
      model is called.
    - ``before_model_callback``: records the model request as context for
      the checks below.  It does not call the webhook.
    - ``after_model_callback``: checks the model response.  Blocked tool
      calls are removed before ADK runs them; a blocked response is replaced
      with the refusal.
    - ``before_tool_callback``: safety net that checks a tool call before it
      runs, unless ``after_model_callback`` already allowed the same call.
      This catches tool calls that bypassed ``after_model_callback``, which
      happens when another plugin's ``before_model_callback`` returns a
      cached response.
    - ``after_tool_callback``: checks each tool result before the model
      sees it.  A blocked result is replaced with the refusal.

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
        self._invocations: dict[str, _InvocationState] = {}

    def _get_policy_client(self) -> RubrikPolicyClient:
        if self._policy_client is None:
            if self._kwargs:
                self._policy_client = RubrikPolicyClient(**self._kwargs)
            else:
                self._policy_client = RubrikPolicyClient.from_env()
        return self._policy_client

    def _state(self, ctx: Any) -> _InvocationState:
        key = _invocation_id(ctx)
        state = self._invocations.get(key)
        if state is None:
            while len(self._invocations) >= _MAX_TRACKED_INVOCATIONS:
                self._invocations.pop(next(iter(self._invocations)))
            state = self._invocations[key] = _InvocationState()
        return state

    async def close(self) -> None:
        if self._policy_client:
            await self._policy_client.close()
        log.info("[%s] Shut down", self.PLUGIN_NAME)

    # -- callbacks ------------------------------------------------------------

    async def on_user_message_callback(
        self, *, invocation_context: Any, user_message: Any
    ) -> Any:
        """Check a new user prompt before ADK saves it to the session."""
        if not _is_user_prompt(user_message):
            return None

        try:
            client = self._get_policy_client()
            context = _session_context(
                invocation_context, tail=_contents_to_messages([user_message])
            )
            refusal = await client.evaluate_prompt(context)
        except _WEBHOOK_ERRORS:
            log.exception(
                "[%s] Webhook call failed in on_user_message_callback", self.PLUGIN_NAME
            )
            if self._get_policy_client().fail_open:
                return None
            refusal = f"Request blocked: {_CHECK_UNAVAILABLE}"

        if refusal is None:
            return None
        log.info("[%s] Blocked user prompt", self.PLUGIN_NAME)
        self._state(invocation_context).prompt_refusal = refusal
        return types.Content(role="user", parts=[types.Part(text=_REDACTED_PROMPT)])

    async def before_run_callback(self, *, invocation_context: Any) -> Any:
        """End the run with the refusal when the user prompt was blocked."""
        state = self._invocations.get(_invocation_id(invocation_context))
        if state is None or state.prompt_refusal is None:
            return None
        refusal, state.prompt_refusal = state.prompt_refusal, None
        return types.Content(role="model", parts=[types.Part(text=refusal)])

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        self._invocations.pop(_invocation_id(invocation_context), None)

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: Any
    ) -> Any:
        """Record the model request as context for the response check."""
        try:
            self._state(callback_context).requests[_agent_name(callback_context)] = (
                _request_context(llm_request, callback_context)
            )
        except Exception:  # context is best effort; never fail the model call
            log.debug("[%s] Could not record model request", self.PLUGIN_NAME, exc_info=True)
        return None

    async def after_model_callback(
        self, *, callback_context: Any, llm_response: Any
    ) -> Any:
        """Check the model response; strip blocked tool calls or replace it."""
        if not llm_response or not llm_response.content:
            return None
        if not llm_response.content.parts:
            return None

        fc_parts = [
            (i, part)
            for i, part in enumerate(llm_response.content.parts)
            if part.function_call is not None
        ]
        # Streamed text chunks are checked once ADK aggregates them into the
        # final response.  Tool calls are checked in every response.
        partial = bool(getattr(llm_response, "partial", False))
        text = "" if partial else _response_text(llm_response.content)
        if not fc_parts and not text:
            return None

        try:
            client = self._get_policy_client()
            tool_calls, index_by_id = _gemini_parts_to_tool_calls(fc_parts)
            context = self._model_context(callback_context)
            result = await client.evaluate_response(tool_calls, text, context)
        except _WEBHOOK_ERRORS:
            log.exception(
                "[%s] Webhook call failed in after_model_callback", self.PLUGIN_NAME
            )
            return _handle_error(llm_response, self._get_policy_client().fail_open)

        if result.response_blocked:
            log.info("[%s] Blocked model response", self.PLUGIN_NAME)
            llm_response.content = types.Content(
                role=llm_response.content.role or "model",
                parts=[types.Part(text=result.explanation)],
            )
            return llm_response

        if not partial:
            approved = self._state(callback_context).approved_calls
            for tc in result.allowed:
                approved[_call_signature(tc.name, tc.arguments)] += 1

        if not result.blocked:
            return None

        log.info(
            "[%s] Blocked tools: %s",
            self.PLUGIN_NAME,
            [tc.name for tc in result.blocked],
        )
        return _apply_policy_result(llm_response, index_by_id, result)

    async def before_tool_callback(
        self, *, tool: Any, tool_args: Any, tool_context: Any
    ) -> Optional[dict[str, str]]:
        """Safety net — evaluates a tool call that was not already checked.

        Catches tools that bypassed ``after_model_callback`` — this happens
        when another plugin's ``before_model_callback`` returns a cached LLM
        response, causing ADK to skip the entire ``after_model_callback``
        chain.
        """
        arguments = json.dumps(tool_args or {}, default=str)
        approved = self._state(tool_context).approved_calls
        signature = _call_signature(tool.name, arguments)
        if approved[signature] > 0:
            approved[signature] -= 1
            return None

        try:
            client = self._get_policy_client()
            tc = ToolCall(name=tool.name, arguments=arguments)
            context = self._session_context(tool_context)
            result = await client.evaluate_tool_calls([tc], context)
        except _WEBHOOK_ERRORS:
            log.exception(
                "[%s] Webhook call failed in before_tool_callback", self.PLUGIN_NAME
            )
            if self._get_policy_client().fail_open:
                return None
            self._mark_blocked(tool_context)
            return {"error": f"Tool execution blocked: {_CHECK_UNAVAILABLE}"}

        if not result.blocked:
            return None

        log.info("[%s] Safety net blocked: %s", self.PLUGIN_NAME, tool.name)
        self._mark_blocked(tool_context)
        return {
            "error": result.explanation
            or f"Tool '{tool.name}' blocked by Rubrik Agent Cloud policy"
        }

    async def after_tool_callback(
        self, *, tool: Any, tool_args: Any, tool_context: Any, result: Any
    ) -> Optional[dict[str, str]]:
        """Check a tool result before the model sees it."""
        call_id = getattr(tool_context, "function_call_id", None) or ""
        state = self._invocations.get(_invocation_id(tool_context))
        if call_id and state is not None and call_id in state.blocked_call_ids:
            # ADK runs after-tool callbacks on the refusal that
            # before_tool_callback returned; there is no tool output to check.
            return None

        try:
            client = self._get_policy_client()
            context = self._session_context(
                tool_context, tail=[_tool_result_message(tool.name, call_id, result)]
            )
            refusal = await client.evaluate_prompt(context)
        except _WEBHOOK_ERRORS:
            log.exception(
                "[%s] Webhook call failed in after_tool_callback", self.PLUGIN_NAME
            )
            if self._get_policy_client().fail_open:
                return None
            return {"error": f"Tool result withheld: {_CHECK_UNAVAILABLE}"}

        if refusal is None:
            return None
        log.info("[%s] Blocked tool result: %s", self.PLUGIN_NAME, tool.name)
        return {"error": refusal}

    # -- context --------------------------------------------------------------

    def _model_context(self, callback_context: Any) -> PolicyContext:
        """The recorded model request, or the session if none was recorded."""
        state = self._invocations.get(_invocation_id(callback_context))
        request = state.requests.get(_agent_name(callback_context)) if state else None
        return request or self._session_context(callback_context)

    def _session_context(
        self, ctx: Any, tail: Optional[list[dict[str, Any]]] = None
    ) -> PolicyContext:
        """Session history, with the system prompt and tools of the recorded
        model request when there is one."""
        context = _session_context(ctx, tail)
        state = self._invocations.get(_invocation_id(ctx))
        request = state.requests.get(_agent_name(ctx)) if state else None
        if request is None:
            return context
        messages = list(context.messages)
        if request.messages and request.messages[0].get("role") == "system":
            messages = [m for m in messages if m.get("role") != "system"]
            messages.insert(0, request.messages[0])
        return replace(
            context, messages=tuple(messages), tools=request.tools, model=request.model
        )

    def _mark_blocked(self, tool_context: Any) -> None:
        call_id = getattr(tool_context, "function_call_id", None)
        if call_id:
            self._state(tool_context).blocked_call_ids.add(call_id)


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
            arguments=_json_args(part.function_call.args),
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
        log.warning("Webhook unreachable — failing open, response allowed")
        return None

    new_parts = [
        p for p in llm_response.content.parts if p.function_call is None
    ]
    new_parts.append(
        types.Part(text=f"Tool execution suspended: {_CHECK_UNAVAILABLE}")
    )
    llm_response.content = types.Content(
        role=llm_response.content.role, parts=new_parts
    )
    return llm_response


def _call_signature(name: str, arguments: str) -> str:
    try:
        arguments = json.dumps(json.loads(arguments), sort_keys=True)
    except (TypeError, ValueError):
        pass
    return f"{name}\x00{arguments}"


# -- context building -----------------------------------------------------------


def _invocation_id(ctx: Any) -> str:
    try:
        return getattr(ctx, "invocation_id", "") or ""
    except Exception:
        return ""


def _agent_name(ctx: Any) -> str:
    try:
        return getattr(ctx, "agent_name", "") or ""
    except Exception:
        return ""


def _request_context(llm_request: Any, callback_context: Any) -> PolicyContext:
    """Build a PolicyContext from the LlmRequest that ADK is about to send."""
    config = getattr(llm_request, "config", None)
    messages: list[dict[str, Any]] = []
    system = _system_text(getattr(config, "system_instruction", None))
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(_contents_to_messages(getattr(llm_request, "contents", None) or []))
    return PolicyContext(
        messages=tuple(messages),
        tools=tuple(_tools_from_config(config)),
        model=getattr(llm_request, "model", None) or "",
        correlation_id=_invocation_id(callback_context),
    )


def _session_context(
    ctx: Any, tail: Optional[list[dict[str, Any]]] = None
) -> PolicyContext:
    """Build a PolicyContext from the session events of the current agent.

    ``ctx`` is an ADK InvocationContext, CallbackContext, or ToolContext.
    ``tail`` is appended as the turn being evaluated.
    """
    tail = tail or []
    if ctx is None:
        return PolicyContext(messages=tuple(tail))

    inv = getattr(ctx, "_invocation_context", None) or ctx
    messages: list[dict[str, Any]] = []
    instruction = getattr(getattr(inv, "agent", None), "instruction", None)
    if isinstance(instruction, str) and instruction:
        messages.append({"role": "system", "content": instruction})
    messages.extend(_contents_to_messages(_session_contents(inv)))
    messages.extend(tail)
    return PolicyContext(messages=tuple(messages), correlation_id=_invocation_id(ctx))


def _session_contents(inv: Any) -> list[Any]:
    """Session event contents visible to the current agent branch."""
    session = getattr(inv, "session", None)
    branch = getattr(inv, "branch", None)
    contents = []
    for event in getattr(session, "events", None) or []:
        if getattr(event, "partial", False):
            continue
        event_branch = getattr(event, "branch", None)
        if branch and event_branch and not (
            branch == event_branch or branch.startswith(f"{event_branch}.")
        ):
            continue
        if getattr(event, "content", None) is not None:
            contents.append(event.content)
    return contents


def _tools_from_config(config: Any) -> list[dict[str, Any]]:
    tools = []
    for tool in getattr(config, "tools", None) or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            tools.append(_declaration_to_openai(declaration))
    return tools


def _declaration_to_openai(declaration: Any) -> dict[str, Any]:
    function: dict[str, Any] = {"name": declaration.name or ""}
    if declaration.description:
        function["description"] = declaration.description
    parameters = getattr(declaration, "parameters_json_schema", None)
    if parameters is None and getattr(declaration, "parameters", None) is not None:
        try:
            parameters = declaration.parameters.json_schema.model_dump(
                mode="json", exclude_none=True
            )
        except Exception:
            parameters = None
    if parameters is not None:
        function["parameters"] = parameters
    return {"type": "function", "function": function}


# -- Gemini Content -> OpenAI messages --------------------------------------------


def _contents_to_messages(contents: list[Any]) -> list[dict[str, Any]]:
    """Convert Gemini Content objects to OpenAI chat messages.

    Text parts become ``user`` or ``assistant`` messages, function calls
    become assistant ``tool_calls``, and function responses become ``tool``
    messages.  Thought parts and non-text media are dropped.
    """
    messages: list[dict[str, Any]] = []
    for content in contents:
        parts = getattr(content, "parts", None) or []
        role = "assistant" if getattr(content, "role", None) == "model" else "user"
        texts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        for part in parts:
            if getattr(part, "thought", None):
                continue
            if part.function_call is not None:
                fc = part.function_call
                tool_calls.append(
                    {
                        "id": fc.id or f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {"name": fc.name or "", "arguments": _json_args(fc.args)},
                    }
                )
            elif part.function_response is not None:
                fr = part.function_response
                tool_results.append(
                    _tool_result_message(fr.name or "", fr.id or "", fr.response)
                )
            elif part.text:
                texts.append(part.text)

        if role == "assistant":
            if texts or tool_calls:
                message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
                if tool_calls:
                    message["tool_calls"] = tool_calls
                messages.append(message)
        else:
            messages.extend(tool_results)
            if texts:
                messages.append({"role": "user", "content": "\n".join(texts)})
    return messages


def _tool_result_message(name: str, call_id: str, result: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "tool",
        "name": name,
        "content": _json_text(result),
    }
    if call_id:
        message["tool_call_id"] = call_id
    return message


def _is_user_prompt(content: Any) -> bool:
    """Whether the content is a user turn with text (not a tool result)."""
    if getattr(content, "role", None) not in (None, "user"):
        return False
    parts = getattr(content, "parts", None) or []
    if any(p.function_response is not None for p in parts):
        return False
    return any(p.text for p in parts)


def _response_text(content: Any) -> str:
    return "\n".join(
        p.text
        for p in content.parts
        if p.text and not getattr(p, "thought", None)
    )


def _system_text(instruction: Any) -> str:
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    parts = getattr(instruction, "parts", None)
    if parts is not None:
        return "\n".join(p.text for p in parts if getattr(p, "text", None))
    if isinstance(instruction, list):
        return "\n".join(t for t in (_system_text(i) for i in instruction) if t)
    return getattr(instruction, "text", None) or ""


def _json_args(args: Any) -> str:
    return json.dumps(dict(args) if args else {}, default=str)


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)
