"""ADK adapter for Rubrik Agent Cloud policy enforcement.

Thin wrapper around :class:`RubrikPolicyClient` that implements Google ADK's
:class:`BasePlugin` callbacks.  All policy logic lives in the core client;
this module only handles Gemini <-> OpenAI conversion.

Requires the ``adk`` extra::

    pip install rubrik-agent-cloud-policy-plugin[adk]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Optional

try:
    from google.adk.flows.llm_flows.functions import generate_client_function_call_id
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

_CHECK_UNAVAILABLE = "policy check unavailable"
_REDACTED_PROMPT = "[Prompt blocked by Rubrik Agent Cloud policy]"

# Bounds the per-invocation bookkeeping if after_run_callback never runs for
# some invocations (for example, when a run raises), and the record of
# prompts blocked after ADK saved them.
_MAX_TRACKED_INVOCATIONS = 256


@dataclass
class _InvocationState:
    """Per-invocation bookkeeping shared across callbacks."""

    # Latest model request per agent, used as the context of later checks.
    requests: dict[str, PolicyContext] = field(default_factory=dict)
    # Function call IDs that after_model_callback allowed or failed open on.
    # IDs are unique per call, so an approval never covers a later call.
    approved_call_ids: set = field(default_factory=set)
    # Function call IDs of tools that before_tool_callback blocked.
    blocked_call_ids: set = field(default_factory=set)
    # Refusal for a blocked user prompt, returned by before_run_callback.
    prompt_refusal: Optional[str] = None
    # Whether on_user_message_callback ran for this invocation.
    message_checked: bool = False


class RubrikPolicyPlugin(BasePlugin):
    """ADK plugin for Rubrik Agent Cloud policy enforcement.

    Uses :class:`RubrikPolicyClient` to evaluate agent activity against the
    Rubrik webhook service.  Configuration is read from environment variables.
    Each check sends the conversation so far, the tools declared to the
    model, and the agent ID.

    - ``on_user_message_callback`` and ``before_run_callback``: check a new
      user prompt before it is saved to the session.  A blocked prompt is
      saved as a placeholder, and the run ends with the refusal before the
      model is called.  Tool results that the client sends in the message
      (long-running tools) are checked too; a blocked one is replaced with
      the refusal.
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

    ADK stops calling plugins for a callback once one returns a value, so a
    plugin earlier in the list could hide activity from this one.  The
    plugin moves itself to the front of the runner's plugin list at the
    start and end of each run, and ``auto_instrument`` registers it first.
    If a plugin listed before it in ``plugins=[...]`` returns a value from
    ``on_user_message_callback`` on a runner's first run, ADK saves the
    prompt before this plugin checks it; ``before_run_callback`` then checks
    it and ends the run, and ``before_model_callback`` removes it from later
    model requests in this process.

    Any failure of a check, including a malformed webhook response, follows
    ``fail_open``.  Misconfiguration raises instead: constructor arguments
    are validated at construction, and without them a missing
    ``RUBRIK_WEBHOOK_URL`` raises ``RuntimeError`` at the first callback.

    Without constructor arguments, the client is built on first use so the
    plugin can be created locally (for cloudpickle) without
    ``RUBRIK_WEBHOOK_URL``.  Pickling (and deep copying, which
    ``AdkApp.clone`` does before deploying) drops the client and the
    ``api_key`` argument; the restored plugin reads the key from
    ``RUBRIK_API_KEY``.

    Callbacks may run concurrently on different threads (``Runner.run``
    runs each call on its own thread), so the per-invocation bookkeeping is
    guarded by a lock.
    """

    PLUGIN_NAME = "rubrik_agent_cloud_policy"

    def __init__(self, name: str = PLUGIN_NAME, **kwargs: Any) -> None:
        super().__init__(name=name)
        # Kept apart from the other settings so that pickling drops it.
        self._api_key: Optional[str] = kwargs.pop("api_key", None)
        self._kwargs = kwargs
        self._policy_client: Optional[RubrikPolicyClient] = None
        self._lock = threading.Lock()
        self._invocations: dict[str, _InvocationState] = {}
        # Invocation IDs whose prompt was blocked after ADK saved it.
        self._redacted_invocations: dict[str, None] = {}
        if kwargs or self._api_key:
            # Validate explicit settings now. Without them, defer: the
            # environment may lack RUBRIK_WEBHOOK_URL at pickle time.
            self._get_policy_client()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update(
            _api_key=None,
            _policy_client=None,
            _lock=None,
            _invocations={},
            _redacted_invocations={},
        )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _get_policy_client(self) -> RubrikPolicyClient:
        with self._lock:
            if self._policy_client is None:
                if self._kwargs or self._api_key:
                    api_key = self._api_key or os.getenv(RubrikPolicyClient._API_KEY_ENV)
                    self._policy_client = RubrikPolicyClient(api_key=api_key, **self._kwargs)
                else:
                    self._policy_client = RubrikPolicyClient.from_env()
            return self._policy_client

    def _state(self, ctx: Any) -> _InvocationState:
        """The invocation's bookkeeping, created if needed.  Hold ``_lock``."""
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
        """Check a new user message before ADK saves it to the session.

        Checks each tool result in the message and then its text.  Blocked
        tool results are replaced with the refusal.  When the text is
        blocked, every part except tool results is replaced with a
        placeholder, and the refusal is held for ``before_run_callback``.
        """
        self._move_to_front(invocation_context)
        with self._lock:
            self._state(invocation_context).message_checked = True
        parts = list(getattr(user_message, "parts", None) or [])
        if not parts:
            return None

        session = _session_context(invocation_context)
        new_parts, tool_messages = await self._check_tool_results(session, parts)
        changed = new_parts != parts

        text = _parts_text(parts)
        if text:
            refusal = await self._check_tail(
                session, [*tool_messages, {"role": "user", "content": text}], "Request blocked"
            )
            if refusal is not None:
                log.info("[%s] Blocked user prompt", self.PLUGIN_NAME)
                with self._lock:
                    self._state(invocation_context).prompt_refusal = refusal
                changed = True
                # Drop the text and any media or files that came with it;
                # keep the (already checked) tool results.
                new_parts = [p for p in new_parts if p.function_response is not None]
                new_parts.append(types.Part(text=_REDACTED_PROMPT))

        if not changed:
            return None
        return types.Content(role=user_message.role or "user", parts=new_parts)

    async def before_run_callback(self, *, invocation_context: Any) -> Any:
        """End the run with the refusal when the user prompt was blocked.

        If a plugin listed earlier returned a value from
        ``on_user_message_callback``, this plugin's was skipped and ADK
        saved the prompt unchecked.  The prompt is then checked here, before
        the model is called; if it is blocked, the run ends and
        ``before_model_callback`` keeps it out of later model requests.
        """
        self._move_to_front(invocation_context)
        with self._lock:
            state = self._state(invocation_context)
            refusal, state.prompt_refusal = state.prompt_refusal, None
            fallback = refusal is None and not state.message_checked
            state.message_checked = True
        if fallback and _is_user_prompt(getattr(invocation_context, "user_content", None)):
            refusal = await self._check_tail(
                _session_context(invocation_context), [], "Request blocked"
            )
            if refusal is not None:
                log.warning(
                    "[%s] Blocked a user prompt that ADK already saved, because a "
                    "plugin listed before this one handled on_user_message_callback",
                    self.PLUGIN_NAME,
                )
                with self._lock:
                    redacted = self._redacted_invocations
                    redacted[_invocation_id(invocation_context)] = None
                    while len(redacted) > _MAX_TRACKED_INVOCATIONS:
                        redacted.pop(next(iter(redacted)))
        if refusal is None:
            return None
        return types.Content(role="model", parts=[types.Part(text=refusal)])

    async def after_run_callback(self, *, invocation_context: Any) -> None:
        self._move_to_front(invocation_context)
        with self._lock:
            self._invocations.pop(_invocation_id(invocation_context), None)

    async def before_model_callback(
        self, *, callback_context: Any, llm_request: Any
    ) -> Any:
        """Record the model request as context for the response check.

        Also removes prompts that ``before_run_callback`` blocked after ADK
        saved them.
        """
        try:
            self._redact_saved_prompts(callback_context, llm_request)
            context = _request_context(llm_request, callback_context)
            with self._lock:
                self._state(callback_context).requests[_agent_name(callback_context)] = context
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

        # ADK runs tool calls only from non-partial responses: the final
        # aggregated response in progressive SSE streaming, or the chunk
        # itself otherwise.  Partial chunks are checked through that response.
        if getattr(llm_response, "partial", False):
            return None

        fc_parts = [
            (i, part)
            for i, part in enumerate(llm_response.content.parts)
            if part.function_call is not None
        ]
        text = _response_text(llm_response.content)
        if not fc_parts and not text:
            return None

        client = self._get_policy_client()
        tool_calls: list[ToolCall] = []
        try:
            tool_calls, index_by_id = _gemini_parts_to_tool_calls(fc_parts)
            context = self._model_context(callback_context)
            result = await client.evaluate_response(tool_calls, text, context)
        except Exception:
            log.exception("[%s] Policy check failed in after_model_callback", self.PLUGIN_NAME)
            if not client.fail_open:
                return _fail_closed_response(llm_response, bool(fc_parts))
            # Failing open allowed these calls; don't make
            # before_tool_callback wait out the same failure again.
            log.warning("[%s] Failing open: response allowed", self.PLUGIN_NAME)
            self._approve(callback_context, tool_calls)
            return None

        if result.response_blocked:
            log.info("[%s] Blocked model response", self.PLUGIN_NAME)
            llm_response.content = types.Content(
                role=llm_response.content.role or "model",
                parts=[types.Part(text=result.explanation)],
            )
            return llm_response

        self._approve(callback_context, result.allowed)

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
        call_id = getattr(tool_context, "function_call_id", None) or ""
        with self._lock:
            approved = self._state(tool_context).approved_call_ids
            if call_id in approved:
                approved.discard(call_id)
                return None

        client = self._get_policy_client()
        try:
            tc = ToolCall(name=tool.name, arguments=json.dumps(tool_args or {}, default=str))
            context = self._session_context(tool_context)
            result = await client.evaluate_tool_calls([tc], context)
        except Exception:
            log.exception("[%s] Policy check failed in before_tool_callback", self.PLUGIN_NAME)
            if client.fail_open:
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
        with self._lock:
            state = self._invocations.get(_invocation_id(tool_context))
            if call_id and state is not None and call_id in state.blocked_call_ids:
                # ADK runs after-tool callbacks on the refusal that
                # before_tool_callback returned; there is no tool output to check.
                return None

        refusal = await self._check_tail(
            self._session_context(tool_context),
            [_tool_result_message(tool.name, call_id, result)],
            "Tool result withheld",
        )
        if refusal is None:
            return None
        log.info("[%s] Blocked tool result: %s", self.PLUGIN_NAME, tool.name)
        return {"error": refusal}

    # -- helpers --------------------------------------------------------------

    def _move_to_front(self, invocation_context: Any) -> None:
        plugins = getattr(
            getattr(invocation_context, "plugin_manager", None), "plugins", None
        )
        if not isinstance(plugins, list):
            return
        with self._lock:
            if plugins and plugins[0] is not self and self in plugins:
                plugins.remove(self)
                plugins.insert(0, self)

    async def _check_tool_results(
        self, session: PolicyContext, parts: list[Any]
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Check the tool results in ``parts`` concurrently.

        Each result is checked after the session and the results before it.
        Returns ``parts`` with blocked results replaced by the refusal, and
        the tool messages of all results.
        """
        indexed = [(i, p.function_response) for i, p in enumerate(parts) if p.function_response]
        tool_messages = [
            _tool_result_message(fr.name or "", fr.id or "", fr.response) for _, fr in indexed
        ]
        refusals = await asyncio.gather(
            *(
                self._check_tail(session, tool_messages[: n + 1], "Tool result withheld")
                for n in range(len(tool_messages))
            )
        )
        new_parts = list(parts)
        for (i, fr), refusal in zip(indexed, refusals):
            if refusal is None:
                continue
            log.info("[%s] Blocked tool result: %s", self.PLUGIN_NAME, fr.name)
            new_parts[i] = types.Part(
                function_response=types.FunctionResponse(
                    id=fr.id, name=fr.name, response={"error": refusal}
                )
            )
        return new_parts, tool_messages

    async def _check_tail(
        self, session: PolicyContext, tail: list[dict[str, Any]], unavailable_prefix: str
    ) -> Optional[str]:
        """Check the last of ``tail``, after ``session``, with before_prompt.

        Returns the refusal, or ``None`` when allowed.  When the check fails
        returns ``None`` when failing open, else an unavailability refusal.
        """
        client = self._get_policy_client()
        try:
            context = replace(session, messages=(*session.messages, *tail))
            return await client.evaluate_prompt(context)
        except Exception:
            log.exception("[%s] Policy check failed for a message", self.PLUGIN_NAME)
            if client.fail_open:
                return None
            return f"{unavailable_prefix}: {_CHECK_UNAVAILABLE}"

    def _redact_saved_prompts(self, ctx: Any, llm_request: Any) -> None:
        """Replace prompts that were blocked after ADK saved them."""
        with self._lock:
            redacted = set(self._redacted_invocations)
        if not redacted:
            return
        inv = getattr(ctx, "_invocation_context", None) or ctx
        blocked = [
            event.content
            for event in getattr(getattr(inv, "session", None), "events", None) or []
            if getattr(event, "invocation_id", None) in redacted
            and getattr(event, "author", None) == "user"
            and event.content is not None
        ]
        contents = getattr(llm_request, "contents", None) or []
        for i, content in enumerate(contents):
            if any(_same_content(content, b) for b in blocked):
                contents[i] = types.Content(role="user", parts=[types.Part(text=_REDACTED_PROMPT)])

    def _model_context(self, callback_context: Any) -> PolicyContext:
        """The recorded model request, or the session if none was recorded."""
        with self._lock:
            state = self._invocations.get(_invocation_id(callback_context))
            request = state.requests.get(_agent_name(callback_context)) if state else None
        return request or self._session_context(callback_context)

    def _session_context(self, ctx: Any) -> PolicyContext:
        """Session history, with the system prompt and tools of the recorded
        model request when there is one."""
        context = _session_context(ctx)
        with self._lock:
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

    def _approve(self, ctx: Any, tool_calls: Iterable[ToolCall]) -> None:
        """Let ``before_tool_callback`` skip these calls of the current response."""
        with self._lock:
            self._state(ctx).approved_call_ids.update(tc.id for tc in tool_calls)

    def _mark_blocked(self, tool_context: Any) -> None:
        call_id = getattr(tool_context, "function_call_id", None)
        if call_id:
            with self._lock:
                self._state(tool_context).blocked_call_ids.add(call_id)


# -- Gemini <-> ToolCall helpers (module-level, stateless) ----------------------


def _gemini_parts_to_tool_calls(
    fc_parts: list[tuple[int, Any]],
) -> tuple[list[ToolCall], dict[str, int]]:
    """Convert indexed Gemini ``Part`` objects to framework-agnostic ToolCalls.

    A function call without an ID, or with an ID repeated in the response,
    gets one the way ADK assigns them, before ADK would; ADK keeps it, so
    ``before_tool_callback`` sees the same ID in
    ``tool_context.function_call_id``.

    Returns:
        A tuple of (tool_calls, index_by_id) where index_by_id maps each
        ToolCall.id to its original index in llm_response.content.parts.
    """
    tool_calls = []
    index_by_id: dict[str, int] = {}
    for i, part in fc_parts:
        fc = part.function_call
        if not fc.id or fc.id in index_by_id:
            fc.id = generate_client_function_call_id()
        tc = ToolCall(name=fc.name, arguments=_json_args(fc.args), id=fc.id)
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


def _fail_closed_response(llm_response: Any, had_tool_calls: bool) -> Any:
    """Replace an unchecked response, text and tool calls, with a notice."""
    subject = "Response and tool calls" if had_tool_calls else "Response"
    llm_response.content = types.Content(
        role=llm_response.content.role or "model",
        parts=[types.Part(text=f"{subject} withheld: {_CHECK_UNAVAILABLE}")],
    )
    return llm_response


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


def _session_context(ctx: Any) -> PolicyContext:
    """Build a PolicyContext from the session events of the current agent.

    ``ctx`` is an ADK InvocationContext, CallbackContext, or ToolContext.
    """
    if ctx is None:
        return PolicyContext()

    inv = getattr(ctx, "_invocation_context", None) or ctx
    messages: list[dict[str, Any]] = []
    instruction = getattr(getattr(inv, "agent", None), "instruction", None)
    if isinstance(instruction, str) and instruction:
        messages.append({"role": "system", "content": instruction})
    messages.extend(_contents_to_messages(_session_contents(inv)))
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
    return _parts_text(content.parts)


def _parts_text(parts: list[Any]) -> str:
    """The non-thought text of ``parts``."""
    return "\n".join(p.text for p in parts if p.text and not getattr(p, "thought", None))


def _same_content(a: Any, b: Any) -> bool:
    """Whether two Contents have the same role and parts."""
    try:
        return a.role == b.role and [p.model_dump() for p in a.parts or []] == [
            p.model_dump() for p in b.parts or []
        ]
    except AttributeError:
        return False


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
