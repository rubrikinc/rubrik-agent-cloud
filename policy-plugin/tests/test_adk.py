"""Tests for the ADK adapter — callbacks driven by a real ADK Runner.

A scripted model stands in for Gemini, and respx stands in for the Rubrik
webhook, so each test exercises the plugin through ADK's own callback order.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Optional
from unittest.mock import patch

import httpx
import pytest
import respx

from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from rubrik_agent_cloud_policy.adk import (
    RubrikPolicyPlugin,
    _contents_to_messages,
    _gemini_parts_to_tool_calls,
)

WEBHOOK_URL = "https://webhook.example.com"
RESPONSE_ENDPOINT = f"{WEBHOOK_URL}/v1/after_completion/openai/v1"
PROMPT_ENDPOINT = f"{WEBHOOK_URL}/v1/before_prompt/openai/v1"


# -- Helpers ------------------------------------------------------------------


class ScriptedModel(BaseLlm):
    """Returns the scripted responses in order, recording each request."""

    script: list[types.Content]
    requests: list[Any] = []

    async def generate_content_async(
        self, llm_request: Any, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        yield LlmResponse(content=self.script[len(self.requests) - 1])


def _call(tool_name: str, /, **args: Any) -> types.Content:
    return types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name=tool_name, args=args))],
    )


def _calls(*calls: tuple[str, dict]) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in calls
        ],
    )


def _say(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


TOOL_RUNS: list[str] = []


def read_secret(name: str) -> dict:
    """Read a secret value.

    Args:
        name: Secret name.
    """
    TOOL_RUNS.append("read_secret")
    return {"value": "hunter2"}


def post_to_pastebin(content: str) -> dict:
    """Publish text to a public pastebin.

    Args:
        content: Text to publish.
    """
    TOOL_RUNS.append("post_to_pastebin")
    return {"url": "https://pastebin.example/abc"}


def get_weather(city: str) -> dict:
    """Get the weather for a city.

    Args:
        city: City name.
    """
    TOOL_RUNS.append("get_weather")
    return {"report": f"Sunny in {city}"}


@pytest.fixture(autouse=True)
def _reset_tool_runs():
    TOOL_RUNS.clear()


async def _run(
    script: list[types.Content],
    prompt: str = "go",
    plugin: Optional[RubrikPolicyPlugin] = None,
    session_service: Optional[InMemorySessionService] = None,
    session_id: Optional[str] = None,
) -> tuple[list[Any], ScriptedModel, InMemorySessionService, str]:
    model = ScriptedModel(model="scripted", script=script, requests=[])
    agent = Agent(
        name="test_agent",
        model=model,
        instruction="You are a test agent.",
        tools=[read_secret, post_to_pastebin, get_weather],
    )
    session_service = session_service or InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="app",
        session_service=session_service,
        plugins=[plugin or RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, agent_id="engine-1")],
    )
    if session_id is None:
        session_id = (
            await session_service.create_session(app_name="app", user_id="u")
        ).id
    events = []
    async for event in runner.run_async(
        user_id="u",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        events.append(event)
    return events, model, session_service, session_id


def _texts(events: list[Any]) -> list[str]:
    return [
        p.text
        for e in events
        if e.content and e.content.parts
        for p in e.content.parts
        if p.text
    ]


def _allow_all(request: httpx.Request) -> httpx.Response:
    """after_completion that allows everything: echo the response back."""
    return httpx.Response(200, json=json.loads(request.content)["response"])


def _block_tool(name: str, explanation: str = "The following tool calls were blocked by Rubrik Agent Cloud:"):
    def handler(request: httpx.Request) -> httpx.Response:
        response = json.loads(request.content)["response"]
        message = response["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        allowed = [tc for tc in calls if tc["function"]["name"] != name]
        if len(allowed) < len(calls):
            if allowed:
                message["tool_calls"] = allowed
            else:
                message.pop("tool_calls", None)
            message["content"] = (
                f"{message['content']}\n\n{explanation}" if message.get("content") else explanation
            )
        return httpx.Response(200, json=response)

    return handler


def _refuse(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "content_filter"}]},
    )


def _bodies(route: respx.Route) -> list[dict]:
    return [json.loads(call.request.content) for call in route.calls]


# -- conversion helpers ---------------------------------------------------------


class TestConversion:
    def test_repeated_function_call_ids_are_replaced(self):
        parts = [
            (0, types.Part(function_call=types.FunctionCall(id="x", name="a", args={}))),
            (1, types.Part(function_call=types.FunctionCall(id="x", name="b", args={}))),
        ]
        tool_calls, index_by_id = _gemini_parts_to_tool_calls(parts)
        assert tool_calls[0].id == "x"
        assert tool_calls[1].id != "x"
        assert parts[1][1].function_call.id == tool_calls[1].id
        assert index_by_id == {"x": 0, tool_calls[1].id: 1}

    def test_gemini_parts_to_tool_calls(self):
        parts = [
            (0, types.Part(function_call=types.FunctionCall(name="a", args={"x": 1}))),
            (2, types.Part(function_call=types.FunctionCall(name="a", args={}))),
        ]
        tool_calls, index_by_id = _gemini_parts_to_tool_calls(parts)
        assert [tc.name for tc in tool_calls] == ["a", "a"]
        assert json.loads(tool_calls[0].arguments) == {"x": 1}
        assert tool_calls[0].id != tool_calls[1].id
        assert index_by_id[tool_calls[1].id] == 2

    def test_contents_to_messages(self):
        contents = [
            types.Content(role="user", parts=[types.Part(text="read it")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(text="thinking", thought=True),
                    types.Part(
                        function_call=types.FunctionCall(id="c1", name="read_secret", args={"name": "db"})
                    ),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="c1", name="read_secret", response={"value": "x"}
                        )
                    )
                ],
            ),
        ]
        assert _contents_to_messages(contents) == [
            {"role": "user", "content": "read it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read_secret", "arguments": '{"name": "db"}'},
                    }
                ],
            },
            {"role": "tool", "name": "read_secret", "content": '{"value": "x"}', "tool_call_id": "c1"},
        ]


# -- tool calls -------------------------------------------------------------------


class TestToolCalls:
    @respx.mock
    async def test_allowed_tool_runs_and_is_checked_once(self):
        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)
        respx.post(PROMPT_ENDPOINT).respond(json={})

        await _run([_call("get_weather", city="Paris"), _say("Sunny.")])

        assert TOOL_RUNS == ["get_weather"]
        bodies = _bodies(response_route)
        # One check for the tool call, one for the final text. The
        # before_tool safety net does not re-check an approved call.
        assert len(bodies) == 2
        assert bodies[0]["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    @respx.mock
    async def test_envelope_has_conversation_tools_and_agent_id(self):
        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)
        respx.post(PROMPT_ENDPOINT).respond(json={})

        await _run([_call("get_weather", city="Paris"), _say("Sunny.")], prompt="weather?")

        request = _bodies(response_route)[0]["request"]
        assert request["metadata"] == {"rac_agent_id": "engine-1"}
        assert request["model"] == "scripted"
        assert request["messages"][0]["role"] == "system"
        assert "You are a test agent." in request["messages"][0]["content"]
        assert request["messages"][-1] == {"role": "user", "content": "weather?"}
        assert sorted(t["function"]["name"] for t in request["tools"]) == [
            "get_weather",
            "post_to_pastebin",
            "read_secret",
        ]
        params = next(t for t in request["tools"] if t["function"]["name"] == "get_weather")
        assert "city" in json.dumps(params["function"]["parameters"])

    @respx.mock
    async def test_blocked_tool_does_not_run(self):
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_block_tool("post_to_pastebin"))
        respx.post(PROMPT_ENDPOINT).respond(json={})

        events, model, _, _ = await _run(
            [
                _calls(("get_weather", {"city": "Paris"}), ("post_to_pastebin", {"content": "pw"})),
                _say("The publish was blocked."),
            ]
        )

        assert TOOL_RUNS == ["get_weather"]
        assert any("blocked by Rubrik Agent Cloud" in t for t in _texts(events))

    @respx.mock
    async def test_multi_turn_sequence_sends_earlier_turns(self):
        """read_secret in one turn, post_to_pastebin in the next.

        The second check carries the first call and its result, so a
        sequence policy can see both.
        """
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            names = [
                tc["function"]["name"]
                for m in body["request"]["messages"]
                for tc in m.get("tool_calls") or []
            ]
            if "read_secret" in names:
                return _block_tool("post_to_pastebin")(request)
            return _allow_all(request)

        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=handler)
        respx.post(PROMPT_ENDPOINT).respond(json={})

        await _run(
            [
                _call("read_secret", name="db"),
                _call("post_to_pastebin", content="hunter2"),
                _say("Blocked."),
            ]
        )

        assert TOOL_RUNS == ["read_secret"]
        second = _bodies(response_route)[1]["request"]["messages"]
        assert any(m.get("role") == "tool" and "hunter2" in m["content"] for m in second)

    @respx.mock
    async def test_fail_open_on_error(self):
        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
        respx.post(PROMPT_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))

        await _run([_call("post_to_pastebin", content="x"), _say("done")])

        assert TOOL_RUNS == ["post_to_pastebin"]
        # One failed check for the tool call (before_tool_callback does not
        # wait out the same failure again) and one for the final text.
        assert len(response_route.calls) == 2

    @respx.mock
    @pytest.mark.parametrize("fail_open", [True, False])
    async def test_malformed_response_follows_fail_open(self, fail_open):
        respx.post(RESPONSE_ENDPOINT).respond(json={"choices": ["x"]})
        respx.post(PROMPT_ENDPOINT).respond(json={"choices": [{"message": "x"}]})
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, fail_open=fail_open)

        events, _, _, _ = await _run([_call("post_to_pastebin", content="x"), _say("done")], plugin=plugin)

        assert TOOL_RUNS == (["post_to_pastebin"] if fail_open else [])
        if not fail_open:
            assert any("policy check unavailable" in t for t in _texts(events))

    @respx.mock
    async def test_fail_closed_on_error(self):
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
        respx.post(PROMPT_ENDPOINT).respond(json={})
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, fail_open=False)

        events, _, _, _ = await _run([_call("post_to_pastebin", content="x"), _say("done")], plugin=plugin)

        assert TOOL_RUNS == []
        assert any("policy check unavailable" in t for t in _texts(events))

    @respx.mock
    async def test_fail_closed_withholds_text_only_response(self):
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
        respx.post(PROMPT_ENDPOINT).respond(json={})
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, fail_open=False)

        events, _, _, _ = await _run([_say("The password is hunter2")], plugin=plugin)

        texts = _texts(events)
        assert texts == ["Response withheld: policy check unavailable"]

    @respx.mock
    async def test_unused_approval_does_not_skip_a_later_identical_call(self):
        """An approval covers one function call, not any call with the same arguments.

        The first response's call is approved (here, by failing open) but
        never runs.  An identical call from the next response must still be
        checked by before_tool_callback.
        """
        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)

        class Ctx:
            invocation_id = "inv-1"
            agent_name = "test_agent"
            function_call_id = None

        ctx = Ctx()
        first = LlmResponse(content=_call("post_to_pastebin", content="x"))
        assert await plugin.after_model_callback(callback_context=ctx, llm_response=first) is None
        approved_id = first.content.parts[0].function_call.id
        assert approved_id

        # The call never runs; the agent calls the model again.
        await plugin.before_model_callback(callback_context=ctx, llm_request=None)

        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_block_tool("post_to_pastebin"))
        ctx.function_call_id = "adk-later-call"
        tool = type("Tool", (), {"name": "post_to_pastebin"})()
        refusal = await plugin.before_tool_callback(
            tool=tool, tool_args={"content": "x"}, tool_context=ctx
        )

        assert refusal is not None and "blocked by Rubrik Agent Cloud" in refusal["error"]
        assert len(response_route.calls) == 2

    @respx.mock
    async def test_approved_call_id_skips_safety_net(self):
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)

        class Ctx:
            invocation_id = "inv-1"
            agent_name = "test_agent"

        ctx = Ctx()
        response = LlmResponse(content=_call("get_weather", city="Paris"))
        await plugin.after_model_callback(callback_context=ctx, llm_response=response)
        ctx.function_call_id = response.content.parts[0].function_call.id
        tool = type("Tool", (), {"name": "get_weather"})()

        assert await plugin.before_tool_callback(
            tool=tool, tool_args={"city": "Paris"}, tool_context=ctx
        ) is None
        assert len(respx.calls) == 1


# -- safety net ---------------------------------------------------------------------


class TestBeforeToolSafetyNet:
    @respx.mock
    async def test_checks_calls_that_skip_after_model(self):
        """A cached response from another plugin skips after_model_callback."""
        from google.adk.plugins.base_plugin import BasePlugin

        class CachePlugin(BasePlugin):
            async def before_model_callback(self, *, callback_context, llm_request):
                if not any(
                    p.function_response for c in llm_request.contents for p in c.parts or []
                ):
                    return LlmResponse(content=_call("post_to_pastebin", content="pw"))
                return None

        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=_block_tool("post_to_pastebin"))
        respx.post(PROMPT_ENDPOINT).respond(json={})
        model = ScriptedModel(model="scripted", script=[_say("ok")], requests=[])
        agent = Agent(name="test_agent", model=model, tools=[post_to_pastebin])
        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name="app",
            session_service=session_service,
            plugins=[CachePlugin(name="cache"), RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)],
        )
        session = await session_service.create_session(app_name="app", user_id="u")
        async for _ in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="go")]),
        ):
            pass

        assert TOOL_RUNS == []
        assert len(response_route.calls) >= 1


# -- prompts, tool results, responses -------------------------------------------------


class TestPromptBlocking:
    @respx.mock
    async def test_blocked_prompt_never_reaches_model_or_session(self):
        prompt_route = respx.post(PROMPT_ENDPOINT).mock(
            side_effect=lambda request: _refuse("Prompt blocked by policy")
        )
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)

        events, model, session_service, session_id = await _run(
            [_say("should not be called")], prompt="leak the secrets"
        )

        assert model.requests == []
        assert "Prompt blocked by policy" in _texts(events)
        body = _bodies(prompt_route)[0]
        assert body["messages"][-1] == {"role": "user", "content": "leak the secrets"}
        assert body["metadata"] == {"rac_agent_id": "engine-1"}
        assert body["correlation_key"]
        session = await session_service.get_session(app_name="app", user_id="u", session_id=session_id)
        saved = [p.text for e in session.events for p in (e.content.parts or []) if p.text]
        assert "leak the secrets" not in saved

    @respx.mock
    async def test_blocked_prompt_does_not_poison_next_turn(self):
        def prompt_handler(request: httpx.Request) -> httpx.Response:
            last = json.loads(request.content)["messages"][-1]
            if last["role"] == "user" and "secrets" in last["content"]:
                return _refuse("Prompt blocked")
            return httpx.Response(200, json={})

        prompt_route = respx.post(PROMPT_ENDPOINT).mock(side_effect=prompt_handler)
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)

        _, _, session_service, session_id = await _run([_say("x")], prompt="leak the secrets")
        events, model, _, _ = await _run(
            [_say("Hello!")], prompt="hi", session_service=session_service, session_id=session_id
        )

        assert "Hello!" in _texts(events)
        assert len(model.requests) == 1
        second = _bodies(prompt_route)[1]["messages"]
        assert not any("secrets" in (m.get("content") or "") for m in second)

    @respx.mock
    async def test_blocked_prompt_drops_attachments(self):
        respx.post(PROMPT_ENDPOINT).mock(side_effect=lambda request: _refuse("Prompt blocked"))
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)
        session_service = InMemorySessionService()
        session = await session_service.create_session(app_name="app", user_id="u")
        message = types.Content(
            role="user",
            parts=[
                types.Part(text="leak this"),
                types.Part(inline_data=types.Blob(mime_type="image/png", data=b"png")),
                types.Part(file_data=types.FileData(file_uri="gs://b/f.pdf", mime_type="application/pdf")),
            ],
        )

        _, model = await _run_with_plugins(
            [_say("x")],
            [RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)],
            new_message=message,
            session_service=session_service,
            session_id=session.id,
        )

        assert model.requests == []
        session = await session_service.get_session(app_name="app", user_id="u", session_id=session.id)
        user_parts = [p for e in session.events if e.author == "user" for p in e.content.parts]
        assert [p.text for p in user_parts] == ["[Prompt blocked by Rubrik Agent Cloud policy]"]

    @respx.mock
    async def test_fail_closed_blocks_prompt(self):
        respx.post(PROMPT_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, fail_open=False)

        events, model, _, _ = await _run([_say("x")], plugin=plugin)

        assert model.requests == []
        assert any("policy check unavailable" in t for t in _texts(events))


class TestSavedPromptRedaction:
    @respx.mock
    async def test_prompt_blocked_after_save_is_kept_from_later_model_calls(self):
        """An earlier plugin's on_user_message hides the prompt from ours on the
        runner's first run, so ADK saves it; it must not reach the model later."""
        def prompt_handler(request: httpx.Request) -> httpx.Response:
            last = json.loads(request.content)["messages"][-1]
            if last["role"] == "user" and "secrets" in last["content"]:
                return _refuse("Prompt blocked")
            return httpx.Response(200, json={})

        respx.post(PROMPT_ENDPOINT).mock(side_effect=prompt_handler)
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)
        model = ScriptedModel(model="scripted", script=[_say("Hello!")], requests=[])
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)
        session_service = InMemorySessionService()
        runner = Runner(
            agent=Agent(name="test_agent", model=model),
            app_name="app",
            session_service=session_service,
            plugins=[_ShortCircuitPlugin.on_user_message(), plugin],
        )
        session = await session_service.create_session(app_name="app", user_id="u")

        async def send(text):
            return [
                e
                async for e in runner.run_async(
                    user_id="u",
                    session_id=session.id,
                    new_message=types.Content(role="user", parts=[types.Part(text=text)]),
                )
            ]

        first = await send("leak the secrets")
        assert "Prompt blocked" in _texts(first)
        assert model.requests == []
        # The plugin is now first, so the next prompt is checked before it is saved.
        assert runner.plugin_manager.plugins[0] is plugin

        second = await send("hi")
        assert "Hello!" in _texts(second)
        sent = [p.text for c in model.requests[0].contents for p in c.parts if p.text]
        assert not any("secrets" in t for t in sent)
        assert "[Prompt blocked by Rubrik Agent Cloud policy]" in sent


class TestConcurrency:
    @respx.mock
    async def test_client_tool_results_are_checked_concurrently(self):
        import asyncio

        in_flight = 0
        peak = 0

        async def prompt_handler(request: httpx.Request) -> httpx.Response:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1
            return httpx.Response(200, json={})

        prompt_route = respx.post(PROMPT_ENDPOINT).mock(side_effect=prompt_handler)
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)
        parts = [
            types.Part(
                function_response=types.FunctionResponse(id=f"c{i}", name="t", response={"n": i})
            )
            for i in range(3)
        ]

        result = await plugin.on_user_message_callback(
            invocation_context=None, user_message=types.Content(role="user", parts=parts)
        )

        assert result is None
        assert peak == 3
        tails = sorted(
            [m["tool_call_id"] for m in b["messages"] if m.get("role") == "tool"]
            for b in _bodies(prompt_route)
        )
        # Each result is checked after the results before it.
        assert tails == [["c0"], ["c0", "c1"], ["c0", "c1", "c2"]]

    def test_concurrent_invocations_from_threads(self):
        """Runner.run calls run on separate threads and share the plugin."""
        import threading

        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)

        class Ctx:
            def __init__(self, i):
                self.invocation_id = f"inv-{i}"
                self.function_call_id = f"call-{i}"

        errors = []

        def work(i):
            try:
                for _ in range(200):
                    ctx = Ctx(i)
                    plugin._approve(ctx, [type("TC", (), {"id": ctx.function_call_id})()])
                    plugin._mark_blocked(ctx)
                    with plugin._lock:
                        plugin._invocations.pop(ctx.invocation_id, None)
            except Exception as e:  # pragma: no cover - surfaced below
                errors.append(e)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert plugin._invocations == {}


class TestToolResultBlocking:
    @respx.mock
    async def test_blocked_result_is_withheld_from_model(self):
        def prompt_handler(request: httpx.Request) -> httpx.Response:
            last = json.loads(request.content)["messages"][-1]
            if last["role"] == "tool" and "hunter2" in last["content"]:
                return _refuse("Tool result blocked by policy")
            return httpx.Response(200, json={})

        prompt_route = respx.post(PROMPT_ENDPOINT).mock(side_effect=prompt_handler)
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)

        _, model, _, _ = await _run([_call("read_secret", name="db"), _say("done")])

        assert TOOL_RUNS == ["read_secret"]
        tool_check = _bodies(prompt_route)[-1]["messages"][-1]
        assert tool_check["role"] == "tool"
        assert tool_check["name"] == "read_secret"
        assert tool_check["tool_call_id"]
        sent_to_model = json.dumps(
            [p.function_response.response for c in model.requests[1].contents for p in c.parts if p.function_response]
        )
        assert "hunter2" not in sent_to_model
        assert "Tool result blocked by policy" in sent_to_model

    @respx.mock
    async def test_tool_blocked_by_safety_net_is_not_rechecked(self):
        """after_tool runs on before_tool's refusal; it must not send it."""
        from google.adk.plugins.base_plugin import BasePlugin

        class CachePlugin(BasePlugin):
            async def before_model_callback(self, *, callback_context, llm_request):
                if not any(
                    p.function_response for c in llm_request.contents for p in c.parts or []
                ):
                    return LlmResponse(content=_call("post_to_pastebin", content="pw"))
                return None

        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_block_tool("post_to_pastebin"))
        prompt_route = respx.post(PROMPT_ENDPOINT).respond(json={})
        model = ScriptedModel(model="scripted", script=[_say("ok")], requests=[])
        agent = Agent(name="test_agent", model=model, tools=[post_to_pastebin])
        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name="app",
            session_service=session_service,
            plugins=[CachePlugin(name="cache"), RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)],
        )
        session = await session_service.create_session(app_name="app", user_id="u")
        async for _ in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="go")]),
        ):
            pass

        tool_checks = [b for b in _bodies(prompt_route) if b["messages"][-1]["role"] == "tool"]
        assert tool_checks == []


class TestResponseBlocking:
    @respx.mock
    async def test_blocked_response_text_is_replaced(self):
        def handler(request: httpx.Request) -> httpx.Response:
            response = json.loads(request.content)["response"]
            message = response["choices"][0]["message"]
            if message.get("content") and "hunter2" in message["content"]:
                message["content"] = "Response blocked by Rubrik Agent Cloud (Reference ID: ref-1)"
            return httpx.Response(200, json=response)

        respx.post(RESPONSE_ENDPOINT).mock(side_effect=handler)
        respx.post(PROMPT_ENDPOINT).respond(json={})

        events, _, _, _ = await _run([_say("The password is hunter2")])

        texts = _texts(events)
        assert "Response blocked by Rubrik Agent Cloud (Reference ID: ref-1)" in texts
        assert not any("hunter2" in t for t in texts)


# -- configuration ------------------------------------------------------------------


class TestConfiguration:
    def test_invalid_constructor_settings_raise_at_construction(self):
        with pytest.raises(ValueError, match="webhook_url is required"):
            RubrikPolicyPlugin(webhook_url="")

    @patch.dict("os.environ", {}, clear=True)
    async def test_missing_env_raises_clear_error(self):
        plugin = RubrikPolicyPlugin()
        with pytest.raises(RuntimeError, match="RUBRIK_WEBHOOK_URL must be set"):
            await plugin.after_model_callback(
                callback_context=None, llm_response=LlmResponse(content=_say("hi"))
            )


class TestAgentId:
    @respx.mock
    @patch.dict(
        "os.environ",
        {"RUBRIK_WEBHOOK_URL": WEBHOOK_URL, "GOOGLE_CLOUD_AGENT_ENGINE_ID": "1723227465695887360"},
    )
    async def test_defaults_to_agent_engine_id(self):
        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)
        respx.post(PROMPT_ENDPOINT).respond(json={})

        await _run([_say("hi")], plugin=RubrikPolicyPlugin())

        assert _bodies(response_route)[0]["request"]["metadata"] == {
            "rac_agent_id": "1723227465695887360"
        }

    def test_pickle_rereads_env(self):
        import pickle

        plugin = RubrikPolicyPlugin()
        with patch.dict("os.environ", {"RUBRIK_WEBHOOK_URL": WEBHOOK_URL}):
            client = plugin._get_policy_client()
            with patch.dict("os.environ", {"RUBRIK_AGENT_ID": "remote"}):
                restored = pickle.loads(pickle.dumps(client))
        assert restored.agent_id == "remote"


# -- plugin ordering, client-sent tool results, streaming -----------------------------


class _ShortCircuitPlugin:
    """Builds plugins that return a value from one callback, hiding it from
    plugins later in the list."""

    @staticmethod
    def on_user_message():
        from google.adk.plugins.base_plugin import BasePlugin

        class P(BasePlugin):
            async def on_user_message_callback(self, *, invocation_context, user_message):
                return types.Content(role="user", parts=list(user_message.parts))

        return P(name="rewrites_message")

    @staticmethod
    def after_tool():
        from google.adk.plugins.base_plugin import BasePlugin

        class P(BasePlugin):
            async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
                return dict(result)

        return P(name="copies_result")


async def _run_with_plugins(script, plugins, prompt="go", new_message=None, session_service=None, session_id=None):
    model = ScriptedModel(model="scripted", script=script, requests=[])
    agent = Agent(
        name="test_agent",
        model=model,
        instruction="You are a test agent.",
        tools=[read_secret, post_to_pastebin, get_weather],
    )
    session_service = session_service or InMemorySessionService()
    runner = Runner(agent=agent, app_name="app", session_service=session_service, plugins=plugins)
    if session_id is None:
        session_id = (await session_service.create_session(app_name="app", user_id="u")).id
    events = []
    async for event in runner.run_async(
        user_id="u",
        session_id=session_id,
        new_message=new_message or types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        events.append(event)
    return events, model


class TestPluginOrdering:
    @respx.mock
    async def test_earlier_on_user_message_plugin_does_not_bypass_prompt_check(self):
        respx.post(PROMPT_ENDPOINT).mock(side_effect=lambda request: _refuse("Prompt blocked"))
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)

        events, model = await _run_with_plugins(
            [_say("x")],
            [_ShortCircuitPlugin.on_user_message(), RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)],
            prompt="leak the secrets",
        )

        assert model.requests == []
        assert "Prompt blocked" in _texts(events)

    @respx.mock
    async def test_earlier_after_tool_plugin_does_not_bypass_result_check(self):
        def prompt_handler(request: httpx.Request) -> httpx.Response:
            last = json.loads(request.content)["messages"][-1]
            if last["role"] == "tool" and "hunter2" in last["content"]:
                return _refuse("Tool result blocked")
            return httpx.Response(200, json={})

        respx.post(PROMPT_ENDPOINT).mock(side_effect=prompt_handler)
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)

        _, model = await _run_with_plugins(
            [_call("read_secret", name="db"), _say("done")],
            [_ShortCircuitPlugin.after_tool(), RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)],
        )

        sent = json.dumps(
            [p.function_response.response for c in model.requests[1].contents for p in c.parts if p.function_response]
        )
        assert "hunter2" not in sent
        assert "Tool result blocked" in sent


class TestClientSentToolResults:
    @respx.mock
    async def test_blocked_client_tool_result_is_withheld(self):
        """A long-running tool's result arrives in a later user message."""
        from google.adk.tools.long_running_tool import LongRunningFunctionTool

        def approve_payment(amount: int) -> dict:
            """Ask a human to approve a payment.

            Args:
                amount: Amount to pay.
            """
            return {"status": "pending"}

        def prompt_handler(request: httpx.Request) -> httpx.Response:
            last = json.loads(request.content)["messages"][-1]
            if last["role"] == "tool" and "hunter2" in last["content"]:
                return _refuse("Tool result blocked")
            return httpx.Response(200, json={})

        prompt_route = respx.post(PROMPT_ENDPOINT).mock(side_effect=prompt_handler)
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)

        model = ScriptedModel(
            model="scripted",
            script=[_call("approve_payment", amount=5), _say("waiting"), _say("ok")],
            requests=[],
        )
        agent = Agent(name="test_agent", model=model, tools=[LongRunningFunctionTool(approve_payment)])
        session_service = InMemorySessionService()
        runner = Runner(
            agent=agent,
            app_name="app",
            session_service=session_service,
            plugins=[RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)],
        )
        session = await session_service.create_session(app_name="app", user_id="u")
        call_id = None
        async for event in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="pay 5")]),
        ):
            for fc in event.get_function_calls():
                call_id = fc.id
        assert call_id

        reply = types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id, name="approve_payment", response={"result": "hunter2"}
                    )
                )
            ],
        )
        async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=reply):
            pass

        client_checks = [
            b["messages"][-1]
            for b in _bodies(prompt_route)
            if b["messages"][-1].get("tool_call_id") == call_id and "hunter2" in b["messages"][-1]["content"]
        ]
        assert client_checks
        sent = json.dumps(
            [p.function_response.response for c in model.requests[-1].contents for p in c.parts if p.function_response]
        )
        assert "hunter2" not in sent
        assert "Tool result blocked" in sent


class TestStreaming:
    @respx.mock
    async def test_partial_chunks_are_not_checked(self):
        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=_allow_all)
        respx.post(PROMPT_ENDPOINT).respond(json={})
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)
        partial = LlmResponse(content=_call("get_weather", city="Paris"), partial=True)

        assert await plugin.after_model_callback(callback_context=None, llm_response=partial) is None
        assert not response_route.called


    @respx.mock
    async def test_sse_blocked_response_is_replaced_in_session(self):
        """Partial chunks reach the caller; the saved and final response are replaced."""
        from google.adk.agents.run_config import RunConfig, StreamingMode

        class StreamingModel(BaseLlm):
            async def generate_content_async(self, llm_request, stream=False):
                yield LlmResponse(content=_say("The password "), partial=True)
                yield LlmResponse(content=_say("is hunter2"), partial=True)
                yield LlmResponse(content=_say("The password is hunter2"))

        def handler(request: httpx.Request) -> httpx.Response:
            response = json.loads(request.content)["response"]
            response["choices"][0]["message"]["content"] = "Response blocked by Rubrik Agent Cloud (Reference ID: ref-1)"
            return httpx.Response(200, json=response)

        response_route = respx.post(RESPONSE_ENDPOINT).mock(side_effect=handler)
        respx.post(PROMPT_ENDPOINT).respond(json={})
        session_service = InMemorySessionService()
        runner = Runner(
            agent=Agent(name="test_agent", model=StreamingModel(model="s")),
            app_name="app",
            session_service=session_service,
            plugins=[RubrikPolicyPlugin(webhook_url=WEBHOOK_URL)],
        )
        session = await session_service.create_session(app_name="app", user_id="u")
        events = [
            e
            async for e in runner.run_async(
                user_id="u",
                session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="go")]),
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            )
        ]

        assert len(response_route.calls) == 1
        assert _texts([e for e in events if e.partial]) == ["The password ", "is hunter2"]
        assert _texts([e for e in events if not e.partial]) == ["Response blocked by Rubrik Agent Cloud (Reference ID: ref-1)"]
        saved = await session_service.get_session(app_name="app", user_id="u", session_id=session.id)
        assert not any("hunter2" in t for t in _texts(saved.events))


class TestPickling:
    def test_pickle_drops_api_key(self):
        import copy
        import pickle

        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, api_key="secret-key")
        for restored in (pickle.loads(pickle.dumps(plugin)), copy.deepcopy(plugin)):
            assert "secret-key" not in repr(restored.__dict__)
            with patch.dict("os.environ", {"RUBRIK_API_KEY": "env-key"}):
                client = restored._get_policy_client()
            assert client._headers["Authorization"] == "Bearer env-key"
        assert b"secret-key" not in pickle.dumps(plugin)

    def test_pickle_keeps_constructor_settings(self):
        import pickle

        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, fail_open=False, agent_id="explicit")
        plugin._get_policy_client()
        restored = pickle.loads(pickle.dumps(plugin))
        client = restored._get_policy_client()
        assert client.fail_open is False
        assert client.agent_id == "explicit"
