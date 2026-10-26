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
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
        respx.post(PROMPT_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))

        await _run([_call("post_to_pastebin", content="x"), _say("done")])

        assert TOOL_RUNS == ["post_to_pastebin"]

    @respx.mock
    async def test_fail_closed_on_error(self):
        respx.post(RESPONSE_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
        respx.post(PROMPT_ENDPOINT).respond(json={})
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, fail_open=False)

        events, _, _, _ = await _run([_call("post_to_pastebin", content="x"), _say("done")], plugin=plugin)

        assert TOOL_RUNS == []
        assert any("policy check unavailable" in t for t in _texts(events))


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
    async def test_fail_closed_blocks_prompt(self):
        respx.post(PROMPT_ENDPOINT).mock(side_effect=httpx.ConnectError("refused"))
        plugin = RubrikPolicyPlugin(webhook_url=WEBHOOK_URL, fail_open=False)

        events, model, _, _ = await _run([_say("x")], plugin=plugin)

        assert model.requests == []
        assert any("policy check unavailable" in t for t in _texts(events))


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
                message["content"] = "Response blocked by policy"
            return httpx.Response(200, json=response)

        respx.post(RESPONSE_ENDPOINT).mock(side_effect=handler)
        respx.post(PROMPT_ENDPOINT).respond(json={})

        events, _, _, _ = await _run([_say("The password is hunter2")])

        texts = _texts(events)
        assert "Response blocked by policy" in texts
        assert not any("hunter2" in t for t in texts)


# -- configuration ------------------------------------------------------------------


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
