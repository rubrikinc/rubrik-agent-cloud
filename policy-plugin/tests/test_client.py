"""Tests for RubrikPolicyClient — webhook interaction and format translation."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from rubrik_agent_cloud_policy.client import (
    PolicyContext,
    PolicyResult,
    RubrikPolicyClient,
    ToolCall,
    _parse_prompt_response,
    _parse_webhook_response,
    _to_openai_payload,
    _to_openai_request,
)

WEBHOOK_URL = "https://webhook.example.com"
ENDPOINT = f"{WEBHOOK_URL}/v1/after_completion/openai/v1"
PROMPT_ENDPOINT = f"{WEBHOOK_URL}/v1/before_prompt/openai/v1"


# -- ToolCall defaults --------------------------------------------------------


class TestToolCall:
    def test_auto_generates_id(self):
        tc = ToolCall(name="foo", arguments="{}")
        assert tc.id.startswith("call_")
        assert len(tc.id) > 10

    def test_explicit_id(self):
        tc = ToolCall(name="foo", arguments="{}", id="custom_123")
        assert tc.id == "custom_123"

    def test_two_calls_get_different_ids(self):
        a = ToolCall(name="foo", arguments="{}")
        b = ToolCall(name="foo", arguments="{}")
        assert a.id != b.id


# -- OpenAI payload construction ----------------------------------------------


class TestToOpenaiPayload:
    def test_single_tool(self):
        tc = ToolCall(name="delete_file", arguments='{"path": "/etc/passwd"}', id="c1")
        payload = _to_openai_payload([tc])

        assert payload["object"] == "chat.completion"
        choices = payload["choices"]
        assert len(choices) == 1
        tool_calls = choices[0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0] == {
            "id": "c1",
            "type": "function",
            "function": {"name": "delete_file", "arguments": '{"path": "/etc/passwd"}'},
        }

    def test_multiple_tools(self):
        tcs = [
            ToolCall(name="read_file", arguments="{}", id="c1"),
            ToolCall(name="delete_file", arguments="{}", id="c2"),
        ]
        payload = _to_openai_payload(tcs)
        tool_calls = payload["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert [tc["function"]["name"] for tc in tool_calls] == [
            "read_file",
            "delete_file",
        ]

    def test_empty_list(self):
        payload = _to_openai_payload([])
        assert "tool_calls" not in payload["choices"][0]["message"]
        assert payload["choices"][0]["finish_reason"] == "stop"

    def test_text_and_response_id(self):
        payload = _to_openai_payload([], text="Hello", response_id="inv-1")
        assert payload["id"] == "inv-1"
        assert payload["choices"][0]["message"]["content"] == "Hello"


class TestToOpenaiRequest:
    def test_includes_messages_tools_and_agent_id(self):
        context = PolicyContext(
            messages=({"role": "user", "content": "hi"},),
            tools=({"type": "function", "function": {"name": "get_weather"}},),
            model="gemini-2.5-flash",
            agent_id="123",
        )
        request = _to_openai_request(context)
        assert request == {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "metadata": {"rac_agent_id": "123"},
        }

    def test_default_agent_id(self):
        request = _to_openai_request(PolicyContext(), default_agent_id="456")
        assert request["metadata"] == {"rac_agent_id": "456"}

    def test_no_agent_id_omits_metadata(self):
        request = _to_openai_request(PolicyContext())
        assert "metadata" not in request
        assert "tools" not in request


# -- Webhook response parsing -------------------------------------------------


class TestParseWebhookResponse:
    def test_all_allowed(self):
        """Webhook returns all tools — nothing blocked."""
        originals = [
            ToolCall(name="read_file", arguments="{}", id="c1"),
            ToolCall(name="write_file", arguments="{}", id="c2"),
        ]
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "read_file"}},
                            {"id": "c2", "function": {"name": "write_file"}},
                        ],
                        "content": None,
                    }
                }
            ]
        }
        result = _parse_webhook_response(response, originals)
        assert len(result.allowed) == 2
        assert len(result.blocked) == 0
        assert result.explanation == ""

    def test_some_blocked(self):
        """Webhook removes blocked tools from tool_calls."""
        originals = [
            ToolCall(name="read_file", arguments="{}", id="c1"),
            ToolCall(name="delete_file", arguments="{}", id="c2"),
            ToolCall(name="list_dir", arguments="{}", id="c3"),
        ]
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "read_file"}},
                            {"id": "c3", "function": {"name": "list_dir"}},
                        ],
                        "content": "The following tool calls were blocked by policy:\n- delete_file: blocked by policy",
                    }
                }
            ]
        }
        result = _parse_webhook_response(response, originals)
        assert [tc.name for tc in result.allowed] == ["read_file", "list_dir"]
        assert [tc.name for tc in result.blocked] == ["delete_file"]
        assert "delete_file" in result.explanation

    def test_all_blocked(self):
        """Webhook returns no tool_calls — all blocked."""
        originals = [
            ToolCall(name="delete_file", arguments="{}", id="c1"),
            ToolCall(name="drop_table", arguments="{}", id="c2"),
        ]
        response = {
            "choices": [
                {
                    "message": {
                        "content": "All tool calls were blocked by policy",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        result = _parse_webhook_response(response, originals)
        assert len(result.allowed) == 0
        assert len(result.blocked) == 2

    def test_empty_choices_raises(self):
        with pytest.raises(ValueError, match="empty choices"):
            _parse_webhook_response({"choices": []}, [])

    def test_duplicate_returned_id_allows_only_one(self):
        """A returned id counts once, even if two originals share it."""
        originals = [
            ToolCall(name="delete_file", arguments="{}", id="dup"),
            ToolCall(name="delete_file", arguments="{}", id="dup"),
        ]
        response = {"choices": [{"message": {"tool_calls": [{"id": "dup"}], "content": "blocked"}}]}
        result = _parse_webhook_response(response, originals)
        assert len(result.allowed) == 1
        assert len(result.blocked) == 1

    def test_explanation_appended_to_text(self):
        """A tool block appends the explanation after a blank line."""
        originals = [ToolCall(name="delete_file", arguments="{}", id="c1")]
        response = {"choices": [{"message": {"content": "Sure.\n\nThe following tool calls were blocked"}}]}
        result = _parse_webhook_response(response, originals, sent_text="Sure.")
        assert not result.response_blocked
        assert [tc.name for tc in result.blocked] == ["delete_file"]
        assert result.explanation == "The following tool calls were blocked"

    def test_response_blocked_when_text_replaced(self):
        originals = [ToolCall(name="send_email", arguments="{}", id="c1")]
        response = {"choices": [{"message": {"content": "Response blocked by policy"}}]}
        result = _parse_webhook_response(response, originals, sent_text="Here is the password")
        assert result.response_blocked
        assert result.allowed == ()
        assert [tc.name for tc in result.blocked] == ["send_email"]
        assert result.explanation == "Response blocked by policy"

    def test_text_unchanged_is_not_blocked(self):
        response = {"choices": [{"message": {"content": "Hello"}}]}
        result = _parse_webhook_response(response, [], sent_text="Hello")
        assert not result.response_blocked
        assert result.explanation == ""


class TestParsePromptResponse:
    def test_passthrough(self):
        assert _parse_prompt_response({}) is None

    def test_refusal(self):
        response = {"choices": [{"message": {"content": "Prompt blocked"}}]}
        assert _parse_prompt_response(response) == "Prompt blocked"

    def test_refusal_without_content(self):
        assert _parse_prompt_response({"choices": [{"message": {}}]})


# -- Client integration (mocked HTTP) -----------------------------------------


class TestClientEvaluateToolCalls:
    @respx.mock
    @pytest.mark.asyncio
    async def test_posts_to_webhook_and_parses(self):
        """End-to-end: client builds payload, posts, parses response."""
        originals = [
            ToolCall(name="safe_tool", arguments="{}", id="c1"),
            ToolCall(name="dangerous_tool", arguments="{}", id="c2"),
        ]

        # Webhook allows only safe_tool
        webhook_response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "safe_tool"}},
                        ],
                        "content": "Blocked: dangerous_tool",
                    }
                }
            ]
        }
        route = respx.post(ENDPOINT).respond(json=webhook_response)

        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL, api_key="test-key")
        try:
            result = await client.evaluate_tool_calls(originals)

            assert route.called
            assert [tc.name for tc in result.allowed] == ["safe_tool"]
            assert [tc.name for tc in result.blocked] == ["dangerous_tool"]

            # Verify auth header was sent
            request = route.calls[0].request
            assert request.headers["authorization"] == "Bearer test-key"
            assert request.headers["content-type"] == "application/json"

            # Verify envelope format
            body = json.loads(request.content)
            assert set(body) == {"request", "response"}
            assert body["response"]["object"] == "chat.completion"
            assert len(body["response"]["choices"][0]["message"]["tool_calls"]) == 2
        finally:
            await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_sends_context_and_agent_id(self):
        route = respx.post(ENDPOINT).respond(
            json={"choices": [{"message": {"tool_calls": [{"id": "c1"}]}}]}
        )
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL, agent_id="engine-1")
        context = PolicyContext(
            messages=({"role": "user", "content": "read the secret"},),
            tools=({"type": "function", "function": {"name": "read_secret"}},),
            correlation_id="inv-1",
        )
        try:
            await client.evaluate_tool_calls(
                [ToolCall(name="read_secret", arguments="{}", id="c1")], context
            )
            body = json.loads(route.calls[0].request.content)
            assert body["request"]["messages"] == [{"role": "user", "content": "read the secret"}]
            assert body["request"]["tools"][0]["function"]["name"] == "read_secret"
            assert body["request"]["metadata"] == {"rac_agent_id": "engine-1"}
            assert body["response"]["id"] == "inv-1"
        finally:
            await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_non_object_response_raises(self):
        respx.post(ENDPOINT).respond(json=[])
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL)
        try:
            with pytest.raises(ValueError):
                await client.evaluate_tool_calls([ToolCall(name="foo", arguments="{}")])
        finally:
            await client.close()

    @respx.mock
    def test_new_event_loop_per_call(self):
        """ADK's sync Runner.run starts a new event loop per call."""
        import asyncio

        respx.post(ENDPOINT).respond(
            json={"choices": [{"message": {"tool_calls": [{"id": "c1"}]}}]}
        )
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL)
        for _ in range(2):
            result = asyncio.run(
                client.evaluate_tool_calls([ToolCall(name="foo", arguments="{}", id="c1")])
            )
            assert result.blocked == ()


class TestClientEvaluatePrompt:
    @respx.mock
    @pytest.mark.asyncio
    async def test_passthrough(self):
        route = respx.post(PROMPT_ENDPOINT).respond(json={})
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL, agent_id="engine-1")
        context = PolicyContext(
            messages=({"role": "user", "content": "hello"},), correlation_id="inv-1"
        )
        try:
            assert await client.evaluate_prompt(context) is None
            body = json.loads(route.calls[0].request.content)
            assert body["messages"] == [{"role": "user", "content": "hello"}]
            assert body["metadata"] == {"rac_agent_id": "engine-1"}
            assert body["correlation_key"] == "inv-1"
        finally:
            await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_refusal(self):
        respx.post(PROMPT_ENDPOINT).respond(
            json={"choices": [{"message": {"content": "Prompt blocked"}, "finish_reason": "content_filter"}]}
        )
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL)
        try:
            refusal = await client.evaluate_prompt(
                PolicyContext(messages=({"role": "user", "content": "bad"},))
            )
            assert refusal == "Prompt blocked"
        finally:
            await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_webhook_error_raises(self):
        """Client raises on HTTP errors — caller decides fail-open/closed."""
        respx.post(ENDPOINT).respond(status_code=500)

        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await client.evaluate_tool_calls(
                    [ToolCall(name="foo", arguments="{}")]
                )
        finally:
            await client.close()

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_auth_header_when_no_key(self):
        """No Authorization header sent when api_key is not set."""
        route = respx.post(ENDPOINT).respond(
            json={"choices": [{"message": {"tool_calls": [{"id": "c1"}]}}]}
        )
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL, api_key=None)
        try:
            await client.evaluate_tool_calls(
                [ToolCall(name="foo", arguments="{}", id="c1")]
            )
            assert "authorization" not in route.calls[0].request.headers
        finally:
            await client.close()


# -- Client configuration -----------------------------------------------------


class TestClientConfiguration:
    def test_raises_on_empty_url(self):
        with pytest.raises(ValueError, match="webhook_url is required"):
            RubrikPolicyClient(webhook_url="")

    def test_enabled_with_url(self):
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL)
        assert client._tool_blocking_endpoint == ENDPOINT

    def test_strips_trailing_slash_and_v1(self):
        client = RubrikPolicyClient(webhook_url=f"{WEBHOOK_URL}/v1/")
        assert client._tool_blocking_endpoint == ENDPOINT

    def test_fail_open_default(self):
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL)
        assert client.fail_open is True

    def test_fail_closed_explicit(self):
        client = RubrikPolicyClient(webhook_url=WEBHOOK_URL, fail_open=False)
        assert client.fail_open is False

    @patch.dict("os.environ", {"RUBRIK_WEBHOOK_URL": WEBHOOK_URL})
    def test_from_env(self):
        client = RubrikPolicyClient.from_env()
        assert client._tool_blocking_endpoint == ENDPOINT

    @patch.dict(
        "os.environ",
        {"RUBRIK_WEBHOOK_URL": WEBHOOK_URL, "GOOGLE_CLOUD_AGENT_ENGINE_ID": "1723"},
    )
    def test_from_env_agent_id_defaults_to_agent_engine_id(self):
        assert RubrikPolicyClient.from_env().agent_id == "1723"

    @patch.dict(
        "os.environ",
        {
            "RUBRIK_WEBHOOK_URL": WEBHOOK_URL,
            "GOOGLE_CLOUD_AGENT_ENGINE_ID": "1723",
            "RUBRIK_AGENT_ID": "custom",
        },
    )
    def test_from_env_rubrik_agent_id_wins(self):
        assert RubrikPolicyClient.from_env().agent_id == "custom"

    @patch.dict("os.environ", {"RUBRIK_WEBHOOK_URL": WEBHOOK_URL}, clear=True)
    def test_from_env_no_agent_id(self):
        assert RubrikPolicyClient.from_env().agent_id == ""

    def test_from_env_raises_without_url(self):
        with pytest.raises(RuntimeError, match="RUBRIK_WEBHOOK_URL must be set"):
            RubrikPolicyClient.from_env()
