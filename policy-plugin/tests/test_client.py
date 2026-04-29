"""Tests for RubrikPolicyClient — webhook interaction and format translation."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from rubrik_agent_cloud_policy.client import (
    PolicyResult,
    RubrikPolicyClient,
    ToolCall,
    _parse_webhook_response,
    _to_openai_payload,
)

WEBHOOK_URL = "https://webhook.example.com"
ENDPOINT = f"{WEBHOOK_URL}/v1/after_completion/openai/v1"


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
        assert payload["choices"][0]["message"]["tool_calls"] == []


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

            # Verify payload format
            body = json.loads(request.content)
            assert body["object"] == "chat.completion"
            assert len(body["choices"][0]["message"]["tool_calls"]) == 2
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

    def test_from_env_raises_without_url(self):
        with pytest.raises(RuntimeError, match="RUBRIK_WEBHOOK_URL must be set"):
            RubrikPolicyClient.from_env()
