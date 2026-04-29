"""Tests for the ADK adapter — callback behavior with mocked policy client."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from rubrik_agent_cloud_policy.client import PolicyResult, RubrikPolicyClient, ToolCall

# Import ADK types — tests require google-adk
from google.genai import types

from rubrik_agent_cloud_policy.adk import (
    RubrikPolicyPlugin,
    _apply_policy_result,
    _gemini_parts_to_tool_calls,
)


# -- Helpers ------------------------------------------------------------------


def _make_llm_response(
    parts: list[types.Part], role: str = "model"
) -> Any:
    """Build a minimal LlmResponse-like object with .content."""

    @dataclass
    class FakeLlmResponse:
        content: Optional[types.Content] = None

    return FakeLlmResponse(content=types.Content(role=role, parts=parts))


def _fc_part(name: str, args: Optional[dict] = None) -> types.Part:
    """Build a Gemini Part with a function_call."""
    return types.Part(
        function_call=types.FunctionCall(name=name, args=args or {})
    )


def _text_part(text: str) -> types.Part:
    return types.Part(text=text)


def _make_plugin() -> RubrikPolicyPlugin:
    """Create a plugin with explicit webhook_url (bypasses from_env)."""
    return RubrikPolicyPlugin(webhook_url="https://example.com")


# -- Gemini -> ToolCall conversion ---------------------------------------------


class TestGeminiPartsToToolCalls:
    def test_converts_function_calls(self):
        parts = [
            (0, _fc_part("delete_file", {"path": "/tmp"})),
            (2, _fc_part("read_file", {"path": "/etc"})),
        ]
        tool_calls, index_by_id = _gemini_parts_to_tool_calls(parts)
        assert len(tool_calls) == 2
        assert tool_calls[0].name == "delete_file"
        assert json.loads(tool_calls[0].arguments) == {"path": "/tmp"}
        assert tool_calls[1].name == "read_file"
        # index_by_id maps tool call IDs to original part indices
        assert index_by_id[tool_calls[0].id] == 0
        assert index_by_id[tool_calls[1].id] == 2

    def test_empty_args(self):
        parts = [(0, _fc_part("no_args"))]
        tool_calls, index_by_id = _gemini_parts_to_tool_calls(parts)
        assert json.loads(tool_calls[0].arguments) == {}
        assert index_by_id[tool_calls[0].id] == 0

    def test_duplicate_names_get_unique_ids(self):
        """Two calls to the same function get distinct IDs and indices."""
        parts = [
            (0, _fc_part("delete_file", {"path": "/a"})),
            (1, _fc_part("delete_file", {"path": "/b"})),
        ]
        tool_calls, index_by_id = _gemini_parts_to_tool_calls(parts)
        assert tool_calls[0].id != tool_calls[1].id
        assert index_by_id[tool_calls[0].id] == 0
        assert index_by_id[tool_calls[1].id] == 1


# -- after_model_callback tests -----------------------------------------------


class TestAfterModelCallback:
    @pytest.mark.asyncio
    async def test_returns_none_for_no_function_calls(self):
        plugin = _make_plugin()
        resp = _make_llm_response([_text_part("Hello")])
        result = await plugin.after_model_callback(
            callback_context=None, llm_response=resp
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_all_allowed(self):
        plugin = _make_plugin()

        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com")
        plugin._policy_client.evaluate_tool_calls = AsyncMock(
            return_value=PolicyResult(
                allowed=(ToolCall(name="safe", arguments="{}"),),
                blocked=(),
                explanation="",
            )
        )
        resp = _make_llm_response([_fc_part("safe")])
        result = await plugin.after_model_callback(
            callback_context=None, llm_response=resp
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_strips_blocked_tools(self):
        plugin = _make_plugin()

        async def mock_evaluate(tool_calls):
            allowed = tuple(tc for tc in tool_calls if tc.name != "delete_file")
            blocked = tuple(tc for tc in tool_calls if tc.name == "delete_file")
            return PolicyResult(
                allowed=allowed,
                blocked=blocked,
                explanation="delete_file blocked by policy",
            )

        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com")
        plugin._policy_client.evaluate_tool_calls = mock_evaluate

        resp = _make_llm_response([
            _fc_part("read_file"),
            _fc_part("delete_file"),
            _text_part("Let me help"),
        ])
        result = await plugin.after_model_callback(
            callback_context=None, llm_response=resp
        )

        assert result is not None
        parts = result.content.parts
        fc_names = [p.function_call.name for p in parts if p.function_call]
        text_parts = [p.text for p in parts if p.text]
        assert fc_names == ["read_file"]
        assert "Let me help" in text_parts
        assert "delete_file blocked by policy" in text_parts

    @pytest.mark.asyncio
    async def test_duplicate_names_only_blocks_matched_id(self):
        """Two calls to delete_file — only one blocked."""
        plugin = _make_plugin()

        async def mock_evaluate(tool_calls):
            allowed = tuple(tc for tc in tool_calls if '"path": "/a"' in tc.arguments)
            blocked = tuple(tc for tc in tool_calls if '"path": "/b"' in tc.arguments)
            return PolicyResult(allowed=allowed, blocked=blocked, explanation="blocked /b")

        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com")
        plugin._policy_client.evaluate_tool_calls = mock_evaluate

        resp = _make_llm_response([
            _fc_part("delete_file", {"path": "/a"}),
            _fc_part("delete_file", {"path": "/b"}),
        ])
        result = await plugin.after_model_callback(
            callback_context=None, llm_response=resp
        )

        assert result is not None
        parts = result.content.parts
        fc_parts = [p for p in parts if p.function_call]
        assert len(fc_parts) == 1
        assert dict(fc_parts[0].function_call.args) == {"path": "/a"}

    @pytest.mark.asyncio
    async def test_fail_open_on_error(self):
        plugin = _make_plugin()
        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com", fail_open=True)
        plugin._policy_client.evaluate_tool_calls = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        resp = _make_llm_response([_fc_part("dangerous")])
        result = await plugin.after_model_callback(
            callback_context=None, llm_response=resp
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_fail_closed_strips_all_on_error(self):
        plugin = _make_plugin()
        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com", fail_open=False)
        plugin._policy_client.evaluate_tool_calls = AsyncMock(
            side_effect=httpx.ReadTimeout("timeout")
        )

        resp = _make_llm_response([
            _fc_part("tool_a"),
            _fc_part("tool_b"),
            _text_part("thinking"),
        ])
        result = await plugin.after_model_callback(
            callback_context=None, llm_response=resp
        )

        assert result is not None
        fc_names = [p.function_call.name for p in result.content.parts if p.function_call]
        assert fc_names == []
        texts = [p.text for p in result.content.parts if p.text]
        assert "thinking" in texts
        assert any("policy check unavailable" in t for t in texts)

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_response(self):
        plugin = _make_plugin()
        assert await plugin.after_model_callback(
            callback_context=None, llm_response=None
        ) is None

    @pytest.mark.asyncio
    async def test_returns_none_for_null_parts(self):
        plugin = _make_plugin()
        resp = _make_llm_response([])
        resp.content = types.Content(role="model", parts=None)
        assert await plugin.after_model_callback(
            callback_context=None, llm_response=resp
        ) is None


# -- before_tool_callback tests -----------------------------------------------


class TestBeforeToolCallback:
    @dataclass
    class FakeTool:
        name: str

    @pytest.mark.asyncio
    async def test_returns_none_when_allowed(self):
        plugin = _make_plugin()
        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com")
        plugin._policy_client.evaluate_tool_calls = AsyncMock(
            return_value=PolicyResult(
                allowed=(ToolCall(name="safe", arguments="{}"),),
                blocked=(),
                explanation="",
            )
        )
        result = await plugin.before_tool_callback(
            tool=self.FakeTool("safe"), tool_args={}, tool_context=None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_error_when_blocked(self):
        plugin = _make_plugin()
        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com")
        plugin._policy_client.evaluate_tool_calls = AsyncMock(
            return_value=PolicyResult(
                allowed=(),
                blocked=(ToolCall(name="bad_tool", arguments="{}"),),
                explanation="bad_tool blocked by Rubrik Agent Cloud policy",
            )
        )
        result = await plugin.before_tool_callback(
            tool=self.FakeTool("bad_tool"), tool_args={"x": 1}, tool_context=None
        )
        assert result is not None
        assert "bad_tool blocked by Rubrik Agent Cloud policy" in result["error"]

    @pytest.mark.asyncio
    async def test_fail_open_on_error(self):
        plugin = _make_plugin()
        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com", fail_open=True)
        plugin._policy_client.evaluate_tool_calls = AsyncMock(
            side_effect=httpx.ConnectError("boom")
        )
        result = await plugin.before_tool_callback(
            tool=self.FakeTool("foo"), tool_args={}, tool_context=None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_fail_closed_on_error(self):
        plugin = _make_plugin()
        plugin._policy_client = RubrikPolicyClient(webhook_url="https://example.com", fail_open=False)
        plugin._policy_client.evaluate_tool_calls = AsyncMock(
            side_effect=httpx.ConnectError("boom")
        )
        result = await plugin.before_tool_callback(
            tool=self.FakeTool("foo"), tool_args={}, tool_context=None
        )
        assert result is not None
        assert "policy check unavailable" in result["error"]
