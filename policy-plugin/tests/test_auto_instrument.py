"""Tests for auto-instrumentation of ADK Runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import rubrik_agent_cloud_policy.auto_instrument as auto_mod


@pytest.fixture(autouse=True)
def _reset_instrumentation():
    """Reset the global instrumentation state and restore Runner.__init__."""
    from google.adk.runners import Runner

    original_init = Runner.__init__
    auto_mod._INSTRUMENTED = False
    yield
    auto_mod._INSTRUMENTED = False
    Runner.__init__ = original_init


class TestAutoInstrument:
    def test_returns_true_on_first_call(self):
        result = auto_mod.auto_instrument()
        assert result is True
        assert auto_mod.is_instrumented()

    def test_returns_false_on_second_call(self):
        auto_mod.auto_instrument()
        result = auto_mod.auto_instrument()
        assert result is False

    def test_patches_runner_init(self):
        from google.adk.runners import Runner

        original_init = Runner.__init__
        auto_mod.auto_instrument()
        assert Runner.__init__ is not original_init

    @patch.dict("os.environ", {"RUBRIK_WEBHOOK_URL": "https://webhook.test.com"})
    def test_injects_plugin_into_runner(self):
        """After patching, a new Runner should have the Rubrik plugin."""
        auto_mod.auto_instrument()

        from google.adk.runners import Runner

        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        runner = Runner(
            agent=mock_agent,
            app_name="test_app",
            session_service=MagicMock(),
        )

        from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

        plugin_names = [
            p.PLUGIN_NAME
            for p in runner.plugin_manager.plugins
            if hasattr(p, "PLUGIN_NAME")
        ]
        assert "rubrik_agent_cloud_policy" in plugin_names

    def test_no_injection_when_url_not_set(self):
        """Plugin is not injected when RUBRIK_WEBHOOK_URL is not set."""
        auto_mod.auto_instrument()

        from google.adk.runners import Runner

        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        runner = Runner(
            agent=mock_agent,
            app_name="test_app",
            session_service=MagicMock(),
        )

        from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

        rubrik_plugins = [
            p
            for p in runner.plugin_manager.plugins
            if isinstance(p, RubrikPolicyPlugin)
        ]
        assert len(rubrik_plugins) == 0

    def test_is_instrumented_false_initially(self):
        assert not auto_mod.is_instrumented()
