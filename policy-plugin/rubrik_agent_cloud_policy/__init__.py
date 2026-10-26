"""Rubrik Agent Cloud policy enforcement for AI agent frameworks.

Core client:
    from rubrik_agent_cloud_policy import RubrikPolicyClient, PolicyContext, ToolCall, PolicyResult

ADK adapter (requires `pip install rubrik-agent-cloud-policy-plugin[adk]`):
    from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

Auto-instrumentation:
    from rubrik_agent_cloud_policy.auto_instrument import auto_instrument
"""

from rubrik_agent_cloud_policy.client import (
    PolicyContext,
    PolicyResult,
    RubrikPolicyClient,
    ToolCall,
)

__all__ = ["RubrikPolicyClient", "PolicyContext", "ToolCall", "PolicyResult"]
