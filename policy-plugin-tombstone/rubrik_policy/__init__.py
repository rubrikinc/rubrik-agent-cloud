"""DEPRECATED: rubrik-policy-plugin has been renamed to rubrik-agent-cloud-policy-plugin.

Migrate your imports:
    from rubrik_policy.adk import RubrikPolicyPlugin
  becomes:
    from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

Install the new package:
    pip install rubrik-agent-cloud-policy-plugin
"""

import warnings

warnings.warn(
    "rubrik-policy-plugin is deprecated and will not receive updates. "
    "Install rubrik-agent-cloud-policy-plugin instead: "
    "pip install rubrik-agent-cloud-policy-plugin  — "
    "then update imports from 'rubrik_policy' to 'rubrik_agent_cloud_policy'.",
    DeprecationWarning,
    stacklevel=2,
)

from rubrik_agent_cloud_policy.client import PolicyResult, RubrikPolicyClient, ToolCall

__all__ = ["RubrikPolicyClient", "ToolCall", "PolicyResult"]
