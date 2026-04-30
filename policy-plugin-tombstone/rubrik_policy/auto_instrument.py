"""DEPRECATED: use rubrik_agent_cloud_policy.auto_instrument instead."""

from rubrik_agent_cloud_policy.auto_instrument import *  # noqa: F401,F403
from rubrik_agent_cloud_policy.auto_instrument import auto_instrument, is_instrumented  # noqa: F811

__all__ = ["auto_instrument", "is_instrumented"]
