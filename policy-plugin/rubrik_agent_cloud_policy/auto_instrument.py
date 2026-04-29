"""Auto-instrumentation for ADK agents.

Call :func:`auto_instrument` once at process startup to inject
:class:`RubrikPolicyPlugin` into every :class:`google.adk.runners.Runner`
created afterwards — no per-agent code changes required.

Usage — one line in entrypoint::

    from rubrik_agent_cloud_policy.auto_instrument import auto_instrument
    auto_instrument()

Or zero-code via ``sitecustomize.py`` (runs before any user code)::

    # In site-packages/sitecustomize.py
    import os
    if os.getenv("RUBRIK_AUTO_INSTRUMENT", "").lower() in ("true", "1"):
        from rubrik_agent_cloud_policy.auto_instrument import auto_instrument
        auto_instrument()

Note: ``PYTHONSTARTUP`` does NOT work — it only runs for interactive sessions.
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

_INSTRUMENTED = False


def auto_instrument() -> bool:
    """Monkey-patch ``Runner.__init__`` to inject RubrikPolicyPlugin.

    Safe to call multiple times — only patches once.  Returns ``True`` if
    patching succeeded, ``False`` if it was already applied or ADK is not
    importable.
    """
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return False

    try:
        from google.adk.runners import Runner
    except ImportError:
        log.warning(
            "google-adk not installed — auto_instrument() skipped. "
            "Install with: pip install rubrik-agent-cloud-policy-plugin[adk]"
        )
        return False

    from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

    original_init = Runner.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Inject after the original __init__ so self.plugin_manager exists
        try:
            if not os.getenv("RUBRIK_WEBHOOK_URL"):
                return
            plugin = RubrikPolicyPlugin()
            pm = getattr(self, "plugin_manager", None)
            if pm and hasattr(pm, "register_plugin"):
                pm.register_plugin(plugin)
                log.info("Auto-injected RubrikPolicyPlugin into Runner")
            else:
                log.warning(
                    "Runner has no plugin_manager — "
                    "RubrikPolicyPlugin not injected"
                )
        except Exception:
            log.exception("Failed to auto-inject RubrikPolicyPlugin")

    Runner.__init__ = _patched_init  # type: ignore[method-assign]
    _INSTRUMENTED = True
    log.info("ADK Runner patched for Rubrik Agent Cloud policy enforcement")
    return True


def is_instrumented() -> bool:
    """Return whether auto-instrumentation has been applied."""
    return _INSTRUMENTED
