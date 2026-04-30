# rubrik-policy-plugin (DEPRECATED)

**This package has been renamed to
[rubrik-agent-cloud-policy-plugin](https://pypi.org/project/rubrik-agent-cloud-policy-plugin/).**

## Migration

```bash
pip uninstall rubrik-policy-plugin
pip install rubrik-agent-cloud-policy-plugin
```

Update your imports:

```python
# Before
from rubrik_policy.adk import RubrikPolicyPlugin
from rubrik_policy import RubrikPolicyClient, ToolCall

# After
from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin
from rubrik_agent_cloud_policy import RubrikPolicyClient, ToolCall
```

This tombstone release re-exports the new package under the old import path as a temporary
compatibility shim. It will not receive further updates — migrate to
`rubrik-agent-cloud-policy-plugin` at your earliest convenience.
