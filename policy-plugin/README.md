# rubrik-agent-cloud-policy-plugin

Lightweight Python package for enforcing Rubrik Agent Cloud tool-blocking policies in AI agent frameworks. Framework-agnostic core with a Google ADK adapter.

## Install

```bash
pip install rubrik-agent-cloud-policy-plugin[adk]
```

Only dependencies: `httpx` + `google-adk` (which you already have).

## Quick Start

### Option A: Explicit plugin (one line)

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

runner = Runner(
    agent=my_agent,
    app_name="my_app",
    session_service=session_service,
    plugins=[RubrikPolicyPlugin()],
)
```

### Option B: Auto-instrumentation (zero code changes)

```python
from rubrik_agent_cloud_policy.auto_instrument import auto_instrument
auto_instrument()  # patches all future Runner instances
```

For fully zero-code instrumentation, use `sitecustomize.py` — it runs before any user code in all Python invocations. Add this to your `site-packages/sitecustomize.py`:

```python
import os
if os.getenv("RUBRIK_AUTO_INSTRUMENT", "").lower() in ("true", "1"):
    from rubrik_agent_cloud_policy.auto_instrument import auto_instrument
    auto_instrument()
```

Then set the env var:
```bash
export RUBRIK_AUTO_INSTRUMENT=true
```

> **Note**: `PYTHONSTARTUP` does not work — it only runs for interactive Python sessions, not for `python3 script.py` or Agent Engine containers. Use `sitecustomize.py` instead.

### Option C: ADK_EXTRA_PLUGINS (zero code changes)

```bash
export ADK_EXTRA_PLUGINS='["rubrik_agent_cloud_policy.adk.RubrikPolicyPlugin"]'
```

All options read configuration from environment variables:

```bash
export RUBRIK_WEBHOOK_URL="https://webhooks.example.com/{tenant_short_code}/{webhook_uuid}/"
export RUBRIK_API_KEY="your-api-key"
export RUBRIK_POLICY_FAIL_OPEN="true"                  # default: true (fail-open)
```

## Webhook URL Format

The webhook URL must include the tenant short code and webhook UUID in the path:

```
https://webhooks.{domain}/{tenant_short_code}/{webhook_uuid}/
```

This is the per-tenant webhook URL returned when a webhook is created via the Gateway API. The Istio AuthorizationPolicy only allows POST to specific endpoint paths — GET requests (including health checks) return 403.

## How It Works

1. LLM responds with tool calls (e.g. `get_weather`, `delete_file`)
2. `after_model_callback` intercepts the response
3. Plugin translates Gemini function calls to OpenAI format
4. POSTs to the Rubrik webhook (`/v1/after_completion/openai/v1`)
5. Webhook evaluates against configured policies, returns allowed tools
6. Blocked tools are stripped from the response; explanation text is injected
7. Agent receives only the allowed tool calls

If the webhook is unreachable, the plugin **fails open** by default (all tools allowed). Set `RUBRIK_POLICY_FAIL_OPEN=false` to fail closed.

`before_tool_callback` acts as a safety net for tools that bypass `after_model_callback` (e.g. when another plugin short-circuits via `before_model_callback`).

## Deploy to Vertex AI Agent Engine

### 1. Build the wheel

```bash
cd policy-plugin
pip install build
python -m build
# produces dist/rubrik_agent_cloud_policy_plugin-0.1.1-py3-none-any.whl
```

### 2. Upload to GCS (Agent Engine needs HTTP access during build)

```bash
gsutil cp dist/rubrik_agent_cloud_policy_plugin-0.1.1-py3-none-any.whl gs://{bucket}/packages/
gsutil acl ch -u AllUsers:R gs://{bucket}/packages/rubrik_agent_cloud_policy_plugin-0.1.1-py3-none-any.whl
```

### 3. Deploy

```python
import vertexai
from vertexai import agent_engines
from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

vertexai.init(project="my-project", location="us-central1",
              staging_bucket="gs://my-bucket")

adk_app = agent_engines.AdkApp(
    agent=my_agent,
    plugins=[RubrikPolicyPlugin()],
)

WHEEL_URL = "https://storage.googleapis.com/{bucket}/packages/rubrik_agent_cloud_policy_plugin-0.1.1-py3-none-any.whl"

remote_app = agent_engines.create(
    agent_engine=adk_app,
    display_name="my-agent",
    requirements=[
        f"rubrik-agent-cloud-policy-plugin[adk] @ {WHEEL_URL}",
        "google-cloud-aiplatform>=1.142.0",
        "google-genai>=1.51.0",
    ],
    env_vars={
        "RUBRIK_WEBHOOK_URL": "https://webhooks.example.com/{tenant}/{webhook}/",
    },
)
```

**Important**: Do NOT set `GOOGLE_API_KEY` in `env_vars` — Agent Engine uses Vertex AI service account auth. Setting an API key conflicts with the session service (`Project/location and API key are mutually exclusive`).

### cloudpickle note

Agent Engine serializes plugins locally with `cloudpickle` and deserializes on the remote container. `RubrikPolicyClient` implements `__reduce__` so env vars like `RUBRIK_WEBHOOK_URL` are re-read on the remote side — not baked in from the local environment.

## Direct Client Usage (Framework-Agnostic)

```python
from rubrik_agent_cloud_policy import RubrikPolicyClient, ToolCall

client = RubrikPolicyClient(webhook_url="https://webhooks.example.com/tenant/webhook/")

result = await client.evaluate_tool_calls([
    ToolCall(name="get_weather", arguments='{"city": "London"}'),
    ToolCall(name="delete_file", arguments='{"path": "/etc/passwd"}'),
])

print(result.allowed)      # [ToolCall(name='get_weather', ...)]
print(result.blocked)      # [ToolCall(name='delete_file', ...)]
print(result.explanation)  # "Tool 'delete_file' blocked by Rubrik Agent Cloud"
```

## Development

```bash
cd policy-plugin
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -v   # 44 tests
```

## Architecture

```
rubrik_agent_cloud_policy/
  client.py            # Framework-agnostic core (webhook client + format translation)
  adk.py               # Google ADK BasePlugin adapter
  auto_instrument.py   # Monkey-patch Runner for zero-code instrumentation
  # Future:
  # langchain.py       # LangChain callback handler
  # crewai.py          # CrewAI adapter
```

The core client handles all webhook communication and OpenAI format translation. Framework adapters are thin wrappers that convert framework-specific tool representations to `ToolCall` objects and apply `PolicyResult` back.
