# rubrik-agent-cloud-policy-plugin

Lightweight Python package for enforcing Rubrik Agent Cloud policies in AI agent frameworks. Framework-agnostic core with a Google ADK adapter.

## Install

```bash
pip install "rubrik-agent-cloud-policy-plugin[adk]"
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

> **Note**: `PYTHONSTARTUP` does not work — it only runs for interactive Python sessions, not for `python3 script.py` or Agent Runtime containers. Use `sitecustomize.py` instead.

### Option C: `adk web` and `adk api_server`

The ADK CLI loads extra plugins with the `--extra_plugins` flag:

```bash
adk api_server --extra_plugins rubrik_agent_cloud_policy.adk.RubrikPolicyPlugin path/to/agents
```

## Configuration

All options read configuration from environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `RUBRIK_WEBHOOK_URL` | Yes | Webhook URL from an Agent Cloud LiteLLM connection, `https://webhooks.{domain}/{tenant_short_code}/{webhook_uuid}/` |
| `RUBRIK_API_KEY` | Yes | API key from the same connection, sent as `Authorization: Bearer`. Supply it from a secret. |
| `RUBRIK_AGENT_ID` | No | Agent ID that Agent Cloud uses to apply policies scoped to specific agents. Defaults to `GOOGLE_CLOUD_AGENT_ENGINE_ID`, which Vertex AI Agent Runtime sets to the agent's reasoning engine ID. |
| `RUBRIK_POLICY_FAIL_OPEN` | No | `true` (default): allow activity when the webhook errors or times out (5 s). `false`: block it. |

The webhook allows only `POST` to its policy endpoints; `GET` requests, including health checks, return 403.

## How It Works

The plugin checks four points in each ADK run. Each check sends the conversation so far, the tools declared to the model, and the agent ID, so policies can evaluate a step in the context of earlier steps (for example, reading a secret and then publishing it).

| ADK callback | What is checked | Webhook endpoint | When blocked |
|--------------|-----------------|------------------|--------------|
| `on_user_message_callback` | User prompt, and tool results the client sends back (long-running tools) | `/v1/before_prompt/openai/v1` | A blocked prompt is saved to the session as a placeholder, and the run ends with the refusal before the model is called. A blocked tool result is replaced with the refusal. |
| `after_model_callback` | Model response: tool calls and text | `/v1/after_completion/openai/v1` | Blocked tool calls are removed before they run and an explanation is added. A blocked response is replaced with the refusal. |
| `before_tool_callback` | A tool call that `after_model_callback` did not check | `/v1/after_completion/openai/v1` | The tool does not run; the model receives an error result. |
| `after_tool_callback` | Tool result | `/v1/before_prompt/openai/v1` | The model receives an error result instead of the tool output. |

`before_tool_callback` is a safety net for tool calls that skip `after_model_callback`, which happens when another plugin's `before_model_callback` returns a cached response. Tool calls that `after_model_callback` already allowed are not checked again.

Streamed chunks are not checked one by one; the response that ADK acts on is.

ADK stops calling plugins for a callback once one returns a value, so a plugin earlier in the list could hide activity from this one. At the start of each run the plugin moves itself to the front of the runner's plugin list. This changes the order of the runner's other plugins: a plugin you placed first (for example, a response cache) runs after this one.

The webhook accepts request bodies up to 15 MiB. When the conversation is larger, the plugin drops the oldest messages from the check, keeping the system prompt and the turn being evaluated, and logs a warning.

Limitations:

- When another plugin's `before_model_callback` returns a cached response, ADK skips `after_model_callback`. Tool calls in that response are still checked by `before_tool_callback`; its text is not checked.
- With `StreamingMode.SSE`, ADK sends partial text to the caller before the final response is checked. A blocked response is replaced in the session and in the final event, but text already streamed to the user is not withdrawn. Tool calls are checked before they run in every mode.
- In live (bidirectional streaming) mode, ADK does not call plugin tool callbacks or `on_user_message_callback`, so only `after_model_callback` runs.
- Only text is checked in prompts. When a prompt is blocked, its images and files are dropped along with the text.

If a check fails (the webhook is unreachable, times out, or returns a malformed response), the plugin **fails open** by default. Set `RUBRIK_POLICY_FAIL_OPEN=false` to fail closed: prompts, tool calls, and tool results are then blocked while checks fail.

Misconfiguration is not a check failure. Invalid constructor arguments raise when the plugin is created, and a missing `RUBRIK_WEBHOOK_URL` raises `RuntimeError` at the first callback.

The webhook endpoints are the ones a LiteLLM proxy uses, so the credentials come from an Agent Cloud LiteLLM connection.

## Deploy to Vertex AI Agent Runtime

Store the API key in Secret Manager and let the Reasoning Engine Service Agent read it:

```bash
read -rs RUBRIK_API_KEY
printf '%s' "$RUBRIK_API_KEY" | gcloud secrets create rubrik-agent-cloud-webhook-key \
  --data-file=- --project=PROJECT_ID
unset RUBRIK_API_KEY

gcloud secrets add-iam-policy-binding rubrik-agent-cloud-webhook-key \
  --member="serviceAccount:service-PROJECT_NUMBER@gcp-sa-aiplatform-re.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" \
  --project=PROJECT_ID
```

Then deploy with the plugin and the environment variables:

```python
import vertexai
from vertexai import agent_engines
from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

vertexai.init(project="my-project", location="us-central1",
              staging_bucket="gs://my-bucket")

app = agent_engines.AdkApp(
    agent=my_agent,
    plugins=[RubrikPolicyPlugin()],
)

remote_app = agent_engines.create(
    agent_engine=app,
    display_name="my-agent",
    requirements=[
        "google-cloud-aiplatform[agent-engines,adk]>=1.142.0",
        # Match the version installed locally, which is what gets pickled.
        "rubrik-agent-cloud-policy-plugin[adk]==0.2.0",
    ],
    env_vars={
        "RUBRIK_WEBHOOK_URL": "https://webhooks.example.com/{tenant}/{webhook}/",
        "RUBRIK_API_KEY": {"secret": "rubrik-agent-cloud-webhook-key", "version": "latest"},
    },
)
```

Agent Runtime sets `GOOGLE_CLOUD_AGENT_ENGINE_ID` to the reasoning engine ID. That is the agent ID under which the Agent Cloud Vertex AI connection (including Gemini Enterprise agents) records the agent, so the plugin identifies the agent without further configuration. For an agent that the connection does not discover, set `RUBRIK_AGENT_ID`.

**Important**: Do NOT set `GOOGLE_API_KEY` in `env_vars` — Agent Runtime uses Vertex AI service account auth. Setting an API key conflicts with the session service (`Project/location and API key are mutually exclusive`).

### cloudpickle note

Agent Runtime serializes plugins locally with `cloudpickle` and deserializes on the remote container. `RubrikPolicyClient` implements `__reduce__` so env vars like `RUBRIK_WEBHOOK_URL` are re-read on the remote side — not baked in from the local environment.

## Direct Client Usage (Framework-Agnostic)

```python
from rubrik_agent_cloud_policy import PolicyContext, RubrikPolicyClient, ToolCall

client = RubrikPolicyClient(
    webhook_url="https://webhooks.example.com/tenant/webhook/",
    api_key="...",
    agent_id="my-agent",
)
context = PolicyContext(messages=({"role": "user", "content": "Clean up /etc"},))

result = await client.evaluate_response(
    [
        ToolCall(name="get_weather", arguments='{"city": "London"}'),
        ToolCall(name="delete_file", arguments='{"path": "/etc/passwd"}'),
    ],
    text="",
    context=context,
)
print(result.allowed)      # (ToolCall(name='get_weather', ...),)
print(result.blocked)      # (ToolCall(name='delete_file', ...),)
print(result.explanation)  # "The following tool calls were blocked by Rubrik Agent Cloud: ..."

refusal = await client.evaluate_prompt(context)  # None when allowed
```

## Development

```bash
cd policy-plugin
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest tests/ -v
```

## Architecture

```
rubrik_agent_cloud_policy/
  client.py            # Framework-agnostic core (webhook client + format translation)
  adk.py               # Google ADK BasePlugin adapter
  auto_instrument.py   # Monkey-patch Runner for zero-code instrumentation
```

The core client handles all webhook communication and OpenAI format translation. Framework adapters are thin wrappers that convert framework-specific messages and tool calls to OpenAI format and apply the result back.
