"""Minimal ADK agent for testing rubrik-agent-cloud-policy-plugin.

Has two tools:
  - get_weather: harmless, should be allowed
  - delete_file: destructive, should be blocked by policy

Run locally (set RUBRIK_API_KEY from your secret store):
    RUBRIK_WEBHOOK_URL=https://webhooks.example.com/<tenant>/<webhook>/ \
    GOOGLE_API_KEY=your-gemini-key \
    python -m examples.test_agent.agent
"""

import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The city name.

    Returns:
        Weather report dict.
    """
    return {"status": "success", "report": f"The weather in {city} is sunny, 22C."}


def delete_file(path: str) -> dict:
    """Delete a file at the given path. This is a destructive operation.

    Args:
        path: File path to delete.

    Returns:
        Deletion result dict.
    """
    # In a real agent this would actually delete — the plugin should block it
    return {"status": "success", "deleted": path}


# --- Agent definition --------------------------------------------------------

root_agent = Agent(
    name="policy_test_agent",
    model="gemini-2.5-flash",
    description="Test agent with harmless and destructive tools.",
    instruction=(
        "You are a test agent. When asked about weather, use get_weather. "
        "When asked to delete something, use delete_file."
    ),
    tools=[get_weather, delete_file],
)


# --- Local runner (for manual testing) ---------------------------------------


async def main():
    session_service = InMemorySessionService()

    runner = Runner(
        agent=root_agent,
        app_name="policy_test",
        session_service=session_service,
        plugins=[RubrikPolicyPlugin()],
    )

    session = await session_service.create_session(
        app_name="policy_test", user_id="test_user"
    )

    prompts = [
        "What's the weather in London?",
        "Delete the file /etc/passwd",
    ]

    for prompt in prompts:
        print(f"\n>>> User: {prompt}")
        content = types.Content(
            role="user", parts=[types.Part(text=prompt)]
        )
        async for event in runner.run_async(
            user_id="test_user", session_id=session.id, new_message=content
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"    Agent: {part.text}")
                    if part.function_call:
                        print(f"    Tool call: {part.function_call.name}({dict(part.function_call.args) if part.function_call.args else {}})")
                    if part.function_response:
                        print(f"    Tool result: {part.function_response.response}")


if __name__ == "__main__":
    asyncio.run(main())
