"""Deploy the test agent to Vertex AI Agent Engine.

Usage:
    # Authenticate first
    gcloud auth login
    gcloud auth application-default login

    # Store the webhook API key in Secret Manager and grant the Reasoning
    # Engine Service Agent access (see the README), then deploy:
    RUBRIK_WEBHOOK_URL=https://webhooks.your-dataplane.example.com/<tenant>/<webhook>/ \
    python -m examples.test_agent.deploy --api-key-secret rubrik-agent-cloud-webhook-key
"""

import argparse
import logging
import os
from importlib.metadata import version

import vertexai
from vertexai import agent_engines

from examples.test_agent.agent import root_agent
from rubrik_agent_cloud_policy.adk import RubrikPolicyPlugin

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEFAULT_PROJECT = "your-gcp-project"
DEFAULT_LOCATION = "us-central1"
DEFAULT_BUCKET = f"gs://{DEFAULT_PROJECT}-adk-staging"


def deploy(project: str, location: str, bucket: str, api_key_secret: str) -> str:
    webhook_url = os.environ.get("RUBRIK_WEBHOOK_URL", "")
    if not webhook_url:
        raise SystemExit(
            "RUBRIK_WEBHOOK_URL must be set. Example:\n"
            "  export RUBRIK_WEBHOOK_URL=https://webhook.your-dataplane.example.com"
        )

    vertexai.init(project=project, location=location, staging_bucket=bucket)

    adk_app = agent_engines.AdkApp(
        agent=root_agent,
        plugins=[RubrikPolicyPlugin()],
    )

    log.info("Deploying to %s/%s ...", project, location)

    # Do not set GOOGLE_API_KEY: Agent Runtime authenticates with its service
    # account, and an API key conflicts with the session service.
    env_vars = {
        "RUBRIK_WEBHOOK_URL": webhook_url,
        "RUBRIK_API_KEY": {"secret": api_key_secret, "version": "latest"},
    }
    # Pin the plugin to the version installed here, which is what was
    # pickled. Agent Runtime installs it from PyPI, so local changes must be
    # released before they can be deployed.
    plugin_version = version("rubrik-agent-cloud-policy-plugin")
    requirements = [
        "google-cloud-aiplatform[agent-engines,adk]>=1.142.0",
        f"rubrik-agent-cloud-policy-plugin[adk]=={plugin_version}",
    ]

    remote_app = agent_engines.create(
        agent_engine=adk_app,
        display_name="rubrik-agent-cloud-policy-test-agent",
        description="Test agent for validating rubrik-agent-cloud-policy-plugin tool blocking",
        requirements=requirements,
        env_vars=env_vars,
    )

    log.info("Deployed: %s", remote_app.resource_name)
    return remote_app.resource_name


def main():
    parser = argparse.ArgumentParser(description="Deploy test agent to Agent Engine")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument(
        "--api-key-secret",
        required=True,
        help="Secret Manager secret that holds the webhook API key",
    )
    args = parser.parse_args()

    resource_name = deploy(args.project, args.location, args.bucket, args.api_key_secret)
    print(f"\nAgent deployed: {resource_name}")
    print(f"View at: https://console.cloud.google.com/vertex-ai/agents/agent-engines?project={args.project}")


if __name__ == "__main__":
    main()
